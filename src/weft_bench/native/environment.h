#pragma once
#include <cerrno>
#include <sstream>
#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <unistd.h>
#if defined(__riscv)
#include <sys/auxv.h>
#include <sys/prctl.h>
#endif
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#if defined(__linux__)
#include <asm/prctl.h>
#endif
#endif

namespace wb {
inline std::string read_text(const std::string &path) {
  std::ifstream file(path);
  std::ostringstream value;
  value << file.rdbuf();
  auto text=value.str();
  while(!text.empty() && (text.back()=='\n'||text.back()=='\r'))text.pop_back();
  return text;
}
inline std::string status_field(const std::string &key) {
  std::ifstream file("/proc/self/status");std::string line;
  while(std::getline(file,line))if(line.rfind(key+":",0)==0) {
    auto start=line.find_first_not_of(" \t",key.size()+1);
    return start==std::string::npos?"":line.substr(start);
  }
  return "";
}
inline std::string xstate_json() {
  std::ostringstream out;out << "{";
#if defined(__x86_64__) || defined(__i386__)
  unsigned a=0,b=0,c=0,d=0;
  bool leaf1=__get_cpuid(1,&a,&b,&c,&d);
  bool osxsave=leaf1 && (c&(1u<<27));
  out << "\"osxsave\":" << (osxsave?"true":"false");
  if(osxsave) {
    unsigned lo,hi;
    __asm__ volatile("xgetbv" : "=a"(lo),"=d"(hi) : "c"(0));
    out << ",\"xcr0\":" << ((uint64_t(hi)<<32)|lo);
  }
#if defined(__linux__) && defined(ARCH_GET_XCOMP_SUPP) && defined(ARCH_GET_XCOMP_PERM)
  unsigned long long supported=0,permitted=0;
  errno=0;long support_rc=syscall(SYS_arch_prctl,ARCH_GET_XCOMP_SUPP,&supported);int support_errno=errno;
  errno=0;long permission_rc=syscall(SYS_arch_prctl,ARCH_GET_XCOMP_PERM,&permitted);int permission_errno=errno;
  out << ",\"xcomp_supported_mask\":" << supported << ",\"xcomp_support_errno\":" << (support_rc? support_errno:0)
      << ",\"xcomp_permitted_mask\":" << permitted << ",\"xcomp_permission_errno\":" << (permission_rc?permission_errno:0)
      << ",\"tile_data_permitted_in_this_thread\":" << (!permission_rc&&(permitted&(1ull<<18))?"true":"false");
#endif
#else
  out << "\"query\":\"not-x86\"";
#endif
  out << "}";return out.str();
}
inline std::string riscv_vector_json() {
  std::ostringstream out; out << "{";
#if defined(__linux__) && defined(__riscv)
  int control=prctl(70,0UL,0UL,0UL,0UL);
  bool enabled=(getauxval(AT_HWCAP)&(1UL<<('V'-'A'))) && control>=0 && (control&3)!=1;
  out << "\"linux_vector_enabled\":" << (enabled?"true":"false")
      << ",\"control\":" << control << ",\"cpus\":[";
  if(enabled) {
    cpu_set_t original;
    if(sched_getaffinity(0,sizeof(original),&original))throw std::runtime_error("cannot observe RVV affinity");
    bool first=true; int error=0; bool width_mismatch=false;
    for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&original)) {
      cpu_set_t one; CPU_ZERO(&one); CPU_SET(cpu,&one);
      if(sched_setaffinity(0,sizeof(one),&one)){error=errno;break;}
      unsigned long bytes=0; __asm__ volatile("csrr %0, 0xc22":"=r"(bytes));
#ifdef WB_EXPECTED_VLEN_BITS
      width_mismatch |= bytes*8 != WB_EXPECTED_VLEN_BITS;
#endif
      if(!first)out<<",";first=false;
      out<<"{\"cpu\":"<<cpu<<",\"vlen_bits\":"<<(bytes*8)<<"}";
    }
    if(sched_setaffinity(0,sizeof(original),&original))throw std::runtime_error("cannot restore RVV affinity");
    if(error)throw std::runtime_error("cannot select CPU for RVV observation: "+std::to_string(error));
    if(width_mismatch)throw std::runtime_error("configured VLEN differs from an executing CPU; select a homogeneous CPU set");
  }
#ifdef WB_EXPECTED_VLEN_BITS
  else throw std::runtime_error("configured RVV target has no Linux-enabled vector state");
#endif
  out << "]";
#else
  out << "\"query\":\"not-riscv\"";
#endif
  out << "}";return out.str();
}
inline void describe_environment() {
  std::ostringstream out;out << "{\"schema\":1,\"xstate\":" << xstate_json();
  out << ",\"riscv_vector\":" << riscv_vector_json();
#if defined(__linux__)
  utsname system{};if(uname(&system)==0)
    out << ",\"architecture\":" << json(system.machine) << ",\"kernel\":" << json(system.release);
  out << ",\"cpu_allowed_list\":" << json(status_field("Cpus_allowed_list"))
      << ",\"memory_allowed_list\":" << json(status_field("Mems_allowed_list"))
      << ",\"load_average\":" << json(read_text("/proc/loadavg"));
  cpu_set_t cpus;CPU_ZERO(&cpus);
  out << ",\"affinity\":[";bool first=true;
  if(sched_getaffinity(0,sizeof(cpus),&cpus)==0)for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&cpus)) {
    if(!first)out<<",";first=false;
    std::string base="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
    std::string frequency="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/cpufreq/";
    auto policy_value=[&](const char *name) {
      auto value=read_text(frequency+name);
      return value.empty()?std::string("null"):json(value);
    };
    out << "{\"cpu\":" << cpu << ",\"socket\":" << json(read_text(base+"physical_package_id"))
        << ",\"core\":" << json(read_text(base+"core_id"))
        << ",\"thread_siblings\":" << json(read_text(base+"thread_siblings_list"))
        << ",\"frequency_policy\":{\"driver\":" << policy_value("scaling_driver")
        << ",\"governor\":" << policy_value("scaling_governor")
        << ",\"energy_preference\":" << policy_value("energy_performance_preference")
        << ",\"minimum_khz\":" << policy_value("scaling_min_freq")
        << ",\"maximum_khz\":" << policy_value("scaling_max_freq") << "}}";
  }
  out << "]";
#ifdef SYS_get_mempolicy
  int policy=-1;unsigned long nodes[16]{};
  errno=0;long rc=syscall(SYS_get_mempolicy,&policy,nodes,sizeof(nodes)*8,nullptr,0);int error=errno;
  out << ",\"numa\":{\"policy\":" << policy << ",\"errno\":" << (rc?error:0) << ",\"nodes\":[";
  first=true;
  if(!rc)for(size_t node=0;node<sizeof(nodes)*8;++node)if(nodes[node/(sizeof(long)*8)]&(1ul<<(node%(sizeof(long)*8)))) {
    if(!first)out<<",";first=false;out<<node;
  }
  out << "]}";
#endif
  int current=sched_getcpu();out<<",\"current_cpu\":"<<current;
  std::ifstream cpuinfo("/proc/cpuinfo");std::string line,model,features;
  int processor=-1;
  while(std::getline(cpuinfo,line)) {
    auto colon=line.find(':');if(colon==std::string::npos)continue;
    auto key=line.substr(0,colon);auto value=line.substr(colon+1);
    if(key.find("processor")==0) {std::istringstream number(value);number>>processor;}
    if(processor!=current)continue;
    if(model.empty()&&(key.find("model name")==0||key.find("uarch")==0))model=value;
    if(features.empty()&&(key.find("flags")==0||key.find("isa")==0))features=value;
  }
  out << ",\"cpu_model\":" << json(model) << ",\"advertised_features\":" << json(features);
#endif
#if defined(__x86_64__) || defined(__i386__)
  unsigned a=0,b=0,c=0,d=0;out << ",\"cpuid\":[";bool first_cpuid=true;
  for(unsigned sub=0;sub<2;++sub)if(__get_cpuid_count(7,sub,&a,&b,&c,&d)) {
    if(!first_cpuid)out<<",";first_cpuid=false;
    out << "{\"leaf\":7,\"subleaf\":"<<sub<<",\"eax\":"<<a<<",\"ebx\":"<<b<<",\"ecx\":"<<c<<",\"edx\":"<<d<<"}";
  }
  out << "]";
#endif
  out << "}";std::printf("WB_ENVIRONMENT %s\n",out.str().c_str());
}
}
