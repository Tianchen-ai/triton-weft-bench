#pragma once
#include "timer.h"
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <functional>
#include <limits>
#include <type_traits>
#include <memory>
#include <optional>
#include <dlfcn.h>
#include <thread>
#if defined(__linux__) && defined(__AMX_TILE__)
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <asm/prctl.h>
#endif

namespace wb {
template<class T> void poison_output(std::vector<T> &output) {
  for (size_t i = 0; i < output.size(); ++i) {
    if constexpr (std::is_floating_point_v<T>)
      output[i] = std::numeric_limits<T>::quiet_NaN();
    else
      output[i] = static_cast<T>((i & 1) ? 0x5a : 0xa5);
  }
}
inline std::string json(const std::string &s) {
  std::string result="\"";
  for(unsigned char c:s) {
    if(c=='"'||c=='\\') {result+='\\';result+=c;}
    else if(c=='\n')result+="\\n";
    else if(c<32){char buf[7];std::snprintf(buf,sizeof(buf),"\\u%04x",c);result+=buf;}
    else result+=c;
  }
  return result+'"';
}
struct InputSet {
  uint32_t seed;
  std::map<std::string,std::string> files;
  std::map<std::string,double> parameters;
  std::set<std::string> consumed;
  explicit InputSet(uint32_t s):seed(s){}
  float random() {seed=seed*1664525U+1013904223U;return (float(seed>>8)/8388608.0f-1)*.5f;}
  template<class T> void fill(std::vector<T>& v) {for(auto &x:v)x=T(random());}
  void discard(size_t count) {
    // Advance the same uint32 LCG without generating values replaced by a
    // saved input. Later generated ports retain their original random stream.
    uint32_t multiplier=1664525U,increment=1013904223U,scale=1,offset=0;
    while(count) {
      if(count&1) {offset=offset*multiplier+increment;scale*=multiplier;}
      increment*=multiplier+1;multiplier*=multiplier;count>>=1;
    }
    seed=seed*scale+offset;
  }
  template<class T> bool load(const std::string &name,std::vector<T>& v) {
    auto item=files.find(name);if(item==files.end())return false;
    std::ifstream f(item->second,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streampos(v.size()*sizeof(T)))throw std::runtime_error("input byte size mismatch: "+name);
    f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(T));
    if(!f)throw std::runtime_error("cannot read input: "+name);
    consumed.insert(name);return true;
  }
  float parameter(const std::string &name,float fallback) const {
    auto it=parameters.find(name);return it==parameters.end()?fallback:float(it->second);
  }
  void finish() const {for(auto &p:files)if(!consumed.count(p.first))throw std::runtime_error("unknown input role: "+p.first);}
};
inline void close(float actual,double expected,size_t index,double atol=1e-4,double rtol=2e-3) {
  if(!std::isfinite(actual)||!std::isfinite(expected)||std::abs(actual-expected)>atol+rtol*std::abs(expected))
    throw std::runtime_error("numeric mismatch at output "+std::to_string(index)+": "+std::to_string(actual)+" vs "+std::to_string(expected));
}
template<class T> void save_buffer(const std::string &path,const std::vector<T> &values) {
  std::ofstream f(path,std::ios::binary);
  f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));
  if(!f)throw std::runtime_error("cannot save buffer: "+path);
}
struct Execution {
  virtual ~Execution() = default;
  virtual void prepare() = 0;
  virtual void sample(bool refresh) = 0;
  virtual void report() = 0;
};
// Libraries may cache AMX configuration initialization per thread. Preserve
// each implementation's hardware context at a batch boundary when alternating
// unrelated kernels in one worker. The restore is outside the timed interval.
struct TileConfiguration {
#if defined(__linux__) && defined(__AMX_TILE__)
  alignas(64) unsigned char bytes[64] = {};
  bool available = false;
  void capture() {
    unsigned long permitted = 0;
    available = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &permitted) == 0 &&
                (permitted & (1UL << 18));
    if (available) _tile_storeconfig(bytes);
  }
  void restore() { if (available) _tile_loadconfig(bytes); }
#else
  void capture() {}
  void restore() {}
#endif
};
template<class Run,class Collect,class Reset,class Check,class Initialize,class T>
struct NativeExecution : Execution {
  const char *name;
  Run run;
  Collect collect;
  Reset reset;
  Check check;
  Initialize initialize;
  std::vector<T>& output;
  bool stateful, save;
  const char *correctness;
  std::function<void()> activate;
  bool exported=false;
  bool finished=false;
  const char *stage="run";
  std::optional<Timing<Run,Reset>> timing;
  TileConfiguration tile_configuration;
  NativeExecution(const char* name,Run run,Collect collect,Reset reset,Check check,Initialize initialize,
                  std::vector<T>& output,bool stateful,bool save,const char* correctness,
                  std::function<void()> activate = {})
      :name(name),run(run),collect(collect),reset(reset),check(check),initialize(initialize),output(output),
       stateful(stateful),save(save),correctness(correctness),activate(std::move(activate)) {}
  void fail(const std::exception &e) {
    finished=true;
    std::printf("WB_RESULT {\"implementation\":%s,\"status\":%s,\"stage\":%s,\"error\":%s,\"outputs_available\":%s,\"samples_ms\":[]}\n",json(name).c_str(),json(std::string(stage)=="check"?"check_failed":"run_failed").c_str(),json(stage).c_str(),json(e.what()).c_str(),exported?"true":"false");
  }
  void prepare() override {
  try {
    if (activate) activate();
    initialize();
    reset();run();collect();
    if(save) {
      save_buffer(std::string(name)+"-Y.bin",output);
      exported=true;
    }
    stage="check";check();
    if(option("WB_CHECK_ONLY",0)!=0) {
      std::printf("WB_RESULT {\"implementation\":%s,\"status\":\"checked\",\"correctness\":%s,\"samples_ms\":[],\"invocations\":1,\"output_elements\":%zu",json(name).c_str(),json(correctness).c_str(),output.size());
      describe_output_storage(output.data(), output.size() * sizeof(T));
      std::printf("}\n");
      finished=true;
      return;
    }
    stage="measure";
    timing.emplace(run,reset,correctness,stateful,name,output.data(),output.size()*sizeof(T));
    timing->prepare();
    tile_configuration.capture();
  } catch(const std::exception &e) {
    fail(e);
  }
  }
  void sample(bool refresh) override {
    if (finished) return;
    try {
      tile_configuration.restore();
      if (activate) activate();
      timing->sample(refresh);
    } catch(const std::exception &e) { fail(e); }
  }
  void report() override {
    if (!finished) { timing->report(); finished=true; }
  }
};
struct ExecutionGroup {
  std::vector<std::unique_ptr<Execution>> executions;
  static void yield_idle_teams() {
    if (!option("WB_YIELD_IDLE_TEAMS",0)) return;
    // A competing runtime must not retain spinning workers across a sample
    // boundary. Each implementation warms its own team before timed sampling.
    if (auto release=reinterpret_cast<void(*)()>(dlsym(RTLD_DEFAULT,"weft_parallel_release"))) release();
    // libomp builds can expose an unsupported pause-resource stub. Let the
    // configured normal blocktime expire instead of changing either runtime.
    double settle=option("WB_IDLE_TEAM_SETTLE_MS",0);
    if (settle < 0) throw std::runtime_error("negative idle-team settle time");
    if (settle) std::this_thread::sleep_for(std::chrono::duration<double,std::milli>(settle));
  }
  template<class Run,class Collect,class Reset,class Check,class Initialize,class T>
  void add(const char* name,Run run,Collect collect,Reset reset,Check check,Initialize initialize,
           std::vector<T>& output,bool stateful,bool save,const char* correctness,
           std::function<void()> activate = {}) {
    executions.push_back(std::make_unique<NativeExecution<Run,Collect,Reset,Check,Initialize,T>>(
        name,run,collect,reset,check,initialize,output,stateful,save,correctness,std::move(activate)));
  }
  void run() {
    for (auto &execution:executions) { execution->prepare(); yield_idle_teams(); }
    const int samples=static_cast<int>(option("WB_SAMPLES",5));
    for (int round=0;round<samples;++round)
      for (std::size_t position=0;position<executions.size();++position) {
        std::size_t index=(round&1)?executions.size()-1-position:position;
        // Virtual dispatch is outside Timing::sample's measured interval.
        // Every native invocation still uses its concrete typed lambda.
        executions[index]->sample(true);
        yield_idle_teams();
      }
    for (auto &execution:executions) execution->report();
  }
};
template<class Run,class Collect,class Reset,class Check,class Initialize,class T>
void execute(const char* name,Run run,Collect collect,Reset reset,Check check,Initialize initialize,std::vector<T>& output,bool stateful,bool save,const char* correctness,std::function<void()> activate = {}) {
  NativeExecution<Run,Collect,Reset,Check,Initialize,T> execution(name,run,collect,reset,check,initialize,output,stateful,save,correctness,std::move(activate));
  execution.prepare();
  const int samples=static_cast<int>(option("WB_SAMPLES",5));
  for (int i=0;i<samples;++i) execution.sample(false);
  execution.report();
}
}
#include "environment.h"
