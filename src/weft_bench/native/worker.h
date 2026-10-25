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
  bool exported=false;
  bool finished=false;
  const char *stage="run";
  std::optional<Timing<Run,Reset>> timing;
  NativeExecution(const char* name,Run run,Collect collect,Reset reset,Check check,Initialize initialize,
                  std::vector<T>& output,bool stateful,bool save,const char* correctness)
      :name(name),run(run),collect(collect),reset(reset),check(check),initialize(initialize),output(output),
       stateful(stateful),save(save),correctness(correctness) {}
  void fail(const std::exception &e) {
    finished=true;
    std::printf("WB_RESULT {\"implementation\":%s,\"status\":%s,\"stage\":%s,\"error\":%s,\"outputs_available\":%s,\"samples_ms\":[]}\n",json(name).c_str(),json(std::string(stage)=="check"?"check_failed":"run_failed").c_str(),json(stage).c_str(),json(e.what()).c_str(),exported?"true":"false");
  }
  void prepare() override {
  try {
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
  } catch(const std::exception &e) {
    fail(e);
  }
  }
  void sample(bool refresh) override {
    if (finished) return;
    try { timing->sample(refresh); } catch(const std::exception &e) { fail(e); }
  }
  void report() override {
    if (!finished) { timing->report(); finished=true; }
  }
};
struct ExecutionGroup {
  std::vector<std::unique_ptr<Execution>> executions;
  template<class Run,class Collect,class Reset,class Check,class Initialize,class T>
  void add(const char* name,Run run,Collect collect,Reset reset,Check check,Initialize initialize,
           std::vector<T>& output,bool stateful,bool save,const char* correctness) {
    executions.push_back(std::make_unique<NativeExecution<Run,Collect,Reset,Check,Initialize,T>>(
        name,run,collect,reset,check,initialize,output,stateful,save,correctness));
  }
  void run() {
    for (auto &execution:executions) execution->prepare();
    const int samples=static_cast<int>(option("WB_SAMPLES",5));
    for (int round=0;round<samples;++round)
      for (std::size_t position=0;position<executions.size();++position) {
        std::size_t index=(round&1)?executions.size()-1-position:position;
        // Virtual dispatch is outside Timing::sample's measured interval.
        // Every native invocation still uses its concrete typed lambda.
        executions[index]->sample(true);
      }
    for (auto &execution:executions) execution->report();
  }
};
template<class Run,class Collect,class Reset,class Check,class Initialize,class T>
void execute(const char* name,Run run,Collect collect,Reset reset,Check check,Initialize initialize,std::vector<T>& output,bool stateful,bool save,const char* correctness) {
  NativeExecution<Run,Collect,Reset,Check,Initialize,T> execution(name,run,collect,reset,check,initialize,output,stateful,save,correctness);
  execution.prepare();
  const int samples=static_cast<int>(option("WB_SAMPLES",5));
  for (int i=0;i<samples;++i) execution.sample(false);
  execution.report();
}
}
#include "environment.h"
