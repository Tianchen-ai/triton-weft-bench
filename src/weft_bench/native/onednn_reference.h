#pragma once
#include "worker.h"
#include <oneapi/dnnl/dnnl.hpp>
#include <dlfcn.h>
#include <unordered_map>
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#include <omp.h>
#endif

struct OneDNNReference {
  dnnl::engine engine{dnnl::engine::kind::cpu,0};
  dnnl::stream stream{engine};
  dnnl::matmul::primitive_desc descriptor;
  dnnl::matmul operation;
  std::unordered_map<int,dnnl::memory> arguments;
  int thread_limit;
  OneDNNReference(void *input,void *weight,void *output,int64_t m,int64_t n,int64_t k,int kind,int threads):thread_limit(threads) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    omp_set_dynamic(0);omp_set_num_threads(threads);thread_limit=omp_get_max_threads();
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
    if(threads!=1)throw std::runtime_error("sequential oneDNN only supports one thread");
#else
    throw std::runtime_error("configure oneDNN with the OMP or sequential CPU runtime");
#endif
    using type=dnnl::memory::data_type;
    auto source_type=kind==0?type::f32:kind==1?type::bf16:type::s8;
    auto output_type=kind==2?type::s32:type::f32;
    auto src=dnnl::memory::desc({m,k},source_type,dnnl::memory::dims{k,1});
    auto weights=dnnl::memory::desc({k,n},source_type,dnnl::memory::dims{1,k});
    auto dst=dnnl::memory::desc({m,n},output_type,dnnl::memory::dims{n,1});
    dnnl::primitive_attr attributes;
    attributes.set_fpmath_mode(dnnl::fpmath_mode::strict);
    attributes.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    descriptor=dnnl::matmul::primitive_desc(engine,src,weights,dst,attributes);
    operation=dnnl::matmul(descriptor);
    arguments.emplace(DNNL_ARG_SRC,dnnl::memory(src,engine,input));
    arguments.emplace(DNNL_ARG_WEIGHTS,dnnl::memory(weights,engine,weight));
    arguments.emplace(DNNL_ARG_DST,dnnl::memory(dst,engine,output));
    arguments.emplace(DNNL_ARG_SCRATCHPAD,dnnl::memory(descriptor.scratchpad_desc(),engine));
  }
  void run(){operation.execute(stream,arguments);stream.wait();}
  void describe(const char *name) const {
    const auto *version=dnnl_version();std::string implementation=descriptor.impl_info_str();
    Dl_info library{};dladdr(reinterpret_cast<const void*>(&dnnl_version),&library);
    std::printf("WB_DETAILS {\"implementation\":%s,\"reference_dispatch\":{\"library\":\"oneDNN\",\"version\":\"%d.%d.%d\",\"primitive\":%s,\"amx_implementation\":%s,\"thread_limit\":%d,\"cpu_runtime\":%d,\"effective_cpu_isa\":%d,\"fpmath\":\"strict\",\"weights\":\"plain-transposed-view\",\"external_reorder\":false,\"dso\":%s,\"post_execute_xstate\":%s}}\n",
      wb::json(name).c_str(),version->major,version->minor,version->patch,wb::json(implementation).c_str(),
      implementation.find("amx")!=std::string::npos?"true":"false",thread_limit,DNNL_CPU_THREADING_RUNTIME,
      int(dnnl_get_effective_cpu_isa()),wb::json(library.dli_fname?library.dli_fname:"").c_str(),wb::xstate_json().c_str());
  }
};
