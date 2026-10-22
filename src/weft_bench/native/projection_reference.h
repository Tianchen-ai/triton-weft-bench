#pragma once
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#if WB_PROJECTION_KEEP_F32
#include <dlfcn.h>
#endif
#ifndef WB_THREADS
#define WB_THREADS 1
#endif
#ifndef WB_GGML_MODEL_WEIGHTS
#define WB_GGML_MODEL_WEIGHTS 0
#endif

struct ProjectionReference {
  ggml_context *context;
  ggml_context *weight_context=nullptr;
  ggml_backend_t backend;
  ggml_backend_buffer_t buffer;
  ggml_backend_buffer_t weight_buffer=nullptr;
  std::string weight_buffer_name;
  ggml_cgraph *graph;
  ggml_backend_graph_plan_t plan;
  ggml_threadpool_t pool;
  ggml_tensor *result;
  std::vector<float> &output;
#if WB_PROJECTION_KEEP_F32
  const void *mixed_precision_entry=nullptr;
#endif
  ProjectionReference(ggml_type type,size_t m,size_t n,size_t k,const void *weights,const float *activation,std::vector<float> &out):output(out) {
    ggml_cpu_init();context=ggml_init({1024*1024,nullptr,true});
#if WB_GGML_MODEL_WEIGHTS
    weight_context=ggml_init({1024*1024,nullptr,true});
    if(!weight_context)throw std::runtime_error("GGML weight context allocation failed");
    auto *w=ggml_new_tensor_2d(weight_context,type,k,n);
#else
    auto *w=ggml_new_tensor_2d(context,type,k,n);
#endif
    auto *x=ggml_new_tensor_2d(context,GGML_TYPE_F32,k,m);
    result=ggml_mul_mat(context,w,x);graph=ggml_new_graph(context);ggml_build_forward_expand(graph,result);
    backend=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(backend,WB_THREADS);
    auto pool_params=ggml_threadpool_params_default(WB_THREADS);
    pool=ggml_threadpool_new(&pool_params);ggml_backend_cpu_set_threadpool(backend,pool);
#if WB_GGML_MODEL_WEIGHTS
    auto *device=ggml_backend_get_device(backend);
    auto *registry=ggml_backend_dev_backend_reg(device);
    auto extras=reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
      ggml_backend_reg_get_proc_address(registry,"ggml_backend_dev_get_extra_bufts"));
    std::vector<ggml_backend_buffer_type_t> candidates;
    if(extras)for(auto *entry=extras(device);entry&&*entry;++entry)candidates.push_back(*entry);
    candidates.push_back(ggml_backend_dev_buffer_type(device));
    ggml_backend_buffer_type_t selected=nullptr;
    for(auto candidate:candidates) {
      // Normal llama model loading probes the operation with a zero-size weight
      // buffer, then selects the first supported CPU extra/default buffer type.
      w->buffer=ggml_backend_buft_alloc_buffer(candidate,0);
      bool supported=w->buffer&&ggml_backend_dev_supports_op(device,result);
      ggml_backend_buffer_free(w->buffer);w->buffer=nullptr;
      if(supported){selected=candidate;break;}
    }
    if(!selected)throw std::runtime_error("no CPU model weight buffer supports this projection");
    weight_buffer=ggml_backend_alloc_ctx_tensors_from_buft(weight_context,selected);
    if(!weight_buffer)throw std::runtime_error("GGML model weight buffer allocation failed");
    ggml_backend_buffer_set_usage(weight_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    weight_buffer_name=ggml_backend_buft_name(selected);
#endif
    buffer=ggml_backend_alloc_ctx_tensors(context,backend);
    if(!buffer)throw std::runtime_error("GGML tensor allocation failed");
#if !WB_GGML_MODEL_WEIGHTS
    weight_buffer_name=ggml_backend_buffer_name(buffer);
#endif
    ggml_backend_tensor_set(w,weights,0,ggml_nbytes(w));
    ggml_backend_tensor_set(x,activation,0,ggml_nbytes(x));
    plan=ggml_backend_graph_plan_create(backend,graph);
    if(!plan)throw std::runtime_error("GGML graph plan creation failed");
#if WB_PROJECTION_KEEP_F32 && WB_GGML_AMX_F16_F32
    if(type==GGML_TYPE_F16 && weight_buffer_name=="AMX")
      mixed_precision_entry=dlsym(RTLD_DEFAULT,"_Z24ggml_backend_amx_mul_matPK19ggml_compute_paramsP11ggml_tensor");
#endif
  }
  void run(){
    if(ggml_backend_graph_plan_compute(backend,plan)!=GGML_STATUS_SUCCESS)throw std::runtime_error("GGML projection failed");
  }
  void collect(){ggml_backend_tensor_get(result,output.data(),0,ggml_nbytes(result));}
#if WB_PROJECTION_KEEP_F32
  void check_activation_precision() const {
    if(!mixed_precision_entry)throw std::runtime_error(
      "projection precision contract mismatch: F16 weights require preserved F32 activation; selected GGML buffer "
      +weight_buffer_name+" has no established F16/F32 compute path (rounding X to F16 is not equivalent)");
  }
#endif
  std::string dispatch_details() const {
    std::string fields=",\"weight_buffer_type\":"+wb::json(weight_buffer_name);
#if WB_GGML_MODEL_WEIGHTS
    fields+=",\"weights_policy\":\"model-default\",\"weight_preparation\":\"persistent tensor_set/repack before timing\",\"weight_buffer_usage\":\"WEIGHTS\",\"compute_observation\":\"GGML graph with selected CPU model weight buffer\"";
    fields+=",\"projection_rows\":"+std::to_string(result->ne[1])+",\"weight_type\":"+wb::json(ggml_type_name(result->src[0]->type));
#else
    fields+=",\"weights_policy\":\"plain\",\"weight_preparation\":\"plain weight upload before timing\"";
#endif
#if WB_PROJECTION_KEEP_F32
    Dl_info info{};if(mixed_precision_entry)dladdr(mixed_precision_entry,&info);
    fields+=",\"required_activation_precision\":\"f32-preserved\",\"activation_precision\":"
      +wb::json(mixed_precision_entry?"f32-preserved":"not-established")
      +",\"compute_path\":"+wb::json(mixed_precision_entry?"ggml_backend_amx_mul_mat/f16/tinygemm_kernel_avx<float,f16,float>":"not-established")
      +",\"resolved_compute_symbol\":"+wb::json(info.dli_sname?info.dli_sname:"")
      +",\"compute_dso\":"+wb::json(info.dli_fname?info.dli_fname:"")
      +",\"precision_evidence\":\"selected buffer trait plus inspected upstream source and resolved entry\"";
#endif
    return fields;
  }
  ~ProjectionReference(){ggml_backend_graph_plan_free(backend,plan);ggml_backend_buffer_free(buffer);if(weight_buffer)ggml_backend_buffer_free(weight_buffer);ggml_backend_free(backend);ggml_threadpool_free(pool);ggml_free(context);if(weight_context)ggml_free(weight_context);}
};
