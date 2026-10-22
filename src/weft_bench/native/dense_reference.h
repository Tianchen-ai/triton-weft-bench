#pragma once
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ops.h"
#include "binary-ops.h"

extern "C" void wb_ggml_rotate(size_t, size_t, size_t, const float *, const float *, float *);
#ifndef WB_THREADS
#define WB_THREADS 1
#endif

struct DenseReference {
  ggml_context *ctx = nullptr;
  ggml_tensor *x=nullptr, *w=nullptr, *b=nullptr, *first=nullptr, *second=nullptr, *y=nullptr;
  ggml_cgraph *graph=nullptr;
  ggml_threadpool *pool=nullptr;
  ggml_cplan plan{};
  ggml_compute_params params{};
  std::vector<float> intermediate1, intermediate2, cache;
  std::vector<int32_t> positions;
  std::vector<uint8_t> workspace;

  DenseReference(std::vector<float> &input, std::vector<float> &weight, std::vector<float> &bias,
                 std::vector<float> &cosine, std::vector<float> &sine, std::vector<float> &output,
                 float epsilon=1e-5f,float scale=1.25f,float scalar_bias=-.5f) {
    ggml_cpu_init();
    ctx=ggml_init({1024*1024,nullptr,true});
    if (!ctx) throw std::runtime_error("GGML context allocation failed");
    params.ith=0;params.nth=1;params.use_ref=false;
#if WB_KIND == 3
    cache.resize(WB_T*WB_P*2);
    for(size_t i=0;i<WB_T*WB_P;++i) { cache[2*i]=cosine[i];cache[2*i+1]=sine[i]; }
#elif WB_KIND == 5
    x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,WB_P*2,WB_H,WB_T);x->data=input.data();
    positions.resize(WB_T);for(size_t i=0;i<positions.size();++i)positions[i]=int32_t(i);
    auto *pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,WB_T);pos->data=positions.data();
    y=ggml_rope(ctx,x,pos,WB_P*2,GGML_ROPE_TYPE_NEOX);y->data=output.data();
    workspace.resize((WB_P*2+64)*sizeof(float));
#else
    x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,WB_KIND==4 ? WB_K : WB_N,WB_M);x->data=input.data();
    w=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,WB_KIND==4 ? WB_K : WB_N,WB_KIND==4 ? WB_N : 1);w->data=weight.data();
    b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,WB_N);b->data=bias.data();
    if (WB_KIND==0) {first=ggml_rms_norm(ctx,x,epsilon);y=ggml_mul(ctx,first,w);}
    else if(WB_KIND==1) {
      first=ggml_norm(ctx,x,epsilon);second=ggml_mul(ctx,first,w);y=ggml_add(ctx,second,b);
      intermediate1.resize(WB_M*WB_N);intermediate2.resize(WB_M*WB_N);
      first->data=intermediate1.data();second->data=intermediate2.data();
    }
    else if(WB_KIND==2) {y=ggml_soft_max(ctx,x);workspace.resize((WB_N+64)*sizeof(float));}
#if WB_KIND == 6
    else {y=ggml_scale_bias(ctx,x,scale,scalar_bias);}
#else
    else {y=ggml_mul_mat(ctx,w,x);}
#endif
    y->data=output.data();
    if(WB_KIND==4) {
      graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
      auto config=ggml_threadpool_params_default(WB_THREADS);pool=ggml_threadpool_new(&config);
      plan=ggml_graph_plan(graph,WB_THREADS,pool);workspace.resize(plan.work_size);plan.work_data=workspace.data();
    }
#endif
    params.wsize=workspace.size();params.wdata=workspace.data();
  }
  void run(std::vector<float> &input,std::vector<float> &output) {
#if WB_KIND == 3
    wb_ggml_rotate(WB_T,WB_H,WB_P,cache.data(),input.data(),output.data());
#else
    if(WB_KIND==0)ggml_compute_forward_rms_norm_mul_fused(&params,first,y);
    else if(WB_KIND==1){ggml_compute_forward_norm(&params,first);ggml_compute_forward_mul(&params,second);ggml_compute_forward_add(&params,y);}
    else if(WB_KIND==2)ggml_compute_forward_soft_max(&params,y);
    else if(WB_KIND==5)ggml_compute_forward_rope(&params,y);
#if WB_KIND == 6
    else ggml_compute_forward_scale(&params,y);
#else
    else if(ggml_graph_compute(graph,&plan)!=GGML_STATUS_SUCCESS)throw std::runtime_error("GGML graph execution failed");
#endif
#endif
  }
  ~DenseReference(){if(pool)ggml_threadpool_free(pool);if(ctx)ggml_free(ctx);}
};
