#pragma once
#include "worker.h"
#include <cblas.h>
#include <dlfcn.h>
#include <climits>

struct OpenBLASReference {
  const float *x,*w;float *y;int m,n,k;
  OpenBLASReference(const float *input,const float *weight,float *output,size_t rows,size_t columns,size_t reduction,int threads):
      x(input),w(weight),y(output),m(int(rows)),n(int(columns)),k(int(reduction)) {
    static_assert(sizeof(blasint)==4,"this adapter uses the LP64 CBLAS ABI");
    if(rows>INT_MAX||columns>INT_MAX||reduction>INT_MAX)throw std::runtime_error("CBLAS dimensions exceed LP64 range");
    openblas_set_num_threads(threads);
  }
  void run() {cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,n,k,1.0f,x,k,w,k,0.0f,y,n);}
  void describe(const char *name) const {
    Dl_info library{};dladdr(reinterpret_cast<const void*>(&cblas_sgemm),&library);
    std::printf("WB_DETAILS {\"implementation\":%s,\"reference_dispatch\":{\"library\":\"OpenBLAS\",\"entry\":\"cblas_sgemm\",\"core\":%s,\"configuration\":%s,\"configured_threads\":%d,\"dso\":%s,\"weights\":\"plain-transposed-view\"}}\n",
      wb::json(name).c_str(),wb::json(openblas_get_corename()).c_str(),wb::json(openblas_get_config()).c_str(),
      openblas_get_num_threads(),wb::json(library.dli_fname?library.dli_fname:"").c_str());
  }
};
