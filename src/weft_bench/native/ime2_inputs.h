#pragma once
#include "worker.h"
#include <cstring>

// Native IME2 i8 ABI: block32 weights in N32 cohorts, activations in M1/M4
// cohorts. Preparation is outside timing for both providers. The independent
// oracle uses the original code arrays, never either implementation's output.
struct IME2Inputs {
  size_t M=WB_M,N=WB_N,K=WB_K,output_size=M*N;
  std::vector<uint8_t> weights,activation;
  std::vector<float> expected;
  explicit IME2Inputs(wb::InputSet &inputs) {
    if(M%WB_TILE_ROWS || N%32 || K%32) throw std::runtime_error("incomplete IME2 packed tile");
    std::vector<int8_t> w(N*K),x(M*K);
    for(auto *values:{&w,&x}) for(auto &v:*values) {
      inputs.seed=inputs.seed*1664525U+1013904223U;
      v=int8_t(int(inputs.seed>>24)-128);
    }
    inputs.load("W_codes",w); inputs.load("X_codes",x); inputs.finish();
    weights.resize(N*(K/32)*34); activation.resize(M*(K/32)*38); expected.resize(M*N);
    for(size_t n=0;n<N;++n) for(size_t b=0;b<K/32;++b) {
      auto *record=weights.data()+((n/32)*(K/32)+b)*32*34;
      const _Float16 scale=(_Float16)weight_scale(n,b);
      std::memcpy(record+(n%32)*2,&scale,2);
      std::memcpy(record+64+(n%32)*32,w.data()+n*K+b*32,32);
    }
    for(size_t m=0;m<M;++m) for(size_t b=0;b<K/32;++b) {
      auto *record=activation.data()+((m/WB_TILE_ROWS)*(K/32)+b)*WB_TILE_ROWS*38;
      float scale=activation_scale(m,b); int16_t negative=0;
      for(size_t k=0;k<32;++k) negative-=x[m*K+b*32+k];
      std::memcpy(record+(m%WB_TILE_ROWS)*4,&scale,4);
      std::memcpy(record+WB_TILE_ROWS*4+(m%WB_TILE_ROWS)*2,&negative,2);
      std::memcpy(record+WB_TILE_ROWS*6+(m%WB_TILE_ROWS)*32,x.data()+m*K+b*32,32);
    }
    for(size_t m=0;m<M;++m) for(size_t n=0;n<N;++n) {
      double sum=0;
      for(size_t b=0;b<K/32;++b) {
        int32_t dot=0;
        for(size_t k=0;k<32;++k) dot+=int32_t(x[m*K+b*32+k])*int32_t(w[n*K+b*32+k]);
        sum+=double(dot)*activation_scale(m,b)*weight_scale(n,b);
      }
      expected[m*N+n]=float(sum);
    }
  }
  static float weight_scale(size_t n,size_t b) { return float(1+(n+3*b)%7)/128; }
  static float activation_scale(size_t m,size_t b) { return float(1+(m+5*b)%11)/256; }
  void check(const std::vector<float> &y) const {
    if(y.size()!=expected.size()) throw std::runtime_error("IME2 output extent differs");
    for(size_t i=0;i<y.size();++i) wb::close(y[i],expected[i],i);
  }
};
