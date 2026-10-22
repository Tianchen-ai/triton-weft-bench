#pragma once
#include "worker.h"
#include <limits>

struct DenseInputs {
  const size_t M=WB_M,N=WB_N,K=WB_K,T=WB_T,H=WB_H,P=WB_P;
  const bool gemm=WB_KIND==4,rope=WB_KIND==3||WB_KIND==5;
  size_t output_size=rope?T*H*2*P:M*N;
  float epsilon,scale,scalar_bias;
  std::vector<float> input,weight,bias,cosine,sine;
  DenseInputs(wb::InputSet &inputs):epsilon(inputs.parameter("epsilon",1e-5f)),
    scale(inputs.parameter("scale",1.25f)),scalar_bias(inputs.parameter("bias",-.5f)),
    input(rope?output_size:gemm?M*K:output_size),weight(gemm?N*K:N),bias(N),cosine(T*P),sine(T*P) {
    inputs.fill(input);inputs.fill(weight);inputs.fill(bias);
    for(size_t i=0;i<T*P;++i){cosine[i]=std::cos(float(i)*.01f);sine[i]=std::sin(float(i)*.01f);}
    inputs.load("X",input);
    if(gemm)inputs.load("W",weight);
    if(WB_KIND==0||WB_KIND==1)inputs.load("Weight",weight);
    if(WB_KIND==1)inputs.load("Bias",bias);
    if(WB_KIND==3){inputs.load("Cos",cosine);inputs.load("Sin",sine);}
    if(WB_KIND==5)prepare_rope();
    inputs.finish();
  }
  void prepare_rope() {
    float step=std::pow(10000.0f,-1.0f/P);
    for(size_t t=0;t<T;++t){float theta=float(t);for(size_t p=0;p<P;++p){::sincosf(theta,&sine[t*P+p],&cosine[t*P+p]);theta*=step;}}
  }
  void check(const std::vector<float>& y) const {
    for(auto x:y)if(!std::isfinite(x))throw std::runtime_error("numeric nonfinite output");
    if(gemm) {
      for(size_t s=0;s<64;++s){size_t index=s*(output_size-1)/63,r=index/N,c=index%N;double v=0;
        for(size_t k=0;k<K;++k)v+=double(input[r*K+k])*weight[c*K+k];wb::close(y[index],v,index);}
    } else if(WB_KIND==6) {
      for(size_t i=0;i<output_size;++i)wb::close(y[i],double(input[i])*scale+scalar_bias,i);
    } else if(rope) {
      for(size_t t=0;t<T;++t)for(size_t h=0;h<H;++h)for(size_t p=0;p<P;++p){
        size_t i=(t*H+h)*2*P+p;double x=input[i],v=input[i+P],c=cosine[t*P+p],s=sine[t*P+p];
        wb::close(y[i],x*c-v*s,i);wb::close(y[i+P],x*s+v*c,i+P);}
    } else {
      for(size_t r=0;r<M;++r){double sum=0,squares=0,maximum=-INFINITY;
        for(size_t c=0;c<N;++c){double x=input[r*N+c];sum+=x;maximum=std::max(maximum,x);}
        double mean=WB_KIND==1?sum/N:0;
        for(size_t c=0;c<N;++c){double x=input[r*N+c]-mean;squares+=WB_KIND==2?std::exp(x-maximum):x*x;}
        for(size_t c=0;c<N;++c){double x=input[r*N+c];double v=WB_KIND==2?std::exp(x-maximum)/squares:(x-mean)/std::sqrt(squares/N+epsilon)*weight[c]+(WB_KIND==1?bias[c]:0);
          wb::close(y[r*N+c],v,r*N+c);}
      }
    }
  }
};
