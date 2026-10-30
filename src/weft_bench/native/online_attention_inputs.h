#pragma once
#include "worker.h"
#include <algorithm>
#include <cmath>
struct OnlineAttentionInputs{
  size_t tq=WB_Tq,tk=WB_Tk,d=WB_D,output_size=tq*d;
  float scale;
  std::vector<float> query,key,value;
  OnlineAttentionInputs(wb::InputSet &inputs):scale(inputs.parameter("scale",1/std::sqrt(float(d)))),
    query(tq*d),key(tk*d),value(tk*d){
    inputs.fill(query);inputs.fill(key);inputs.fill(value);
    inputs.load("Q",query);inputs.load("K",key);inputs.load("V",value);inputs.finish();
  }
  void check(const std::vector<float>&y)const{
    for(size_t q=0;q<tq;++q){
      std::vector<double> scores(tk);double maximum=-INFINITY,den=0;
      for(size_t k=0;k<tk;++k){double v=0;for(size_t i=0;i<d;++i)v+=double(query[q*d+i])*key[k*d+i];scores[k]=v*scale;maximum=std::max(maximum,scores[k]);}
      for(auto &v:scores){v=std::exp(v-maximum);den+=v;}
      for(size_t i=0;i<d;++i){double result=0;for(size_t k=0;k<tk;++k)result+=scores[k]*value[k*d+i];result/=den;
        if(!std::isfinite(y[q*d+i])||std::abs(y[q*d+i]-result)>2e-5+2e-4*std::abs(result))
          throw std::runtime_error("attention mismatch at "+std::to_string(q*d+i));}
    }
  }
};
inline void attention_softmax(float *p,size_t rows,size_t cols){
  for(size_t i=0;i<rows;++i){float maximum=-INFINITY,total=0;
    for(size_t j=0;j<cols;++j)maximum=std::max(maximum,p[i*cols+j]);
    for(size_t j=0;j<cols;++j){p[i*cols+j]=std::exp(p[i*cols+j]-maximum);total+=p[i*cols+j];}
    #pragma omp simd
    for(size_t j=0;j<cols;++j)p[i*cols+j]/=total;
  }
}
inline void attention_materialized(const OnlineAttentionInputs &a,float *scores,float *y){
  for(size_t q=0;q<a.tq;++q)for(size_t k=0;k<a.tk;++k){float sum=0;
    #pragma omp simd reduction(+:sum)
    for(size_t i=0;i<a.d;++i)sum+=a.query[q*a.d+i]*a.key[k*a.d+i];
    scores[q*a.tk+k]=sum*a.scale;}
  attention_softmax(scores,a.tq,a.tk);std::fill(y,y+a.output_size,0);
  for(size_t q=0;q<a.tq;++q)for(size_t k=0;k<a.tk;++k){
    float weight=scores[q*a.tk+k];
    #pragma omp simd
    for(size_t i=0;i<a.d;++i)y[q*a.d+i]+=weight*a.value[k*a.d+i];
  }
}
