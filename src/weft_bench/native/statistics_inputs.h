#pragma once
#include "worker.h"
#include <algorithm>
#include <cmath>

struct StatisticsInputs {
  const size_t M=WB_M,N=WB_N;
  size_t output_size=3*M;
  std::vector<float> input;
  std::vector<long double> expected;
  explicit StatisticsInputs(wb::InputSet &inputs):input(M*N),expected(3*M,0) {
    const float offset=inputs.parameter("offset",0),scale=inputs.parameter("scale",1);
    inputs.fill(input);
    for(auto &v:input)v=v*scale+offset;
    inputs.load("X",input);inputs.finish();
    for(size_t row=0;row<M;++row){
      long double mean=0,m2=0;
      for(size_t j=0;j<N;++j){const long double value=input[row*N+j],delta=value-mean;
        mean+=delta/(j+1);m2+=delta*(value-mean);}
      expected[3*row]=N;expected[3*row+1]=mean;expected[3*row+2]=m2;
    }
  }
  void check(const std::vector<float> &out)const {
    for(size_t i=0;i<out.size();++i){
      if(!std::isfinite(out[i]))throw std::runtime_error("statistics nonfinite output");
      const long double allowed=i%3==0?0:1e-5L+1e-4L*std::abs(expected[i]);
      if(std::abs((long double)out[i]-expected[i])>allowed)
        throw std::runtime_error("statistics mismatch at "+std::to_string(i)+": "+std::to_string(out[i])+" vs "+std::to_string((double)expected[i]));
    }
  }
};

inline void wb_welford_sequential(const float *x,float *y,size_t M,size_t N){
  for(size_t row=0;row<M;++row){float mean=0,m2=0;
    for(size_t i=0;i<N;++i){const float value=x[row*N+i],delta=value-mean;
      mean+=delta/float(i+1);m2+=delta*(value-mean);}
    y[3*row]=float(N);y[3*row+1]=mean;y[3*row+2]=m2;
  }
}

inline void wb_welford_blocked(const float *x,float *y,size_t M,size_t N,size_t block,int threads=1){
  #pragma omp parallel for num_threads(threads) if(threads>1)
  for(size_t row=0;row<M;++row){float mean=0,m2=0;size_t count=0;
    for(size_t start=0;start<N;start+=block){const size_t b=std::min(block,N-start);float bm=0,b2=0;
      #pragma omp simd reduction(+:bm)
      for(size_t j=0;j<b;++j)bm+=x[row*N+start+j];
      bm/=float(b);
      #pragma omp simd reduction(+:b2)
      for(size_t j=0;j<b;++j){const float d=x[row*N+start+j]-bm;b2+=d*d;}
      const size_t total=count+b;const float delta=bm-mean,fraction=float(b)/float(total);
      mean+=delta*fraction;m2+=b2+delta*delta*(float(count)*fraction);count=total;
    }
    y[3*row]=float(count);y[3*row+1]=mean;y[3*row+2]=m2;
  }
}
