#pragma once
#include "worker.h"
#include <limits>

struct SharedSemanticsInputs {
  static constexpr size_t B=WB_SHARED_B,M=WB_SHARED_M,N=WB_SHARED_N,K=WB_SHARED_K;
  size_t output_size=0;
  std::vector<float> x,a,b,acc;
  std::vector<double> expected;

  explicit SharedSemanticsInputs(wb::InputSet &inputs) {
    if(WB_SHARED_KIND==5 || WB_SHARED_KIND==6) {
      a.resize(B*M*K);b.resize(B*K*N);acc.resize(B*M*N);
      inputs.fill(a);inputs.fill(b);inputs.fill(acc);
      inputs.load("A",a);inputs.load("B",b);inputs.load("Acc",acc);
      output_size=B*M*N;expected.resize(output_size);
      for(size_t batch=0;batch<B;++batch)for(size_t m=0;m<M;++m)for(size_t n=0;n<N;++n) {
        const size_t index=(batch*M+m)*N+n;
        double value=acc[index];
        for(size_t k=0;k<K;++k)value+=double(a[(batch*M+m)*K+k])*b[(batch*K+k)*N+n];
        expected[index]=value;
      }
    } else {
      x.resize(M*N);inputs.fill(x);inputs.load("X",x);
      output_size=(WB_SHARED_KIND==2 || WB_SHARED_KIND==3)?M:M*N;
      expected.resize(output_size);
      for(size_t m=0;m<M;++m) {
        double sum=0,maximum=-std::numeric_limits<double>::infinity();
        for(size_t n=0;n<N;++n){sum+=x[m*N+n];maximum=std::max(maximum,double(x[m*N+n]));}
        if(WB_SHARED_KIND==2)expected[m]=sum;
        else if(WB_SHARED_KIND==3)expected[m]=maximum;
        else if(WB_SHARED_KIND==4) {
          double denominator=0;
          for(size_t n=0;n<N;++n){expected[m*N+n]=std::exp(double(x[m*N+n])-maximum);denominator+=expected[m*N+n];}
          for(size_t n=0;n<N;++n)expected[m*N+n]/=denominator;
        } else for(size_t n=0;n<N;++n) {
          const size_t i=m*N+n;
          if(WB_SHARED_KIND==1)expected[i]=x[i]+float(n);
          else if(WB_SHARED_KIND==7)expected[i]=(x[i]+3.0f)*0.25f;
          else if(WB_SHARED_KIND==8)expected[i]=x[i]*1.25f-0.5f;
          else throw std::runtime_error("unknown shared numerical workload");
        }
      }
    }
    inputs.finish();
  }
  void check(const std::vector<float> &output) const {
    if(output.size()!=expected.size())throw std::runtime_error("shared numerical output shape mismatch");
    for(size_t i=0;i<output.size();++i)wb::close(output[i],expected[i],i);
  }
};
