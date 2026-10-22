#pragma once
#include "worker.h"

struct VectorAddInputs {
  static constexpr size_t N=WB_VECTOR_N, SX=WB_VECTOR_SX, SZ=WB_VECTOR_SZ, SY=WB_VECTOR_SY;
  static constexpr size_t guard=WB_VECTOR_GUARD;
  size_t output_size=1+(N+guard)*SY;
  std::vector<float> x,z;
  std::vector<double> expected;

  explicit VectorAddInputs(wb::InputSet &inputs):x(1+(N+guard)*SX),z(1+(N+guard)*SZ),expected(output_size,0) {
    inputs.fill(x); inputs.fill(z);
    inputs.load("X",x); inputs.load("Z",z);
    for(size_t i=0;i<N;++i)expected[1+i*SY]=double(x[1+i*SX])+z[1+i*SZ];
    inputs.finish();
  }

  void check(const std::vector<float> &output) const {
    if(output.size()!=expected.size())throw std::runtime_error("vector output storage size mismatch");
    for(size_t i=0;i<output.size();++i) {
      const bool active=i>=1 && (i-1)%SY==0 && (i-1)/SY<N;
      if(active)wb::close(output[i],expected[i],i);
      else if(output[i]!=0)throw std::runtime_error("vector kernel wrote into masked or strided output padding at "+std::to_string(i));
    }
  }
};
