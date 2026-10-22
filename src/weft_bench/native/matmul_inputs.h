#pragma once
#include "worker.h"
#include <cstring>
#include <limits>

namespace wb {
inline uint16_t bf16_encode(float value) {
  uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));
  if((bits&0x7f800000u)==0x7f800000u)
    return uint16_t(bits>>16)|((bits&0x007fffffu)?0x40u:0u);
  return uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
}
inline float bf16_decode(uint16_t value) {
  uint32_t bits=uint32_t(value)<<16;float result;std::memcpy(&result,&bits,sizeof(result));return result;
}
}

struct MatmulInputs {
#if WB_MATMUL_INTEGER
  using Input=int8_t;using Output=int32_t;
#else
  using Input=uint16_t;using Output=float;
#endif
  const size_t M=WB_M,N=WB_N,K=WB_K,output_size=M*N;
  std::vector<Input> input,weight;
  explicit MatmulInputs(wb::InputSet &inputs):input(M*K),weight(N*K) {
    for(auto *values:{&input,&weight})for(auto &value:*values) {
#if WB_MATMUL_INTEGER
      inputs.seed=inputs.seed*1664525U+1013904223U;
      value=Input(int(inputs.seed>>24)-128);
#else
      value=wb::bf16_encode(inputs.random());
#endif
    }
#if WB_MATMUL_INTEGER
    const int8_t edges[]={-128,127,0,-1,1,-127,126};
    for(auto *values:{&input,&weight})for(size_t i=0;i<std::min(values->size(),size_t(7));++i)(*values)[i]=edges[i];
#endif
    inputs.load("X",input);inputs.load("W",weight);inputs.finish();
  }
  void check(const std::vector<Output> &output) const {
#if !WB_MATMUL_INTEGER
    for(auto value:output)if(!std::isfinite(value))throw std::runtime_error("numeric nonfinite output");
#endif
    const size_t checks=output_size<=4096?output_size:64;
    for(size_t sample=0;sample<checks;++sample) {
      size_t index=checks==output_size?sample:sample*(output_size-1)/63,row=index/N,column=index%N;
#if WB_MATMUL_INTEGER
      int64_t expected=0;
      for(size_t k=0;k<K;++k)expected+=int64_t(input[row*K+k])*int64_t(weight[column*K+k]);
      if(expected<std::numeric_limits<int32_t>::min()||expected>std::numeric_limits<int32_t>::max())
        throw std::runtime_error("integer matmul oracle exceeds int32 output range");
      if(output[index]!=expected)throw std::runtime_error("numeric mismatch at output "+std::to_string(index));
#else
      double expected=0;
      for(size_t k=0;k<K;++k)expected+=double(wb::bf16_decode(input[row*K+k]))*wb::bf16_decode(weight[column*K+k]);
      wb::close(output[index],expected,index);
#endif
    }
  }
};
