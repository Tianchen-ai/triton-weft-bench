#pragma once
#include "projection_inputs.h"

struct AMXTileInputs : ProjectionInputs {
  size_t ldc = WB_LDC;
  explicit AMXTileInputs(wb::InputSet &inputs) : ProjectionInputs(inputs, true, true) {
    if((WB_AMX_DECODE ? (N<=0 || N%64 || M!=1) : (N<=0 || N%32 || M<1)) || K<=0 || K%WB_AMX_BLOCK_SIZE)
      throw std::runtime_error("unsupported native AMX row-tile geometry");
    if(ldc<N || (WB_AMX_DECODE && ldc!=N)) throw std::runtime_error("invalid AMX output stride");
    output_size=M*ldc;
  }
  size_t offset(const std::vector<float> &y) const {
    return ldc==N ? 0 : ((64-reinterpret_cast<uintptr_t>(y.data())%64)%64)/sizeof(float);
  }
  float *output(std::vector<float> &y) const { return y.data()+offset(y); }
  void check(const std::vector<float> &y) const {
    const size_t first=offset(y);
    for(size_t m=0;m<M;++m) for(size_t n=0;n<N;++n)
      wb::close(y[first+m*ldc+n],expected[m*N+n],m*N+n);
    for(size_t i=0;i<y.size();++i)
      if((i<first || i>=first+M*ldc || (i-first)%ldc>=N) && y[i]!=0)
        throw std::runtime_error("AMX tile changed output padding");
  }
};
