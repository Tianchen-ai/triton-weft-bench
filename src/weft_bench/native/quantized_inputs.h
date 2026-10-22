#pragma once
#include "worker.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "quants.h"
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

inline void wb_quantize_reference(ggml_type type,const float* x,void* y,size_t n) {
  if(type==GGML_TYPE_Q8_K)quantize_row_q8_K_ref(x,static_cast<block_q8_K*>(y),n);
  else {auto fn=ggml_get_type_traits(type)->from_float_ref;if(!fn)throw std::runtime_error("no quantization oracle");fn(x,y,n);}
}
inline void wb_dequantize_reference(ggml_type type,const void* x,float* y,size_t n) {
  if(type==GGML_TYPE_Q8_K)dequantize_row_q8_K(static_cast<const block_q8_K*>(x),y,n);
  else if(type==GGML_TYPE_Q8_1) {
    auto *blocks=static_cast<const block_q8_1*>(x);
    for(size_t i=0;i<n;++i)y[i]=ggml_fp16_to_fp32(blocks[i/32].d)*blocks[i/32].qs[i%32];
  } else {auto fn=ggml_get_type_traits(type)->to_float;if(!fn)throw std::runtime_error("no dequantization oracle");fn(x,y,n);}
}

// Legacy callers use the RVV extrema/tie contract. An explicitly selected
// first-absolute-max contract models GGML quantizers that use that storage rule.
// Both oracles calculate the bytes here, independently of the timed function.
#ifndef WB_Q8K_FIRST_EXTREME
#define WB_Q8K_FIRST_EXTREME 0
#endif
inline void wb_quantize_contract(ggml_type type,const float *x,uint8_t *y,size_t n) {
  const size_t block=type==GGML_TYPE_Q8_K?256:32,bytes=ggml_type_size(type);
  for(size_t b=0;b<n/block;++b){const float *values=x+b*block;uint8_t *out=y+b*bytes;
    float hi=-INFINITY,lo=INFINITY;
    for(size_t i=0;i<block;++i){hi=std::max(hi,values[i]);lo=std::min(lo,values[i]);}
    float extreme=std::fabs(hi)>std::fabs(lo)?hi:lo;
    if(WB_Q8K_FIRST_EXTREME && type==GGML_TYPE_Q8_K) {
      extreme=0;
      for(size_t i=0;i<block;++i)if(std::fabs(values[i])>std::fabs(extreme))extreme=values[i];
    }
    float inverse=extreme==0?0:-127.0f/extreme;
    float scale=type==GGML_TYPE_Q8_K?(inverse==0?0:1.0f/inverse):std::fabs(extreme)/127.0f;
    if(type!=GGML_TYPE_Q8_K)inverse=scale==0?0:1.0f/scale;
    size_t offset=type==GGML_TYPE_Q8_K?4:type==GGML_TYPE_Q8_1?4:2;
    if(type==GGML_TYPE_Q8_K)std::memcpy(out,&scale,4);
    else {auto half=ggml_fp32_to_fp16(scale);std::memcpy(out,&half,2);}
    auto *q=reinterpret_cast<int8_t*>(out+offset);int sum=0;
    for(size_t i=0;i<block;++i){q[i]=int8_t(std::max(-128.0f,std::min(127.0f,std::nearbyint(values[i]*inverse))));sum+=q[i];}
    if(type==GGML_TYPE_Q8_1){auto half=ggml_fp32_to_fp16(float(sum)*scale);std::memcpy(out+2,&half,2);}
    if(type==GGML_TYPE_Q8_K)for(size_t g=0;g<16;++g){int16_t s=0;for(size_t i=0;i<16;++i)s+=q[g*16+i];std::memcpy(out+260+2*g,&s,2);}
  }
}

// Q8_1 stores a rounded half-precision activation sum. Affine vec-dot uses
// that stored s, not a fresh sum of dequantized activations (GGML's Q4_1/Q5_1
// native vec-dot contract). Keep an independent scalar oracle for these fields.
template<class Weight>
inline double wb_affine_dot_fields(const uint8_t *weights,const uint8_t *activation,size_t n) {
  double sum=0;
  for(size_t b=0;b<n/32;++b) {
    Weight w;block_q8_1 x;
    std::memcpy(&w,weights+b*sizeof(w),sizeof(w));
    std::memcpy(&x,activation+b*sizeof(x),sizeof(x));
    int dot=0;
    for(size_t k=0;k<32;++k) {
      int q=(w.qs[k%16]>>(4*(k/16)))&15;
      if constexpr(sizeof(Weight)==sizeof(block_q5_1))q|=((w.qh[k/8]>>(k%8))&1)<<4;
      dot+=q*int(x.qs[k]);
    }
    sum+=double(ggml_fp16_to_fp32(w.d))*ggml_fp16_to_fp32(x.d)*dot
         +double(ggml_fp16_to_fp32(w.m))*ggml_fp16_to_fp32(x.s);
  }
  return sum;
}
inline bool wb_packed_affine_dot(ggml_type type,const uint8_t *w,const uint8_t *x,size_t n,double &sum) {
  if(type==GGML_TYPE_Q4_1)sum=wb_affine_dot_fields<block_q4_1>(w,x,n);
  else if(type==GGML_TYPE_Q5_1)sum=wb_affine_dot_fields<block_q5_1>(w,x,n);
  else return false;
  return true;
}

struct QuantizedInputs {
  size_t rows=WB_ROWS,K=WB_K,output_size;
  ggml_type weight_type=WB_WEIGHT_TYPE,activation_type=WB_ACTIVATION_TYPE;
  size_t weight_bytes,activation_bytes;
  std::vector<uint8_t> weights,activation;
  std::vector<float> source,expected;
  std::vector<uint8_t> expected_bytes;
  std::vector<int8_t> grid,signs,codebook;
  std::vector<float> f16_bits,scale;
  const uint32_t powers[5]={1,3,9,27,81};
  const ggml_type_traits_cpu *traits;

  QuantizedInputs()=default;

  QuantizedInputs(wb::InputSet &inputs) {
    ggml_cpu_init();
    if(K%ggml_blck_size(weight_type)||K%ggml_blck_size(activation_type))throw std::runtime_error("shape requires whole quantization blocks");
    weight_bytes=ggml_row_size(weight_type,K);activation_bytes=ggml_row_size(activation_type,K);
    traits=ggml_get_type_traits_cpu(weight_type);
    weights.resize(rows*weight_bytes);activation.resize(activation_bytes);
    source.resize(rows*K);
    if(WB_QUANT_KIND==1) {
      output_size=rows*weight_bytes;expected_bytes.resize(output_size);
      for(size_t i=0;i<source.size();++i)source[i]=float(int(i%31)-15)/16.0f;
      if(inputs.seed!=410)inputs.fill(source);
      inputs.load("X",source);
      for(size_t r=0;r<rows;++r)wb_quantize_contract(weight_type,source.data()+r*K,expected_bytes.data()+r*weight_bytes,K);
    } else {
      // Canonical floating inputs are quantized once and shared by both implementations.
      uint32_t state=inputs.seed;
      auto random=[&]{state^=state<<13;state^=state>>17;state^=state<<5;return(float(state>>8)/8388608.0f-1)*.5f;};
      std::vector<float> x(K),w(K),importance(K,1.0f);
      for(auto &v:x)v=random();
      wb_quantize_reference(activation_type,x.data(),activation.data(),K);
      inputs.load("X",activation);
      for(size_t r=0;r<rows;++r){for(auto &v:w)v=random();ggml_quantize_chunk(weight_type,w.data(),weights.data()+r*weight_bytes,0,1,K,importance.data());}
      inputs.load("W",weights);
      output_size=WB_QUANT_KIND==0?rows:rows*K;expected.resize(output_size);
      auto decode_w=ggml_get_type_traits(weight_type)->to_float;
      if(!decode_w)throw std::runtime_error("GGML has no dequantization oracle for selected type");
      if(WB_QUANT_KIND==0) {
        wb_dequantize_reference(activation_type,activation.data(),x.data(),K);
        for(size_t r=0;r<rows;++r){decode_w(weights.data()+r*weight_bytes,w.data(),K);double sum=0;
          if(activation_type!=GGML_TYPE_Q8_1||!wb_packed_affine_dot(weight_type,weights.data()+r*weight_bytes,activation.data(),K,sum))
            for(size_t k=0;k<K;++k)sum+=double(w[k])*x[k];
          expected[r]=float(sum);}
      } else for(size_t r=0;r<rows;++r)decode_w(weights.data()+r*weight_bytes,expected.data()+r*K,K);
    }
    inputs.finish();prepare_tables();
  }
  void prepare_tables() {
    auto table64=[&](const uint64_t* table,size_t n){grid.resize(n*8);for(size_t i=0;i<n;++i)for(size_t j=0;j<8;++j)grid[i*8+j]=int8_t(table[i]>>(j*8));};
    auto table32=[&](const uint32_t* table,size_t n){grid.resize(n*4);for(size_t i=0;i<n;++i)for(size_t j=0;j<4;++j)grid[i*4+j]=int8_t(table[i]>>(j*8));};
    switch(weight_type){
      case GGML_TYPE_IQ1_S:case GGML_TYPE_IQ1_M:table64(iq1s_grid,2048);break;
      case GGML_TYPE_IQ2_XXS:table64(iq2xxs_grid,256);break;
      case GGML_TYPE_IQ2_XS:table64(iq2xs_grid,512);break;
      case GGML_TYPE_IQ2_S:table64(iq2s_grid,1024);break;
      case GGML_TYPE_IQ3_XXS:table32(iq3xxs_grid,256);break;
      case GGML_TYPE_IQ3_S:table32(iq3s_grid,512);break;
      default:break;
    }
    signs.resize(128*8);for(size_t i=0;i<128;++i)for(size_t j=0;j<8;++j)signs[i*8+j]=(ksigns_iq2xs[i]&kmask_iq2xs[j])?-1:1;
    codebook.resize(16);for(size_t i=0;i<16;++i)codebook[i]=(weight_type==GGML_TYPE_MXFP4||weight_type==GGML_TYPE_NVFP4)?kvalues_mxfp4[i]:kvalues_iq4nl[i];
    f16_bits.resize(65536);for(uint32_t i=0;i<65536;++i)f16_bits[i]=ggml_fp16_to_fp32(uint16_t(i));
    scale.resize(256);for(uint32_t i=0;i<256;++i){
      if(weight_type==GGML_TYPE_NVFP4){int e=(i>>3)&15,m=i&7;scale[i]=(i==0||i==127)?0:(e==0?std::ldexp(float(m),-9):std::ldexp(1.0f+m/8.0f,e-7))*.5f;}
      else {uint32_t bits=i<2?0x00200000U<<i:(i-1U)<<23U;std::memcpy(&scale[i],&bits,4);}}
  }
  void check(const std::vector<float>& y) const {for(size_t i=0;i<y.size();++i)wb::close(y[i],expected[i],i);}
  void check(const std::vector<uint8_t>& y) const {if(y!=expected_bytes)throw std::runtime_error("numeric quantization bytes mismatch");}
};
