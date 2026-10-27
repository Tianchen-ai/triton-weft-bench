#pragma once
#include "quantized_inputs.h"

struct ProjectionInputs {
  size_t M=WB_M,N=WB_N,K=WB_K,output_size=M*N;
  ggml_type weight_type=WB_WEIGHT_TYPE,activation_type=WB_ACTIVATION_TYPE;
  size_t weight_bytes,activation_bytes;
  std::vector<uint8_t> weights,quantized;
  std::vector<float> activation,expected;
  std::vector<uint64_t> checked_indices;
  QuantizedInputs tables;
  ProjectionInputs(wb::InputSet &inputs, bool quantized_input=false, bool full_output=false) {
    ggml_cpu_init();
    if(K%ggml_blck_size(weight_type)||K%ggml_blck_size(activation_type))throw std::runtime_error("projection K must contain whole quantization blocks");
    weight_bytes=ggml_row_size(weight_type,K);activation_bytes=ggml_row_size(activation_type,K);
    weights.resize(N*weight_bytes);activation.resize(M*K);expected.resize(output_size);
    inputs.fill(activation);inputs.load("X",activation);
    std::vector<float> w(K),importance(K,1.0f);
    if(inputs.load("W",weights)) inputs.discard(N*K);
    else for(size_t n=0;n<N;++n){inputs.fill(w);ggml_quantize_chunk(weight_type,w.data(),weights.data()+n*weight_bytes,0,1,K,importance.data());}
    tables.prepare_tables();
    std::vector<float> decoded(M*K);
    if(activation_type==GGML_TYPE_F32)decoded=activation;
    else {
      quantized.resize(M*activation_bytes);
      for(size_t m=0;m<M;++m){
#ifdef WB_PROJECTION_Q8K_CONTRACT
        // The scalar GGML reference's rounding trick can contract its multiply
        // into the magic-bias add. Native SIMD rounds the product to f32 first.
        // Use the declared independent storage contract and check the actual
        // CPU quantizer before using these bytes in the projection oracle.
        wb_quantize_contract(activation_type,activation.data()+m*K,quantized.data()+m*activation_bytes,K);
        std::vector<uint8_t> native(activation_bytes);
        auto quantizer=ggml_get_type_traits_cpu(activation_type)->from_float;
        if(!quantizer)throw std::runtime_error("GGML has no native activation quantizer");
        quantizer(activation.data()+m*K,native.data(),K);
        if(std::memcmp(native.data(),quantized.data()+m*activation_bytes,activation_bytes))
          throw std::runtime_error("GGML activation quantization differs from declared Q8_K contract");
#else
        wb_quantize_reference(activation_type,activation.data()+m*K,quantized.data()+m*activation_bytes,K);
#endif
      }
      if(quantized_input)inputs.load("Xq",quantized);
      for(size_t m=0;m<M;++m)wb_dequantize_reference(activation_type,quantized.data()+m*activation_bytes,decoded.data()+m*K,K);
    }
    inputs.finish();
    // Independent complete-column checks for decode; a bounded sample for prefill.
    size_t samples=full_output?output_size:M==1?N:std::min<size_t>(64,output_size);
    for(size_t s=0;s<samples;++s){size_t i=samples==1?0:s*(output_size-1)/(samples-1),m=i/N,n=i%N;
      checked_indices.push_back(i);
      wb_dequantize_reference(weight_type,weights.data()+n*weight_bytes,w.data(),K);double sum=0;
      if(activation_type!=GGML_TYPE_Q8_1||!wb_packed_affine_dot(weight_type,weights.data()+n*weight_bytes,quantized.data()+m*activation_bytes,K,sum))
        for(size_t k=0;k<K;++k)sum+=double(w[k])*decoded[m*K+k];
      expected[i]=float(sum);}
  }
  void check(const std::vector<float> &y) const {
    for(auto value:y)if(!std::isfinite(value))throw std::runtime_error("numeric nonfinite projection");
    for(size_t i:checked_indices)wb::close(y[i],expected[i],i);
  }
};
