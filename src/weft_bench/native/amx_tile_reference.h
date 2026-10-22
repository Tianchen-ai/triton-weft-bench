#pragma once
#if !defined(__AMX_INT8__) || !defined(__AVX512VNNI__)
#error "The GGML AMX tile reference requires AMX INT8 and AVX512 VNNI compilation"
#endif
#include "amx/mmq.cpp"
#include "projection_reference.h"

struct AMXTileReference {
  ProjectionReference weights;
  const AMXTileInputs &input;
  std::vector<float> &output;
  AMXTileReference(const AMXTileInputs &data,std::vector<float> &out)
      : weights(data.weight_type,data.M,data.N,data.K,data.weights.data(),data.activation.data(),out),
        input(data),output(out) {
    if(weights.weight_buffer_name!="AMX")
      throw std::runtime_error("normal GGML did not select its AMX weight buffer");
  }
  void run() {
#if WB_AMX_DECODE
    for(int column=0;column<input.N;column+=64) {
      const char *packed=static_cast<const char *>(weights.result->src[0]->data) +
          PACKED_INDEX(column/16,0,input.K/WB_AMX_BLOCK_SIZE,get_tile_size<WB_AMX_WEIGHT_BLOCK>());
      tinygemm_kernel_vnni<WB_AMX_ACTIVATION_BLOCK,WB_AMX_WEIGHT_BLOCK,float,1,64,WB_AMX_BLOCK_SIZE>::apply(input.K/WB_AMX_BLOCK_SIZE,
          input.quantized.data(),packed,input.output(output)+column,input.ldc);
    }
#else
    ggml_tile_config_init();
    for(int column=0;column<input.N;column+=32) {
      const char *packed=static_cast<const char *>(weights.result->src[0]->data) +
          PACKED_INDEX(column/16,0,input.K/WB_AMX_BLOCK_SIZE,get_tile_size<WB_AMX_WEIGHT_BLOCK>());
      tinygemm_kernel_amx<WB_AMX_ACTIVATION_BLOCK,WB_AMX_WEIGHT_BLOCK,float,WB_AMX_BLOCK_SIZE>(input.M,32,input.K/WB_AMX_BLOCK_SIZE,
          input.quantized.data(),packed,input.output(output)+column,input.ldc);
    }
#endif
  }
};
