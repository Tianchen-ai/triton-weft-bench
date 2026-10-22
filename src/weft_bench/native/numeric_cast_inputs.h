#pragma once
#include "worker.h"
#include <limits>

struct NumericCastInputs {
  size_t output_size=WB_CAST_N;
  std::vector<float> x,expected;

  static float half_roundtrip(float value, bool toward_zero) {
    _Float16 narrowed=static_cast<_Float16>(value);
    float rounded=static_cast<float>(narrowed);
    if(toward_zero && std::isfinite(value) && std::abs(rounded)>std::abs(value)) {
      uint16_t bits;
      std::memcpy(&bits,&narrowed,sizeof(bits));
      --bits;
      std::memcpy(&narrowed,&bits,sizeof(bits));
      rounded=static_cast<float>(narrowed);
    }
    return rounded;
  }

  explicit NumericCastInputs(wb::InputSet &inputs):x(output_size),expected(output_size) {
    const float values[]={0.0f,-0.0f,1.0f,-1.0f,1.00048828125f,-1.00048828125f,
      1.0006f,-1.0006f,0x1p-24f,-0x1p-24f,0x1p-25f,-0x1p-25f,
      0x1.ffcp-15f,-0x1.ffcp-15f,65504.0f,-65504.0f,65520.0f,-65520.0f,
      std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),127.75f,-127.75f,0.9999f,-0.9999f};
    for(size_t i=0;i<output_size;++i) x[i]=values[i%(sizeof(values)/sizeof(values[0]))];
    if(WB_CAST_MODE==4 || WB_CAST_MODE==6 || (WB_CAST_MODE>=10 && WB_CAST_MODE<=14))
      for(size_t i=0;i<output_size;++i)x[i]=(float(int(i%29)-14)+0.75f)*29.0f;
    if(WB_CAST_MODE==16 || WB_CAST_MODE==17) {
      const float boundaries[]={0,1,127,128,255,256,32767,32768,65535,65536,-1,-128,-129,-32768};
      for(size_t i=0;i<output_size;++i)x[i]=boundaries[i%(sizeof(boundaries)/sizeof(boundaries[0]))];
    }
    inputs.load("X",x);inputs.finish();
    for(size_t i=0;i<output_size;++i) {
      switch(WB_CAST_MODE) {
      case 0:expected[i]=half_roundtrip(x[i],false);break;
      case 1:expected[i]=half_roundtrip(x[i],true);break;
      case 2:case 5:expected[i]=x[i];break;
      case 3:case 15:expected[i]=x[i]!=0.0f?1.0f:0.0f;break;
      case 4:case 13:expected[i]=float(static_cast<int32_t>(x[i]));break;
      case 6:expected[i]=float(static_cast<int32_t>(half_roundtrip(x[i],false)));break;
      case 10:case 11:case 16:case 17:{
        int32_t low=static_cast<uint32_t>(static_cast<int32_t>(x[i]))&255;
        expected[i]=float((WB_CAST_MODE==10 || WB_CAST_MODE==16) && low>=128 ? low-256 : low);break;
      }
      case 12:expected[i]=float(static_cast<uint32_t>(static_cast<int32_t>(x[i])));break;
      case 14:expected[i]=float(static_cast<uint32_t>(static_cast<int32_t>(x[i]))&65535);break;
      default:throw std::runtime_error("unsupported cast oracle mode");
      }
    }
  }

  void check(const std::vector<float> &output)const {
    for(size_t i=0;i<output_size;++i) {
      float actual=output[i],reference=expected[i];
      if(std::isnan(reference)) {
        if(!std::isnan(actual))throw std::runtime_error("cast did not preserve NaN at "+std::to_string(i));
      } else if(std::isinf(reference) || reference==0.0f) {
        if(actual!=reference || std::signbit(actual)!=std::signbit(reference))
          throw std::runtime_error("cast changed infinity or signed zero at "+std::to_string(i));
      } else {
        // Distinguish the declared rounding direction even at a half subnormal.
        // Integer/boolean outcomes are exact; half results allow less than half
        // the spacing of adjacent representable values, not arbitrary bit equality.
        double tolerance=(WB_CAST_MODE==0 || WB_CAST_MODE==1)
            ? std::max(0x1p-26,std::abs(double(reference))*0x1p-13):0.0;
        if(!std::isfinite(actual) || std::abs(double(actual)-reference)>tolerance)
          throw std::runtime_error("cast rounding/value mismatch at "+std::to_string(i));
      }
    }
  }
};
