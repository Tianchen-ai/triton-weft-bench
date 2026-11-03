#pragma once
#include "controlled_projection.h"
#include <stdexcept>

template<int BITS> struct StudyVNNI;
#define STUDY_VECTOR(BITS, PREFIX, TYPE) \
template<> struct StudyVNNI<BITS> { \
  using V=TYPE; \
  static V load(const void *p){return PREFIX##_loadu_si##BITS((const TYPE*)p);} \
  static V zero(){return PREFIX##_setzero_si##BITS();} \
  static V bias(){return PREFIX##_set1_epi8(-128);} \
  static V xorb(V a,V b){return PREFIX##_xor_si##BITS(a,b);} \
  static V dot(V a,V b){return PREFIX##_dpbusd_epi32(zero(),a,b);} \
  static V add(V a,V b){return PREFIX##_add_epi32(a,b);} \
  static V sub(V a,V b){return PREFIX##_sub_epi32(a,b);} \
  static V mul(V a,V b){return PREFIX##_mullo_epi32(a,b);} \
  static V narrow(V a){return PREFIX##_srai_epi32(PREFIX##_slli_epi32(a,16),16);} \
};
STUDY_VECTOR(128,_mm,__m128i)
STUDY_VECTOR(256,_mm256,__m256i)
STUDY_VECTOR(512,_mm512,__m512i)
#undef STUDY_VECTOR
inline int32_t study_sum(__m128i a){a=_mm_hadd_epi32(a,a);a=_mm_hadd_epi32(a,a);return _mm_cvtsi128_si32(a);}
inline int32_t study_sum(__m256i a){return study_sum(_mm_add_epi32(_mm256_castsi256_si128(a),_mm256_extracti128_si256(a,1)));}
inline int32_t study_sum(__m512i a){return _mm512_reduce_add_epi32(a);}

template<bool Q6,bool SHARED,bool GROUPED,int BITS,int MR,int NR,bool RETAIN_OUTPUT=false>
inline void study_projection_vnni(const uint8_t *w,const uint8_t *x,float *y,size_t M,size_t N,size_t K){
  using SIMD=StudyVNNI<BITS>;using V=typename SIMD::V;
  constexpr size_t bytes=Q6?210:144,length=Q6?16:32,groups=256/length;
  constexpr int lanes=BITS/8,words=BITS/32;
  if(M%MR || N%NR || K%256)throw std::runtime_error("controlled VNNI tile does not divide its domain");
  if constexpr(RETAIN_OUTPUT)
    if(M!=MR)throw std::runtime_error("retained controlled output requires one row tile");
  const size_t blocks=K/256;std::fill(y,y+M*N,0.0f);
  for(size_t no=0;no<N;no+=NR){
    float retained[MR][NR]{};
    for(size_t b=0;b<blocks;++b){
    PackedBlockPreparation prepared[SHARED?1:MR][NR];
    if constexpr(SHARED)for(int ni=0;ni<NR;++ni)study_prepare<Q6>(w+((no+ni)*blocks+b)*bytes,prepared[0][ni]);
    for(size_t mo=0;mo<M;mo+=MR){
      if constexpr(!SHARED)for(int mi=0;mi<MR;++mi)for(int ni=0;ni<NR;++ni)
        study_prepare<Q6>(w+((no+ni)*blocks+b)*bytes,prepared[mi][ni]);
      V accum[MR][NR];
      for(int mi=0;mi<MR;++mi)for(int ni=0;ni<NR;++ni)accum[mi][ni]=SIMD::zero();
      for(int begin=0;begin<256;begin+=lanes){
        V inputs[MR],bias[MR];
        for(int mi=0;mi<MR;++mi){
          inputs[mi]=SIMD::load(x+((mo+mi)*blocks+b)*292+4+begin);
          if constexpr(Q6)bias[mi]=SIMD::dot(SIMD::bias(),inputs[mi]);
        }
        for(int ni=0;ni<NR;++ni)for(int mi=0;mi<MR;++mi){
          const auto &p=prepared[SHARED?0:mi][ni];
          V q=SIMD::load(p.q+begin);
          if constexpr(Q6)q=SIMD::xorb(q,SIMD::bias());
          V product=SIMD::dot(q,inputs[mi]);
          if constexpr(Q6)product=SIMD::sub(product,bias[mi]);
          if constexpr(GROUPED)product=SIMD::narrow(product);
          alignas(64) int32_t scales[words];
          for(int lane=0;lane<words;++lane)scales[lane]=p.sc[(begin+lane*4)/length];
          // Every four products retain their own numerical boundary and the
          // coefficient for their original 16/32-element group, even at 512b.
          accum[mi][ni]=SIMD::add(accum[mi][ni],SIMD::mul(product,SIMD::load(scales)));
        }
      }
      for(int mi=0;mi<MR;++mi)for(int ni=0;ni<NR;++ni){
        const auto &p=prepared[SHARED?0:mi][ni];const uint8_t *a=x+((mo+mi)*blocks+b)*292;
        int32_t correction=0;
        if constexpr(!Q6)for(size_t s=0;s<groups;++s){int16_t a0,a1;std::memcpy(&a0,a+260+s*4,2);std::memcpy(&a1,a+262+s*4,2);correction+=(int32_t(a0)+a1)*p.mn[s];}
        float dx;std::memcpy(&dx,a,4);const int32_t dot=study_sum(accum[mi][ni]);
        const float value=dx*(p.d*float(dot)-p.dmin*float(correction));
        if constexpr(RETAIN_OUTPUT)retained[mi][ni]+=value;
        else y[(mo+mi)*N+no+ni]+=value;
      }
    }
    }
    if constexpr(RETAIN_OUTPUT)
      for(int mi=0;mi<MR;++mi)for(int ni=0;ni<NR;++ni)
        y[mi*N+no+ni]=retained[mi][ni];
  }
}
