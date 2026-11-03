#pragma once
#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <algorithm>

// A controlled intrinsic implementation; the unmodified optimized GGML route
// is measured separately.  The logical format and arithmetic match all DSLs.
struct PackedBlockPreparation {
  alignas(64) int8_t q[256];
  int32_t sc[16], mn[16];
  float d, dmin;
};
inline float study_half(const uint8_t *p){uint16_t h;std::memcpy(&h,p,2);return _cvtsh_ss(h);}
template<bool Q6> inline void study_prepare(const uint8_t *w,PackedBlockPreparation &p){
  if constexpr(!Q6){
    p.d=study_half(w);p.dmin=study_half(w+2);
    for(int s=0;s<4;++s){
      p.sc[s]=w[4+s]&63;p.mn[s]=w[8+s]&63;
      p.sc[s+4]=(w[12+s]&15)|((w[4+s]>>6)<<4);
      p.mn[s+4]=(w[12+s]>>4)|((w[8+s]>>6)<<4);
    }
    const auto mask=_mm256_set1_epi8(15);
    for(int g=0;g<4;++g){
      auto packed=_mm256_loadu_si256((const __m256i*)(w+16+g*32));
      _mm256_storeu_si256((__m256i*)(p.q+64*g),_mm256_and_si256(packed,mask));
      _mm256_storeu_si256((__m256i*)(p.q+64*g+32),_mm256_and_si256(_mm256_srli_epi16(packed,4),mask));
    }
  }else{
    p.d=study_half(w+208);p.dmin=0;
    for(int s=0;s<16;++s){p.sc[s]=(int8_t)w[192+s];p.mn[s]=0;}
    for(int h=0;h<2;++h)for(int c=0;c<4;++c)for(int l=0;l<32;++l)
      p.q[h*128+c*32+l]=int8_t(((w[h*64+(c%2)*32+l]>>(c/2*4))&15)|(((w[128+h*32+l]>>(c*2))&3)<<4))-32;
  }
}
template<bool GROUPED> inline __m128i study_dot16(const int8_t *w,const int8_t *x){
  auto a=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)w));
  auto b=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)x));
  auto pairs=_mm256_madd_epi16(a,b);
  auto four=_mm256_hadd_epi32(pairs,pairs);
  if constexpr(GROUPED)four=_mm256_srai_epi32(_mm256_slli_epi32(four,16),16);
  return _mm_unpacklo_epi64(_mm256_castsi256_si128(four),_mm256_extracti128_si256(four,1));
}
template<bool Q6,bool GROUPED> inline float study_contract(const PackedBlockPreparation &p,const uint8_t *x){
  constexpr int groups=Q6?16:8,length=Q6?16:32;
  auto accumulated=_mm_setzero_si128();int32_t correction=0;float dx;std::memcpy(&dx,x,4);
  for(int s=0;s<groups;++s){
    auto sub=study_dot16<GROUPED>(p.q+s*length,(const int8_t*)x+4+s*length);
    if constexpr(!Q6)sub=_mm_add_epi32(sub,study_dot16<GROUPED>(p.q+s*length+16,(const int8_t*)x+4+s*length+16));
    accumulated=_mm_add_epi32(accumulated,_mm_mullo_epi32(sub,_mm_set1_epi32(p.sc[s])));
    if constexpr(!Q6){int16_t a,b;std::memcpy(&a,x+260+4*s,2);std::memcpy(&b,x+262+4*s,2);correction+=(int32_t(a)+b)*p.mn[s];}
  }
  accumulated=_mm_hadd_epi32(accumulated,accumulated);accumulated=_mm_hadd_epi32(accumulated,accumulated);
  return dx*(p.d*float(_mm_cvtsi128_si32(accumulated))-p.dmin*float(correction));
}
template<bool Q6,bool SHARED,bool GROUPED>
inline void study_projection(const uint8_t *w,const uint8_t *x,float *y,size_t M,size_t N,size_t K){
  constexpr size_t bytes=Q6?210:144;
  const size_t blocks=K/256;std::fill(y,y+M*N,0.0f);
  for(size_t n=0;n<N;++n)for(size_t b=0;b<blocks;++b){
    PackedBlockPreparation p;
    if constexpr(SHARED)study_prepare<Q6>(w+(n*blocks+b)*bytes,p);
    for(size_t m=0;m<M;++m){
      if constexpr(!SHARED)study_prepare<Q6>(w+(n*blocks+b)*bytes,p);
      y[m*N+n]+=study_contract<Q6,GROUPED>(p,x+(m*blocks+b)*292);
    }
  }
}
