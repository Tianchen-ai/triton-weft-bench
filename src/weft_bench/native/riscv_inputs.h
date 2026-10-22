#pragma once
#include "worker.h"

#ifdef WB_SCAN_KIND
struct RiscvScanInputs {
  static constexpr size_t M=WB_SCAN_M,N=WB_SCAN_N;
  size_t size=M*N,output_size=size*(WB_SCAN_KIND==0?1:2);
  std::vector<int32_t> x,values,expected;
  explicit RiscvScanInputs(wb::InputSet &inputs):x(size),values(size),expected(output_size) {
    for(size_t i=0;i<size;++i) {x[i]=WB_SCAN_KIND==2?int32_t(i%3)-1:int32_t(i%11)-5; values[i]=int32_t(i%7)-3;}
    inputs.load("X",x);if(WB_SCAN_KIND==2)inputs.load("Values",values);inputs.finish();
    size_t lanes=WB_SCAN_AXIS==0?N:M,length=WB_SCAN_AXIS==0?M:N;
    for(size_t lane=0;lane<lanes;++lane) {
      int64_t a=WB_SCAN_KIND==2?1:0,b=0;
      for(size_t step=0;step<length;++step) {
        size_t along=WB_SCAN_REVERSE?length-1-step:step;
        size_t i=WB_SCAN_AXIS==0?along*N+lane:lane*N+along;
        if(WB_SCAN_KIND==2) {b=b*x[i]+values[i];a*=x[i];}
        else {a+=x[i];b=step==0?x[i]:std::max(b,int64_t(x[i]));}
        if(a<INT32_MIN||a>INT32_MAX||b<INT32_MIN||b>INT32_MAX)throw std::runtime_error("scan oracle overflow");
        expected[i]=int32_t(a);if(WB_SCAN_KIND)expected[size+i]=int32_t(b);
      }
    }
  }
  void check(const std::vector<int32_t> &output)const {
    if(output!=expected)throw std::runtime_error("numeric mismatch in integer prefix scan");
  }
};
#endif

#ifdef WB_ATOMIC_KIND
struct RiscvAtomicInputs {
  static constexpr size_t N=WB_ATOMIC_N,CAPACITY=WB_ATOMIC_CAPACITY;
  size_t output_size=CAPACITY;
  std::vector<int32_t> initial,state,values,compare;
  explicit RiscvAtomicInputs(wb::InputSet &inputs):initial(WB_ATOMIC_KIND==8?2:WB_ATOMIC_KIND==9?1:CAPACITY),values(CAPACITY),compare(CAPACITY) {
    if(N>CAPACITY)throw std::runtime_error("atomic active count exceeds storage");
    for(size_t i=0;i<initial.size();++i)initial[i]=int32_t(i)-17;
    if(WB_ATOMIC_KIND==8)initial={11,29};
    if(WB_ATOMIC_KIND==9)initial={0};
    for(size_t i=0;i<CAPACITY;++i){values[i]=19-int32_t(i);compare[i]=int32_t(i)-17+int32_t(i%2);}
    inputs.load("State",initial);
    if(WB_ATOMIC_KIND<8)inputs.load("Values",values);
    if(WB_ATOMIC_KIND==7)inputs.load("Compare",compare);
    inputs.finish();state=initial;
    if(WB_ATOMIC_KIND==9&&initial[0]!=0)throw std::runtime_error("CAS contention starts with zero");
  }
  void reset(){state=initial;}
  void check(const std::vector<int32_t> &old)const {
    for(size_t i=N;i<CAPACITY;++i)if(old[i]!=-999)throw std::runtime_error("numeric mismatch: inactive atomic lane wrote output");
    if(WB_ATOMIC_KIND==8) {
      for(size_t bucket=0;bucket<2;++bucket) {
        std::vector<int32_t> observed;
        for(size_t i=bucket;i<N;i+=2)observed.push_back(old[i]);
        std::sort(observed.begin(),observed.end());
        for(size_t i=0;i<observed.size();++i)if(observed[i]!=initial[bucket]+int32_t(i))throw std::runtime_error("numeric mismatch: atomic old-value permutation");
        if(state[bucket]!=initial[bucket]+int32_t(observed.size()))throw std::runtime_error("numeric mismatch: contended atomic state");
      }
    } else if(WB_ATOMIC_KIND==9) {
      int32_t winner=state[0];size_t zeros=0;
      if(winner<1||winner>int32_t(N))throw std::runtime_error("numeric mismatch: invalid CAS winner");
      for(size_t i=0;i<N;++i){zeros+=old[i]==0;if(old[i]!=0&&old[i]!=winner)throw std::runtime_error("numeric mismatch: CAS old value");}
      if(zeros!=1||old[winner-1]!=0)throw std::runtime_error("numeric mismatch: CAS must have one winner");
    } else {
      for(size_t i=0;i<CAPACITY;++i) {
        int32_t expected=initial[i];
        if(i<N) {
          if(old[i]!=initial[i])throw std::runtime_error("numeric mismatch: atomic returned old value");
          switch(WB_ATOMIC_KIND){
            case 0:expected+=values[i];break;
            case 1:expected=std::max(expected,values[i]);break;
            case 2:expected=std::min(expected,values[i]);break;
            case 3:expected&=values[i];break;
            case 4:expected|=values[i];break;
            case 5:expected^=values[i];break;
            case 6:expected=values[i];break;
            case 7:if(expected==compare[i])expected=values[i];break;
          }
        }
        if(state[i]!=expected)throw std::runtime_error("numeric mismatch: atomic final state or mask");
      }
    }
  }
};
#endif
