#pragma once
#include "worker.h"
#include <sys/mman.h>
#include <unistd.h>
#include <limits>

#if defined(WB_POINTER_G0) && defined(WB_POINTER_G1)
struct PointerGatherInputs {
  static constexpr size_t M=WB_POINTER_M,N=WB_POINTER_N,G0=WB_POINTER_G0,G1=WB_POINTER_G1;
  size_t row_size=G0*N,column_size=M*G1,output_size=row_size+2*column_size;
  std::vector<float> x;
  std::vector<int32_t> indices0,indices1;
  std::vector<double> expected;
  explicit PointerGatherInputs(wb::InputSet &inputs):x(M*N),indices0(row_size),indices1(column_size),expected(output_size) {
    inputs.fill(x);
    for(size_t r=0;r<G0;++r)for(size_t c=0;c<N;++c)
      indices0[r*N+c]=int32_t((N+2-c+(c%2?r:0))%M);
    for(size_t r=0;r<M;++r)for(size_t c=0;c<G1;++c)
      indices1[r*G1+c]=int32_t(((c==2?0:c)*3+r*2+1)%N);
    inputs.load("X",x);inputs.load("I0",indices0);inputs.load("I1",indices1);inputs.finish();
    for(size_t r=0;r<G0;++r)for(size_t c=0;c<N;++c) {
      auto index=indices0[r*N+c];
      if(index<0||size_t(index)>=M)throw std::runtime_error("axis-0 gather index is out of range");
      expected[r*N+c]=x[size_t(index)*N+c];
    }
    for(size_t r=0;r<M;++r)for(size_t c=0;c<G1;++c) {
      auto index=indices1[r*G1+c];
      if(index<0||size_t(index)>=N)throw std::runtime_error("axis-1 gather index is out of range");
      expected[row_size+r*G1+c]=x[r*N+size_t(index)];
      expected[row_size+column_size+r*G1+c]=double(x[r*N+size_t(index)])+0.5;
    }
  }
  void check(const std::vector<float> &output)const {
    for(size_t i=0;i<output_size;++i)wb::close(output[i],expected[i],i);
  }
};
#endif

#ifdef WB_POINTER_M
struct PointerArgmaxInputs {
  static constexpr size_t M=WB_POINTER_M,N=WB_POINTER_N;
  size_t output_size=M;
  std::vector<float> x,expected;
  std::vector<int32_t> indices,expected_indices;
  explicit PointerArgmaxInputs(wb::InputSet &inputs):x(M*N),expected(M),indices(M,-1),expected_indices(M) {
    inputs.fill(x);
    for(size_t m=0;m<M;++m) {
      for(size_t n=0;n<N;++n)x[m*N+n]=-2.0f-std::abs(x[m*N+n]);
      if(m%4==0){x[m*N+std::min(size_t(3),N-1)]=3.5f;x[m*N+N-1]=3.5f;}
      else if(m%4==1){x[m*N]=-0.25f;x[m*N+N/2]=-0.25f;}
      else if(m%4==2)std::fill_n(x.data()+m*N,N,-1.5f);
      else x[m*N+N-1]=5.0f;
    }
    inputs.load("X",x);inputs.finish();
    for(size_t m=0;m<M;++m) {
      expected[m]=x[m*N];expected_indices[m]=0;
      for(size_t n=0;n<N;++n) {
        if(!std::isfinite(x[m*N+n]))throw std::runtime_error("argmax input must be finite");
        if(x[m*N+n]>expected[m]){expected[m]=x[m*N+n];expected_indices[m]=int32_t(n);}
      }
    }
  }
  void check(const std::vector<float> &output)const {
    for(size_t m=0;m<M;++m)
      if(output[m]!=expected[m]||indices[m]!=expected_indices[m])
        throw std::runtime_error("argmax value or first-index mismatch at row "+std::to_string(m));
  }
};

struct PointerProductInputs {
  static constexpr size_t M=WB_POINTER_M,N=WB_POINTER_N;
  size_t output_size=M;
  std::vector<float> x;
  std::vector<double> expected;
  explicit PointerProductInputs(wb::InputSet &inputs):x(M*N),expected(M,1.0) {
    inputs.fill(x);
    for(auto &value:x)value=1.0f+value*0.015625f;
    for(size_t m=0;m<M;++m) {
      if(m%4==1)x[m*N]=-x[m*N];
      else if(m%4==2)x[m*N+N/2]=0.0f;
      else if(m%4==3)for(size_t n=0;n<N;++n)x[m*N+n]=n%2?2.0f:0.5f;
    }
    inputs.load("X",x);inputs.finish();
    for(size_t m=0;m<M;++m)for(size_t n=0;n<N;++n) {
      if(!std::isfinite(x[m*N+n]))throw std::runtime_error("product input must be finite");
      expected[m]*=double(x[m*N+n]);
    }
  }
  void check(const std::vector<float> &output)const {
    for(size_t m=0;m<M;++m)wb::close(output[m],expected[m],m);
  }
};
#endif

struct PointerAliasInputs {
  static constexpr size_t N=WB_POINTER_N;
  size_t output_size=N+WB_POINTER_GUARD+2;
  std::vector<float> initial,state;
  std::vector<double> expected;

  explicit PointerAliasInputs(wb::InputSet &inputs):initial(output_size) {
    inputs.fill(initial);inputs.load("X",initial);inputs.finish();
    expected.assign(initial.begin(),initial.end());
    for(size_t i=0;i<N;++i)expected[i+2]=double(initial[i+1])+1.0;
    state=initial;
  }
  void reset(){std::copy(initial.begin(),initial.end(),state.begin());}
  void check(const std::vector<float> &output)const {
    for(size_t i=0;i<output_size;++i) {
      if(i>=2&&i<N+2)wb::close(output[i],expected[i],i);
      else if(output[i]!=initial[i])throw std::runtime_error("alias kernel modified storage outside its output span");
    }
  }
};

class GuardedFloats {
  void *mapping=nullptr;
  size_t mapped_bytes=0;
  float *pointer=nullptr;
public:
  explicit GuardedFloats(size_t count) {
    const long queried=sysconf(_SC_PAGESIZE);
    if(queried<=0)throw std::runtime_error("cannot determine page size");
    const size_t page=static_cast<size_t>(queried),bytes=count*sizeof(float);
    const size_t writable=((bytes+page-1)/page)*page;
    mapped_bytes=writable+page;
    mapping=mmap(nullptr,mapped_bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(mapping==MAP_FAILED){mapping=nullptr;throw std::runtime_error("cannot allocate guarded input");}
    auto *guard=static_cast<char*>(mapping)+writable;
    if(mprotect(guard,page,PROT_NONE)!=0){munmap(mapping,mapped_bytes);mapping=nullptr;throw std::runtime_error("cannot protect masked-address guard page");}
    pointer=reinterpret_cast<float*>(guard-bytes);
  }
  GuardedFloats(const GuardedFloats&)=delete;
  GuardedFloats& operator=(const GuardedFloats&)=delete;
  ~GuardedFloats(){if(mapping)munmap(mapping,mapped_bytes);}
  float *data(){return pointer;}
};

struct PointerGuardInputs {
  size_t output_size=WB_POINTER_N;
  GuardedFloats x{WB_POINTER_N},z{WB_POINTER_N},y{WB_POINTER_N};
  std::vector<double> expected;
  explicit PointerGuardInputs(wb::InputSet &inputs) {
    std::vector<float> left(output_size),right(output_size);
    inputs.fill(left);inputs.fill(right);inputs.load("X",left);inputs.load("Z",right);inputs.finish();
    std::copy(left.begin(),left.end(),x.data());std::copy(right.begin(),right.end(),z.data());
    expected.resize(output_size);
    for(size_t i=0;i<output_size;++i)expected[i]=double(left[i])+right[i];
  }
  void check(const std::vector<float> &output)const {
    for(size_t i=0;i<output_size;++i)wb::close(output[i],expected[i],i);
  }
};

#ifdef WB_POINTER_STEPS
struct PointerCarryInputs {
  size_t output_size=WB_POINTER_N;
  std::vector<float> x;
  std::vector<double> expected;
  explicit PointerCarryInputs(wb::InputSet &inputs):x(WB_POINTER_STEPS*WB_POINTER_N),expected(WB_POINTER_N,0) {
    inputs.fill(x);inputs.load("X",x);inputs.finish();
    for(size_t step=0;step<WB_POINTER_STEPS;++step)
      for(size_t i=0;i<output_size;++i)expected[i]+=x[step*output_size+i];
  }
  void check(const std::vector<float> &output)const {
    for(size_t i=0;i<output_size;++i)wb::close(output[i],expected[i],i);
  }
};
#endif

struct PointerSelectInputs {
  size_t output_size=WB_POINTER_N;
  int32_t select_x;
  std::vector<float> x,z;
  explicit PointerSelectInputs(wb::InputSet &inputs):x(output_size),z(output_size) {
    const float selected=inputs.parameter("select_x",1.0f);
    if(!std::isfinite(selected)||std::trunc(selected)!=selected||
       double(selected)<std::numeric_limits<int32_t>::min()||double(selected)>std::numeric_limits<int32_t>::max())
      throw std::runtime_error("select_x must be an int32 value");
    select_x=static_cast<int32_t>(selected);
    inputs.fill(x);inputs.fill(z);inputs.load("X",x);inputs.load("Z",z);inputs.finish();
  }
  void check(const std::vector<float> &output)const {
    const auto &expected=select_x?x:z;
    for(size_t i=0;i<output_size;++i)wb::close(output[i],expected[i],i);
  }
};

struct PointerBooleanInputs {
  size_t output_size=1+WB_POINTER_N+WB_POINTER_GUARD;
  std::vector<uint8_t> x;
  explicit PointerBooleanInputs(wb::InputSet &inputs):x(output_size) {
    for(auto &value:x)value=inputs.random()>=0?1:0;
    inputs.load("X",x);inputs.finish();
    for(auto value:x)if(value>1)throw std::runtime_error("boolean storage must contain only 0 or 1");
  }
  void check(const std::vector<uint8_t> &output)const {
    for(size_t i=0;i<output_size;++i) {
      const uint8_t expected=i>=1&&i<=WB_POINTER_N?uint8_t(!x[i]):uint8_t(0xa5);
      if(output[i]!=expected)throw std::runtime_error("boolean output or padding mismatch at "+std::to_string(i));
    }
  }
};
