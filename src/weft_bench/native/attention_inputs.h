#pragma once
#include "worker.h"

// One complete causal GQA call. The reference materializes scores, independently
// of the candidate's online maximum/denominator/state recurrence.
struct AttentionInputs {
  const size_t B=WB_B,T=WB_T,S=WB_S,H=WB_H,G=WB_G,D=WB_D;
  const size_t output_size=B*T*H*D;
  size_t start;
  float scale;
  std::vector<float> query,keys,values;
  AttentionInputs(wb::InputSet &inputs):
    start(size_t(inputs.parameter("start", float(S-T)))),
    scale(inputs.parameter("scale", 1.0f/std::sqrt(float(D)))),
    query(output_size),keys(B*S*G*D),values(B*S*G*D) {
    if(!G || H%G || T>S || start>S-T)
      throw std::runtime_error("invalid causal attention shape or prefix start");
    inputs.fill(query);inputs.fill(keys);inputs.fill(values);
    inputs.load("Q",query);inputs.load("K",keys);inputs.load("V",values);
    inputs.finish();
  }
  void check(const std::vector<float> &output) const {
    for(size_t batch=0;batch<B;++batch)for(size_t token=0;token<T;++token)
      for(size_t head=0;head<H;++head) {
        const size_t group=head/(H/G), query_base=((batch*T+token)*H+head)*D;
        const size_t count=start+token+1;
        std::vector<double> scores(count);
        double maximum=-INFINITY;
        for(size_t key=0;key<count;++key) {
          const size_t key_base=((batch*S+key)*G+group)*D;
          double score=0;
          for(size_t d=0;d<D;++d)score+=double(query[query_base+d])*keys[key_base+d];
          scores[key]=score*scale;
          maximum=std::max(maximum,scores[key]);
        }
        double denominator=0;
        for(auto &score:scores){score=std::exp(score-maximum);denominator+=score;}
        for(size_t d=0;d<D;++d) {
          double reference=0;
          for(size_t key=0;key<count;++key)
            reference+=scores[key]*values[((batch*S+key)*G+group)*D+d];
          const size_t index=query_base+d;
          wb::close(output[index],reference/denominator,index);
        }
      }
  }
};
