"""Native complete attention references in the common worker."""
from ..core import NativeCall,PreparedNative
from .cpu_reference import library_configuration

class AttentionReference:
    def prepare_native(self,case,target,options,directory,executor,workload,name):
        setup="std::vector<float> scores(data.tq*data.tk);"
        if not options.get("blas",False):
            return PreparedNative(NativeCall(name,setup,"attention_materialized(data,scores.data(),output.data());",whole_workload=True),
                                  details={"provider":"c++-simd","algorithm":"materialized-attention"})
        config,includes,links=library_configuration(target,options,"openblas")
        setup+="openblas_set_num_threads("+str(target.settings.get("threads",1))+");"
        invoke="cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,data.tq,data.tk,data.d,data.scale,data.query.data(),data.d,data.key.data(),data.d,0,scores.data(),data.tk);"
        invoke+="attention_softmax(scores.data(),data.tq,data.tk);"
        invoke+="cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,data.tq,data.d,data.tk,1,scores.data(),data.tk,data.value.data(),data.d,0,output.data(),data.d);"
        return PreparedNative(NativeCall(name,setup,invoke,includes=("cblas.h",),whole_workload=True),
                              links,{"provider":"openblas","algorithm":"two-sgemm-and-stable-softmax","reference_build":config},includes)
