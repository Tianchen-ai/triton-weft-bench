"""Complete scaled noncausal attention, with two independent reference algorithms."""
from ..core import NativeWorkload
from ..types import Case

def catalog():
    return {"online_attention":Case("online_attention","online_attention",{"Tq":32,"Tk":512,"D":64},
                                  "complete-scaled-noncausal-attention",dtype="f32",
                                  input_policy="finite-uniform-fixed-seed")}

def ports(case):
    from ..workloads import TensorSpec
    s=case.shape
    return [TensorSpec("Q","f32",(s["Tq"],s["D"])),TensorSpec("K","f32",(s["Tk"],s["D"])),
            TensorSpec("V","f32",(s["Tk"],s["D"])),TensorSpec("Y","f32",(s["Tq"],s["D"]),"output")]

def workload(case):
    return NativeWorkload("online_attention_inputs.h","OnlineAttentionInputs data(inputs);",
        {"Q":"data.query.data()","K":"data.key.data()","V":"data.value.data()","Y":"output.data()"},
        defines=tuple("-DWB_"+k+"="+str(v) for k,v in case.shape.items()),
        scalars={"scale":"data.scale"},data_type="OnlineAttentionInputs",correctness="full-f64-reference-rtol2e-4-atol2e-5")
