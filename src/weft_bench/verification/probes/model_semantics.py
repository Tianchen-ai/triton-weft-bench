"""Executable reference for the Level initialization/body/handoff rules.

This independent evaluator uses immutable NumPy snapshots and scalar state
updates. Its partition iterator is also used to check physical output slicing.
"""
from dataclasses import dataclass,field
import hashlib
import numpy as np

@dataclass
class LogicalLevel:
    extent:int
    partition:int
    events:list=field(default_factory=list)

    def instances(self):
        if self.extent<0 or self.partition<=0:raise ValueError("invalid domain")
        for start in range(0,self.extent,self.partition):
            stop=min(start+self.partition,self.extent)
            self.events.append(("initialize",start,stop))
            yield slice(start,stop)
            self.events.append(("handoff",start,stop))

def narrow16(value):
    value=int(value)%65536
    return value-65536 if value>=32768 else value

def composition_reference(source,initial,steps,rows,block,physical_width=None):
    memory=source.copy();result=np.zeros((len(initial),3),np.int32)
    logical=LogicalLevel(len(initial),rows)
    for region in logical.instances():
        state=initial[region].astype(np.int64).copy()
        cumulative=np.zeros(len(state),np.int64)
        snapshot=memory[region].copy()
        visits=0
        for iteration in range(steps):
            for column in LogicalLevel(source.shape[1],block).instances():
                delta=snapshot[:,column].sum(axis=1,dtype=np.int64)
                # Output slices share the single birth and snapshot above.
                width=physical_width or len(state)
                for start in range(0,len(state),width):
                    for r in range(start,min(start+width,len(state))):
                        state[r]=narrow16(state[r]+delta[r])
                        cumulative[r]+=state[r]
                memory[region,column]+=1 if iteration%2==0 else -1
                visits+=1
        result[region,0]=state;result[region,1]=cumulative;result[region,2]=visits
    return memory,result,logical.events

def run(root,options,compile_target):
    from triton import compile
    from triton.compiler import ASTSource
    from triton.backends.compiler import CPUTarget
    from model_semantics_kernels import semantics_composition
    from numerical_primitives import exact,native_evidence
    rng=np.random.default_rng(20261029);records=[];hashes={}
    variants=[(17,7),(4,16)]
    for rows,block in variants:
        compiled=compile(ASTSource(semantics_composition,{},{"ROWS":rows,"BLOCK":block}),
                         target=CPUTarget(backend="weft",**compile_target),options=options)
        other_target={**compile_target,"march":"x86-64-v3" if compile_target["march"]=="x86-64-v4" else "x86-64-v4"}
        other=compile(ASTSource(semantics_composition,{},{"ROWS":rows,"BLOCK":block}),
                      target=CPUTarget(backend="weft",**other_target),options=options)
        assert other.asm['weftir']==compiled.asm['weftir'],"target selection changed Canonical IR"
        digest=hashlib.sha256(compiled.asm['weftir'].encode()).hexdigest()
        hashes[f'{rows}-{block}']=digest
        native_evidence(compiled,root,f'composition-{rows}-{block}',["level","handoff","materialize","cast"])
        native_evidence(other,root,f'composition-{rows}-{block}-other',["level","handoff","materialize","cast"])
        for m,n in [(0,0),(5,0),(1,1),(19,37),(35,129)]:
            for steps in [0,1,3]:
                x=rng.integers(-5000,5001,(m,n),dtype=np.int32)
                initial=rng.integers(-32768,32768,m,dtype=np.int32)
                expected_x,expected,events=composition_reference(x,initial,steps,rows,block)
                for width in [1,4,8,16]:
                    px,py,pe=composition_reference(x,initial,steps,rows,block,width)
                    exact(px,expected_x);exact(py,expected)
                    assert pe==events
                storage=np.full((m,5),-777,np.int32);y=storage[:,1:4]
                twin=x.copy()
                compiled[(1,)](x,initial,y,steps)
                exact(x,expected_x);exact(y,expected);exact(storage[:,[0,4]],np.full((m,2),-777,np.int32))
                y[:]=-777
                other[(1,)](twin,initial,y,steps)
                exact(twin,expected_x);exact(y,expected);exact(storage[:,[0,4]],np.full((m,2),-777,np.int32))
                records.append({"M":m,"N":n,"steps":steps,"rows":rows,"block":block,
                                "births":len(events)//2,"integer_exact":True,"physical_reference_widths":[1,4,8,16],
                                "native_targets":[compile_target['march'],other_target['march']],"canonical_identical":True})
    from model_semantics_kernels import invalid_body_capture,invalid_sibling_capture,invalid_stage_update
    from model_semantics_kernels import static_births
    from model_semantics_kernels import lexical_helper_extent,composed_windows,mismatched_windows,static_window_copy
    helper=compile(ASTSource(lexical_helper_extent,{},{}),
                   target=CPUTarget(backend="weft",**compile_target),options=options)
    helper_cases=[]
    for parent in [0,1,17,128]:
        for count in [0,1,3,4,8,19]:
            output=np.full(2,-1,np.int32)
            helper[(1,)](np.zeros(parent,np.int32),output,count)
            exact(output,np.asarray([(count+3)//4,(count//2+3)//4],np.int32))
            helper_cases.append({"caller_extent":parent,"helper_extent":count,"visits":int(output[0])})
    native_evidence(helper,root,'lexical-helper-extent',["level","handoff"])
    window=compile(ASTSource(composed_windows,{},{}),
                   target=CPUTarget(backend="weft",**compile_target),options=options)
    for count in [0,1,7,17,31]:
        source=np.arange(31,dtype=np.int32);output=np.full(31,-777,np.int32)
        window[(1,)](np.zeros(4,np.int32),source,output,count)
        expected=np.full(31,-777,np.int32);expected[:count]=source[:count]+1
        exact(output,expected)
    assert window.asm['weftir'].count('"weft_kernel.local_shape"')==1
    native_evidence(window,root,'composed-windows',["local_shape","level","handoff"])
    fixed=compile(ASTSource(static_window_copy,{},{"M":16}),
                  target=CPUTarget(backend="weft",**compile_target),options=options)
    assert '"rows", "exact"' in fixed.asm['weftir']
    source=np.arange(16,dtype=np.int32);output=np.zeros_like(source)
    fixed[(1,)](source,output);exact(output,source+1)
    for choice in [False,True]:
        program=compile(ASTSource(static_births,{},{"SHARED":choice}),
                        target=CPUTarget(backend="weft",**compile_target),options=options)
        source=np.arange(7,dtype=np.float32);original=source.copy();destination=np.zeros_like(source)
        program[(1,)](source,destination)
        exact(source,original+1);exact(destination,original if choice else original*2)
        # Initialization yields the materialized value; it is not a body-local
        # materialize hidden under a compile-time branch.
        import re
        assert re.search(r'weft_kernel.materialize[\s\S]*?weft_kernel.births_yield',program.asm['weftir'])
    rejected=[]
    for kernel,expected in [(invalid_body_capture,("initialization","body")),
                            (invalid_sibling_capture,("initialization","region")),
                            (invalid_stage_update,("staged",)),
                            (mismatched_windows,("inconsistent",))]:
        try:
            compile(ASTSource(kernel,{},{}),target=CPUTarget(backend="weft",**compile_target),options=options)
        except Exception as error:
            diagnostic=str(error)
            assert all(word in diagnostic.lower() for word in expected),diagnostic
            rejected.append({"kernel":kernel.__name__,"diagnostic":diagnostic})
        else:raise AssertionError("invalid region composition accepted: "+kernel.__name__)
    return {"seed":20261029,"cases":records,"canonical_sha256":hashes,"region_rejections":rejected,
            "lexical_helper_cases":helper_cases,
            "checked":"full outputs, input mutation, and untouched output padding"}
