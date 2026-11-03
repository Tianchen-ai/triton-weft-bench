"""Native Exo/TensorIR calls in the same complete projection workload."""
import json, os, subprocess
from pathlib import Path
from ..core import NativeCall, PreparedNative

class IntrinsicProjection:
    def prepare_native(self,case,target,options,directory,executor,workload,name):
        if case.family != 'projection' or executor.host or target.settings.get('threads',1)!=1:
            raise NotImplementedError('controlled intrinsics select local x86 single-thread projection')
        q6=case.data['format']=='q6_k'
        flags=','.join('true' if x else 'false' for x in [q6,options['shared'],options['grouped']])
        setup='std::vector<uint8_t> xq(data.quantized.size());auto quant=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;'
        invoke='for(size_t m=0;m<data.M;++m)quant(data.activation.data()+m*data.K,xq.data()+m*data.activation_bytes,data.K);'
        header='controlled_projection.h'
        if 'vector_bits' in options:
            bits,mr,nr=(int(options.get(k,v)) for k,v in [('vector_bits',256),('MR',1),('NR',1)])
            retained='true' if options.get('retain_output',False) else 'false'
            invoke+=f'study_projection_vnni<{flags},{bits},{mr},{nr},{retained}>(data.weights.data(),xq.data(),output.data(),data.M,data.N,data.K);'
            header='controlled_projection_vnni.h'
        else:
            invoke+=f'study_projection<{flags}>(data.weights.data(),xq.data(),output.data(),data.M,data.N,data.K);'
        collect='if(xq!=data.quantized)throw std::runtime_error("activation quantization mismatch");'
        return PreparedNative(NativeCall(name,setup,invoke,collect=collect,includes=(header,),whole_workload=True),
                              details={'provider':'intrinsic-c','scope':'controlled-complete-projection','shared':options['shared'],'grouped':options['grouped']})

class ScheduledProjection:
    def prepare_native(self,case,target,options,directory,executor,workload,name):
        if case.family!='projection' or executor.host or target.settings.get('threads',1)!=1:
            raise NotImplementedError("controlled projection uses local x86 single-thread execution")
        fmt={'q4_k':'q4','q6_k':'q6'}[case.data['format']]
        provider=options['provider'];out=Path(options['artifact']).resolve() if options.get('artifact') else directory/'artifact'
        out.mkdir(parents=True,exist_ok=True)
        r={'format':fmt,'shape':case.shape,'shared':options['shared'],'grouped':options['grouped'],'output':str(out.resolve())}
        if 'vnni_lanes' in options:
            r['vnni_lanes']=options['vnni_lanes']
        for key in ['vector_bits','MR','NR','retain_output']:
            if key in options:r[key]=options[key]
        req=directory/'compile.json';req.write_text(json.dumps(r,indent=2))
        if not options.get('artifact'):
            with (directory/'compile.log').open('w') as log:
                p=subprocess.run([options['python'],options['compiler'],str(req.resolve())],stdout=log,stderr=subprocess.STDOUT,timeout=1200,
                                 env={**os.environ,'PYTHONNOUSERSITE':'1','TVM_BACKTRACE':'1'})
            if p.returncode:raise RuntimeError(f'{provider} compilation failed; see {directory}/compile.log')
        meta=json.loads((out/'metadata.json').read_text());lib=executor.upload(out/'kernel.so',name+'/kernel.so')
        setup='std::vector<uint8_t> xq(data.quantized.size()); auto quant=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;'
        setup+='void *handle=dlopen('+json.dumps(lib)+',RTLD_NOW|RTLD_LOCAL);if(!handle)throw std::runtime_error(dlerror());'
        invoke='for(size_t m=0;m<data.M;++m)quant(data.activation.data()+m*data.K,xq.data()+m*data.activation_bytes,data.K);'
        includes=['dlfcn.h'];include_flags=[];link_flags=['-ldl']
        if provider=='exo':
            setup+='using fn_t=void(*)(void*,int_fast32_t,int_fast32_t,int_fast32_t,const uint8_t*,const uint8_t*,float*);'
            setup+='auto fn=reinterpret_cast<fn_t>(dlsym(handle,"exo_projection"));if(!fn)throw std::runtime_error(dlerror());'
            invoke+='fn(nullptr,data.M,data.N,data.K/256,data.weights.data(),xq.data(),output.data());'
        elif provider=='tensorir':
            ffi=Path(meta['ffi_root']);includes+=['tvm/ffi/c_api.h']
            runtime=ffi.parent/'tvm/lib/libtvm_runtime.so'
            setup+='void *tvm_rt=dlopen('+json.dumps(str(runtime))+',RTLD_NOW|RTLD_GLOBAL);if(!tvm_rt)throw std::runtime_error(dlerror());'
            setup+='TVMFFIObjectHandle loader=nullptr;const char *loader_name="ffi.ModuleLoadFromFile";TVMFFIByteArray loader_key={loader_name,strlen(loader_name)};'
            setup+='if(TVMFFIFunctionGetGlobal(&loader_key,&loader))throw std::runtime_error("TVM module loader unavailable");'
            setup+='TVMFFIAny load_arg={},loaded_module={};load_arg.type_index=kTVMFFIRawStr;load_arg.v_c_str='+json.dumps(lib)+';'
            setup+='if(TVMFFIFunctionCall(loader,&load_arg,1,&loaded_module))throw std::runtime_error("TVM module initialization failed");'
            include_flags+=['-I'+str(ffi/'include')]
            link_flags += ['-L'+str(ffi/'lib'),'-Wl,-rpath,'+str(ffi/'lib'),'-ltvm_ffi']
            byte_count=144 if fmt=='q4' else 210
            setup+=f'int64_t ws[3]={{WB_N,WB_K/256,{byte_count}}},xs[3]={{WB_M,WB_K/256,292}},ys[2]={{WB_M,WB_N}};'
            setup+='DLTensor tensors[3]={{data.weights.data(),{kDLCPU,0},3,{kDLUInt,8,1},ws,nullptr,0},'
            setup+='{xq.data(),{kDLCPU,0},3,{kDLUInt,8,1},xs,nullptr,0},'
            setup+='{output.data(),{kDLCPU,0},2,{kDLFloat,32,1},ys,nullptr,0}};'
            setup+='TVMFFIAny args[3]={},result={};for(int i=0;i<3;++i){args[i].type_index=kTVMFFIDLTensorPtr;args[i].v_ptr=&tensors[i];}'
            setup+='auto fn=reinterpret_cast<TVMFFISafeCallType>(dlsym(handle,"__tvm_ffi_tensorir_projection"));if(!fn)throw std::runtime_error(dlerror());'
            invoke+='if(fn(nullptr,args,3,&result))throw std::runtime_error("TensorIR native call failed");'
        else:raise ValueError(provider)
        collect='if(xq!=data.quantized)throw std::runtime_error("activation quantization mismatch");'
        return PreparedNative(NativeCall(name,setup,invoke,collect=collect,includes=tuple(includes),whole_workload=True),
                              link_flags,{'provider':provider,'native_call_boundary':'whole projection including activation quantization',
                              'compiler_metadata':meta,'artifacts':{'compiled':str(out.resolve())}},include_flags)
