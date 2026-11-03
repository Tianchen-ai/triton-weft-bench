"""Triton CPU native calls inside the original Triton–Weft Bench worker."""
import json, os, subprocess
from pathlib import Path
from ..core import NativeCall, PreparedNative

class TritonCPUAdapter:
    def prepare_native(self,case,target,options,directory,executor,workload,name):
        options={**options,**options.get('bindings',{}).get(case.name,{})}
        if target.settings.get('threads',1)!=1 or executor.host:
            raise NotImplementedError('this native Triton CPU provider selects local x86 single-thread execution')
        output=directory/'artifact';output.mkdir(parents=True,exist_ok=True)
        request={k:options[k] for k in ['source','program']}
        request.update(case=case.name,shape=case.shape,parameters=options.get('parameters',{}),
                       options=options.get('backend_options',{}),output=str(output.resolve()))
        rp=directory/'triton-cpu-compile.json';rp.write_text(json.dumps(request,indent=2))
        env={**os.environ,**options.get('environment',{})}
        env['PYTHONPATH']=os.pathsep.join(options['pythonpath'])
        env['PYTHONNOUSERSITE']='1';env['TRITON_DEFAULT_BACKEND']='cpu'
        command=[options['python'],str(Path(__file__).with_name('compile_triton_cpu.py')),str(rp.resolve())]
        with (directory/'compile.log').open('w') as log:
            done=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        if done.returncode:raise RuntimeError('Triton CPU compilation failed; see '+str(directory/'compile.log'))
        meta=json.loads((output/'metadata.json').read_text())
        library=executor.upload(output/'kernel.so',name+'/kernel.so')
        types=[];args=[]
        buffers=dict(workload.batch_buffers or workload.buffers)
        projection = options.get('projection', False)
        if projection:
            if case.family != 'projection': raise ValueError('projection adapter requires projection workload')
            buffers['Xq']='cpu_xq.data()'
        for parameter,kind,role,dtype in meta['arguments']:
            if kind=='buffer':
                types.append('void*');args.append('const_cast<void*>(static_cast<const void*>('+buffers[role]+'))')
            elif kind=='scalar':
                types.append({'fp32':'float','i32':'int32_t','i64':'int64_t'}[dtype]);args.append(workload.scalars[role])
            else:raise ValueError(kind)
        types+=['uint32_t']*6
        setup='void *cpu_handle=dlopen('+json.dumps(library)+',RTLD_NOW|RTLD_LOCAL); if(!cpu_handle)throw std::runtime_error(dlerror());'
        if projection:
            setup+='std::vector<uint8_t> cpu_xq(data.quantized.size());auto cpu_quant=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;'
        setup+='using cpu_fn_t=void(*)('+','.join(types)+');'
        setup+='auto cpu_fn=reinterpret_cast<cpu_fn_t>(dlsym(cpu_handle,'+json.dumps(meta['name'])+')); if(!cpu_fn)throw std::runtime_error(dlerror());'
        gx,gy,gz=meta['launch']['grid']
        invoke=f'for(uint32_t z=0;z<{gz};++z)for(uint32_t y=0;y<{gy};++y)for(uint32_t x=0;x<{gx};++x)cpu_fn('+','.join(args+["x","y","z",str(gx),str(gy),str(gz)])+');'
        if projection:
            invoke='for(size_t m=0;m<data.M;++m)cpu_quant(data.activation.data()+m*data.K,cpu_xq.data()+m*data.activation_bytes,data.K);'+invoke
        call=NativeCall(name,setup,invoke,collect='if(cpu_xq!=data.quantized)throw std::runtime_error("activation quantization mismatch");' if projection else '',includes=('dlfcn.h',),whole_workload=True)
        return PreparedNative(call,['-ldl'],{'provider':'triton-cpu','entry':'single-instance',
                'triton_cpu':meta,'artifacts':{'triton_cpu':str(output.resolve())},
                'native_call_boundary':'whole workload, sequential grid traversal'})
