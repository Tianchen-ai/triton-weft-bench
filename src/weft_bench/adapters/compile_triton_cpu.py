"""Compile an ordinary Triton program in an explicitly selected CPU environment."""
import importlib.util, json, os, sys
from pathlib import Path

def main():
    request=json.loads(Path(sys.argv[1]).read_text());out=Path(request['output']);out.mkdir(parents=True,exist_ok=True)
    os.environ['TRITON_CACHE_DIR']=str(out/'cache')
    os.environ['TRITON_DEFAULT_BACKEND']='cpu'
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    spec=importlib.util.spec_from_file_location('paper_triton_cpu_kernels',request['source'])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    program=module.PROGRAMS[request['program']]
    if request['case'] not in program.cases: raise ValueError('program/workload mismatch')
    constants,launch=program.launch(request['shape'],request.get('parameters',{}))
    signature={n:program.arguments[n][2] if n in program.arguments else 'constexpr' for n in program.kernel.arg_names}
    compiled=triton.compile(ASTSource(program.kernel,signature=signature,constexprs=constants),
                           target=GPUTarget('cpu','x86_64',0),options=request.get('options',{}))
    for kind,value in compiled.asm.items():
        path=out/('kernel.'+kind)
        path.write_bytes(value) if isinstance(value,bytes) else path.write_text(str(value))
    meta={'name':compiled.metadata.name,'triton_file':triton.__file__,'version':triton.__version__,
          'signature':signature,'arguments':[(n,*program.arguments[n]) for n in program.kernel.arg_names if n in program.arguments],
          'constants':constants,'launch':launch,'program':request['program'],'options':request.get('options',{}),
          'timing':'native CPU function with sequential program-id traversal; no Python dispatch'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))

if __name__=='__main__':main()
