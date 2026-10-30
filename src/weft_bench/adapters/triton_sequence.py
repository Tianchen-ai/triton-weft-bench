"""Compose explicit native preparation/compute stages in one timed workload.

Each stage uses its exported typed grid ABI and the product's shared executor.
Invocation workspaces are allocated once and shared only by declared names.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import re

from ..core import NativeCall, PreparedNative
from ..libraries import validate_binding
from .triton_native import compile_source, grid_arguments, selected_target
from .workspaces import invocation_workspaces


class TritonSequenceAdapter:
    def prepare_native(self, case, target, options, directory, worker, workload, name):
        options = {**options, **options.get('bindings', {}).get(case.name, {})}
        stages = options.get('stages')
        if not isinstance(stages, list) or not stages:
            raise ValueError('native sequence needs explicit ordered stages')
        if workload.before_call or workload.after_call or options.get('persistent_inputs'):
            raise ValueError('native sequence uses complete raw-input workloads')
        arguments, bindings, compiled = {}, {}, []
        for index, stage in enumerate(stages):
            stage_options = {**options, **stage}
            stage_options.pop('bindings', None)
            stage_directory = directory / f'stage-{index}'
            stage_directory.mkdir(parents=True, exist_ok=True)
            artifact = compile_source(case, target, stage_options, stage_directory)
            manifest = json.loads((artifact / 'manifest.json').read_text())
            from .cpu_target import compiler_target
            if compiler_target(manifest['target']) != selected_target(stage_options, target.settings['target']):
                raise ValueError('sequence stage target differs from requested target')
            abi = manifest['kernel']
            for key, value in abi.get('bindings', {}).items():
                if key in bindings and bindings[key] != value:
                    raise ValueError('sequence stages disagree on dimension ' + key)
                bindings[key] = value
            for argument in abi['arguments']:
                old = arguments.get(argument['name'])
                if old is None:
                    arguments[argument['name']] = copy.deepcopy(argument)
                else:
                    for key in ['kind', 'encoding', 'shape', 'strided', 'storage_bytes', 'record_elements']:
                        if old.get(key) != argument.get(key):
                            raise ValueError('sequence argument ABI differs: ' + argument['name'] + '/' + key)
                    accesses = {old.get('access'), argument.get('access')} - {'none', None}
                    old['access'] = next(iter(accesses)) if len(accesses) == 1 else 'readwrite'
            grid = stage.get('grid')
            if not isinstance(grid, list) or len(grid) > 3 or not grid or any(type(x) is not int or x < 0 for x in grid):
                raise ValueError('stage grid needs one to three nonnegative dimensions')
            compiled.append((artifact, manifest, [*grid, *([1] * (3 - len(grid)))]))
        combined = {'arguments': list(arguments.values()), 'bindings': bindings}
        validate_binding(case, combined, {}, case.shape, whole_workload=True)
        supplied = set(workload.buffers)
        for _, manifest, _ in compiled:
            for arg in manifest['kernel']['arguments']:
                if arg['kind'] not in ('view', 'pointer'):
                    continue
                role, access = arg['name'], arg.get('access', 'none')
                if role not in supplied and access in ('read', 'readwrite'):
                    raise ValueError('sequence reads invocation workspace before preparation: ' + role)
            supplied.update(arg['name'] for arg in manifest['kernel']['arguments']
                            if arg.get('access') in ('write', 'readwrite'))
        workload, scratch, descriptions = invocation_workspaces(case, workload, combined, options.get('workspaces', []))
        runtime = options.get('native_runtime', target.settings.get('triton_runtime', {}))
        if not runtime.get('include') or not runtime.get('library'):
            raise ValueError('sequence needs the selected native runtime')
        setup = 'std::vector<uint8_t> workspace(data.M*data.activation_bytes);' if case.family == 'projection' else ''
        setup += scratch + 'wb::poison_output(output);'
        invokes, details = [], []
        threads = int(options.get('num_cpu_threads', target.settings.get('threads', 1)))
        for index, (artifact, manifest, grid) in enumerate(compiled):
            label = f'sequence_{index}'
            abi = manifest['grid']['kernel']
            remote_library = worker.upload(artifact / 'kernel.so', f'{label}/kernel.so')
            remote_manifest = worker.write(f'{label}/manifest.json', json.dumps({**manifest, **manifest['grid']}))
            metadata = json.loads((artifact / 'triton.json').read_text())
            for filename in metadata['runtime_files']:
                worker.upload(artifact / filename, f'{label}/{filename}')
            declaration = abi['declaration']
            prefix = 'void ' + abi['symbol'] + '('
            if not declaration.startswith(prefix) or not declaration.endswith(');'):
                raise ValueError('unsupported stage declaration')
            parameters = re.sub(r'\brestrict\b', '__restrict__', declaration[len(prefix):])
            setup += (f'using {label}_type = void (*)(' + parameters +
                f'wb::TritonNative {label}_native(' + json.dumps(remote_library) + ',' + json.dumps(remote_manifest) + ');' +
                f'auto {label}_entry={label}_native.entry<{label}_type>();' +
                f'auto {label}_launcher={label}_native.launcher<wb::TritonNative::GridLauncher>(' +
                json.dumps(remote_library) + ',' + json.dumps(manifest['grid']['launcher']) + ');')
            packet = grid_arguments(abi, workload, case.shape, {})
            packet = re.sub(r'\blaunch_(arguments|scalar_\d+|extra_\d+)\b', lambda m: label + '_' + m[0], packet)
            setup += packet
            invokes.append(f'{label}_native.grid({label}_launcher,reinterpret_cast<weft_native_entry>({label}_entry),'
                           f'{label}_launch_arguments,{grid[0]},{grid[1]},{grid[2]},{threads});')
            details.append({'source': stages[index]['source'], 'symbol': stages[index]['symbol'],
                'grid': grid, 'artifact': str(artifact), 'build': metadata.get('native_build')})
        call = NativeCall(name, setup, ''.join(invokes), includes=('triton_native.h',), whole_workload=True)
        flags = ['-I' + runtime['include'], runtime['library'], '-Wl,-rpath,' + str(Path(runtime['library']).parent),
                 *runtime.get('ldflags', [])]
        return PreparedNative(call, flags, {'producer': 'triton', 'entry': 'synchronous-native-sequence',
            'stages': details, 'threads': threads, 'invocation_workspaces': descriptions,
            'preparation_scope': 'all stages execute inside every measured invocation'})
