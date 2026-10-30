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
from ..libraries import bind_metadata, validate_binding
from .triton_native import compile_source, grid_arguments, selected_target
from .workspaces import invocation_workspaces


class TritonSequenceAdapter:
    def prepare_native(self, case, target, options, directory, worker, workload, name):
        options = {**options, **options.get('bindings', {}).get(case.name, {})}
        stages = options.get('stages')
        if not isinstance(stages, list) or not stages:
            raise ValueError('native sequence needs explicit ordered stages')
        if workload.before_call or workload.after_call:
            raise ValueError('native sequence uses complete workloads without outer row loops')
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
            tile_shape = stage.get('tile_shape')
            if tile_shape is not None:
                if (case.family != 'projection' or tile_shape != [32, 32]
                    or case.shape['M'] % 32 or case.shape['N'] % 32
                    or [arg['name'] for arg in abi['arguments']] != ['W', 'Xq', 'Y']
                    or not abi['arguments'][-1].get('strided')):
                    raise ValueError('projection tiles require the typed 32-by-32 W/Xq/Y entry and complete output tiles')
                expected_shapes = {'W': [32, case.shape['K']], 'Xq': [32, case.shape['K']], 'Y': [32, 32]}
                dimensions = {**case.shape, **abi.get('bindings', {}), 'M': 32, 'N': 32}
                if any([int(x) if str(x).isdecimal() else dimensions[x] for x in arg['shape']] != expected_shapes[arg['name']]
                       for arg in abi['arguments']):
                    raise ValueError('projection tile argument extents differ from its declared tile shape')
            for key, value in abi.get('bindings', {}).items():
                if key in bindings and bindings[key] != value:
                    raise ValueError('sequence stages disagree on dimension ' + key)
                bindings[key] = value
            for argument in abi['arguments']:
                argument = copy.deepcopy(argument)
                if tile_shape:
                    argument['shape'] = {'W': [case.shape['N'], case.shape['K']],
                                         'Xq': [case.shape['M'], case.shape['K']],
                                         'Y': [case.shape['M'], case.shape['N']]}[argument['name']]
                    argument['shape'] = list(map(str, argument['shape']))
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
        from .persistent import persistent_inputs
        workload, packing, packed_bases, packed_details = persistent_inputs(case, workload,
            combined, options.get('persistent_inputs', []), {}, options.get('persistent_providers'))
        if any(item['provider'] != 'ggml-model-default/AMX' for item in packed_details):
            raise ValueError('native sequence currently uses the explicit GGML model weight provider')
        validate_binding(case, combined, {}, case.shape, whole_workload=True,
                         packed_inputs=packed_bases)
        supplied = set(workload.buffers)
        for index, (_, manifest, _) in enumerate(compiled):
            initializes = stages[index].get('initializes', [])
            writable = {arg['name'] for arg in manifest['kernel']['arguments']
                        if arg.get('access') in ('write', 'readwrite')}
            if not isinstance(initializes, list) or not set(initializes) <= writable:
                raise ValueError('stage initializes must name writable stage arguments')
            # A kernel may initialize and then consume private scratch during
            # the same invocation. That source contract is explicit here.
            supplied.update(initializes)
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
        setup += scratch + packing + 'wb::poison_output(output);'
        invokes, details = [], []
        threads = int(options.get('num_cpu_threads', target.settings.get('threads', 1)))
        for index, (artifact, manifest, grid) in enumerate(compiled):
            label = f'sequence_{index}'
            deployment = f'{name}/{label}'
            tile_shape = stages[index].get('tile_shape')
            abi = manifest['kernel'] if tile_shape else manifest['grid']['kernel']
            remote_library = worker.upload(artifact / 'kernel.so', f'{deployment}/kernel.so')
            remote_manifest = worker.write(f'{deployment}/manifest.json',
                json.dumps(manifest if tile_shape else {**manifest, **manifest['grid']}))
            metadata = json.loads((artifact / 'triton.json').read_text())
            for filename in metadata['runtime_files']:
                worker.upload(artifact / filename, f'{deployment}/{filename}')
            declaration = abi['declaration']
            prefix = 'void ' + abi['symbol'] + '('
            if not declaration.startswith(prefix) or not declaration.endswith(');'):
                raise ValueError('unsupported stage declaration')
            parameters = re.sub(r'\brestrict\b', '__restrict__', declaration[len(prefix):])
            setup += (f'using {label}_type = void (*)(' + parameters +
                f'wb::TritonNative {label}_native(' + json.dumps(remote_library) + ',' + json.dumps(remote_manifest) + ');' +
                f'auto {label}_entry={label}_native.entry<{label}_type>();')
            if tile_shape:
                # Reuse the framework's exact tile ABI. Only pointer coordinates
                # vary across invocations, keeping the tile program unchanged.
                buffers = dict(workload.buffers)
                for role, coordinate in [('W', 'tile_n'), ('Xq', 'tile_m')]:
                    argument = next(arg for arg in abi['arguments'] if arg['name'] == role)
                    buffers[role] = (f'(reinterpret_cast<const uint8_t*>({buffers[role]})+{coordinate}*'
                                     f'(data.K/{argument["record_elements"]})*{argument["storage_bytes"]})')
                buffers['Y'] = f'({buffers["Y"]}+tile_m*data.N+tile_n)'
                call_abi = {**abi, 'symbol': label + '_entry'}
                call = bind_metadata(call_abi, buffers, {**case.shape, 'M': 32, 'N': 32},
                                     workload.scalars, {}, {**workload.strides, 'Y': (case.shape['N'], 1)})
                invokes.append(f'{label}_native.parallel(data.M/32*(data.N/32),{threads},'
                    f'[&](size_t begin,size_t end){{for(size_t tile=begin;tile<end;++tile){{'
                    'size_t tile_m=tile/(data.N/32)*32,tile_n=tile%(data.N/32)*32;'
                    + call + '}});')
                details.append({'source': stages[index]['source'], 'symbol': stages[index]['symbol'],
                    'tile_shape': tile_shape, 'artifact': str(artifact),
                    'build': metadata.get('native_build')})
                continue
            setup += (f'auto {label}_launcher={label}_native.launcher<wb::TritonNative::GridLauncher>(' +
                json.dumps(remote_library) + ',' + json.dumps(manifest['grid']['launcher']) + ');')
            packet = grid_arguments(abi, workload, case.shape, {})
            packet = re.sub(r'\blaunch_(arguments|scalar_\d+|extra_\d+)\b', lambda m: label + '_' + m[0], packet)
            setup += packet
            invokes.append(f'{label}_native.grid({label}_launcher,reinterpret_cast<weft_native_entry>({label}_entry),'
                           f'{label}_launch_arguments,{grid[0]},{grid[1]},{grid[2]},{threads});')
            details.append({'source': stages[index]['source'], 'symbol': stages[index]['symbol'],
                'grid': grid, 'initializes': stages[index].get('initializes', []),
                'artifact': str(artifact), 'build': metadata.get('native_build')})
        includes = ('triton_native.h', 'projection_reference.h') if packed_details else ('triton_native.h',)
        call = NativeCall(name, setup, ''.join(invokes), includes=includes, whole_workload=True)
        flags = ['-I' + runtime['include'], runtime['library'], '-Wl,-rpath,' + str(Path(runtime['library']).parent),
                 *runtime.get('ldflags', [])]
        return PreparedNative(call, flags, {'producer': 'triton', 'entry': 'synchronous-native-sequence',
            'stages': details, 'threads': threads, 'invocation_workspaces': descriptions,
            'persistent_inputs': packed_details,
            'preparation_scope': 'all stages execute inside every measured invocation'},
            include_flags=['-DWB_GGML_MODEL_WEIGHTS=1'] if packed_details else [])
