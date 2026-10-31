"""Bind the normal shape-specialized FP32 panel entry to the current workload."""
from .triton_native import TritonAdapter


class FP32PanelAdapter(TritonAdapter):
    def prepare_native(self, case, target, options, directory, worker, workload, name):
        return self.prepare(case, target, options, directory, worker, workload, name)

    def prepare(self, case, target, options, directory, worker, workload, name):
        from ..catalogs import resolve_source
        selected = {**options, **options.get('bindings', {}).get(case.name, {})}
        if selected.get('source'):
            selected['source'] = resolve_source(selected['source'])
        if selected.get('symbol') == 'row_blocked_matmul_f32' and not selected.get('artifact'):
            shape = case.shape
            rows, width = shape['M'], shape['K']
            if rows % 8 == 0 and width > 0 and width % 8 == 0:
                selected['constexprs'] = {
                    **selected.get('constexprs', {}), **shape,
                    'BM': 8, 'BN': 32, 'BK': min(width, 4096),
                    'MB': rows // 8, 'UNROLL': 4,
                }
            else:
                selected['symbol'] = 'packed_matmul_f32'
                selected['constexprs'] = {'BM': 4, 'BN': 64, 'BK': 256}
            selected['backend_options'] = {
                **selected.get('backend_options', {}), 'tune_cpu': 'sapphirerapids',
            }
            selected.pop('bindings', None)
        return super().prepare(case, target, selected, directory, worker, workload, name)
