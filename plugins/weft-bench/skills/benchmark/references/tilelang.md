# TileLang–Weft

Select a local plugin environment with `--environment tilelang` when configured;
`--plugin-info` reports its actual Bench installation and target configuration.

- Source JIT: `weft_bench.adapters.triton_jit:TileLangJitAdapter`.
- Source AOT or exported native artifact: `weft_bench.adapters.triton_native:TileLangArtifactAdapter`.
- Source programs: installed `tilelang.weft.kernels`; provide source/symbol,
  source parameters, explicit grid/thread contract and physical `backend_options`.
- Packed/dynamic/state/workspace inputs use existing fixtures and preserve their
  aliases. Do not replace a source JIT failure with an AOT pass result.
- Use the same `show`, `op`, fixed baseline, `prepare-model`, `validate-model` and
  `e2e` commands. Native, grid, Python hot call, cold compile and model phases
  remain separate measurement scopes.
- llama.cpp selects `producer: "tilelang"` and exact compiled role bindings.
  `deployment.vec_dot` provides dynamic row contracts for Q4_K/Q6_K × Q8_K.

See the repository's bilingual `docs/tilelang-weft-integration.md` and
`docs/tilelang-weft-integration.zh-CN.md` for the complete configuration.
Outputs remain under Bench `output/`; public TileLang performance is under evaluation.
