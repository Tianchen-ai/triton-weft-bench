"""Selected dense View checks hosted by the existing runtime worker."""
import numpy as np
import torch
import triton
import dense_view_kernels as kernels

SCENARIOS = ("eager_tensor", "strided_offset", "readonly_zero_stride", "level_2d_tails",
             "same_axis_subtiles", "subview_extent", "alias", "aot")
SENTINEL = -7.0


def strided_fixture(m, n):
    source = torch.arange(m * (n*2+3), dtype=torch.float32).reshape(m, n*2+3) / 2048
    output = torch.full((m, n*3+2), SENTINEL, dtype=torch.float32)
    x, y = source[:, 1:1+2*n:2], output[:, 1:1+3*n:3]
    expected = output.clone()
    expected[:, 1:1+3*n:3] = x * 1.25 - 0.5
    guard = torch.ones_like(output, dtype=torch.bool)
    guard[:, 1:1+3*n:3] = False
    return source, output, x, y, expected, guard


def run_scenario(name, root, options, close, reject):
    if name not in SCENARIOS:
        raise ValueError("unknown dense View scenario " + name)

    def snapshot(label, **arrays):
        values = {key: value.detach().numpy() if isinstance(value, torch.Tensor) else value
                  for key, value in arrays.items()}
        np.savez(root / ("dense-view-" + label + ".npz"), **values)

    def details(compiled):
        abi = compiled.metadata.kernel_abi
        assert sum(arg["kind"] == "view" for arg in abi["arguments"]) >= 2
        return {"typed_view_abi": True, "kernel_abi": abi,
                "compile_target": vars(compiled.metadata.target), "native_build": compiled.metadata.native_build}

    def check_storage(label, fixture, invoke):
        source, output, x, y, expected, guard = fixture
        original = source.clone()
        snapshot(label, source=original, expected_output=expected)
        compiled = invoke(x, y)
        snapshot(label, source=original, actual_source=source, expected_output=expected, actual_output=output)
        close(output, expected)
        assert torch.equal(source, original), "read-only input changed"
        assert bool(torch.all(output[guard] == SENTINEL)), "View store overwrote padding"
        layout = {"shape": list(x.shape), "x_stride": list(x.stride()), "y_stride": list(y.stride()),
                  "x_offset": x.storage_offset(), "y_offset": y.storage_offset(), "padding_unchanged": True}
        return compiled, layout

    if name == "eager_tensor":
        compiled = []
        for m, n in ((3, 65), (5, 97)):
            x = torch.arange(m*n, dtype=torch.float32).reshape(m, n)/2048
            original = x.clone()
            y = torch.full_like(x, SENTINEL)
            expected = x*1.25-0.5
            snapshot(f"eager-{m}x{n}", source=x, expected_output=expected)
            first = kernels.affine_fixed[(1,)](x, y, **options)
            close(y, expected)
            warm = kernels.affine_fixed[(1,)](x, y, **options)
            snapshot(f"eager-{m}x{n}", source=original, actual_source=x, expected_output=expected, actual_output=y)
            close(y, expected)
            assert torch.equal(x, original), "eager read-only input changed"
            assert warm is first, "identical eager launch did not reuse its kernel"
            compiled.append(first)
        assert compiled[0] is compiled[1], "ABI shape change recompiled a fixed-block kernel"
        return {**details(compiled[0]), "shapes": [[3, 65], [5, 97]], "eager_cpu_tensor": True,
                "warm_reuse": True, "dynamic_shape_reuses_kernel": True}

    if name in ("strided_offset", "level_2d_tails"):
        shapes = ((3, 65), (5, 97)) if name == "strided_offset" else ((5, 19),)
        kernel = kernels.affine_fixed if name == "strided_offset" else kernels.tiled_2d
        compiled, records = [], []
        for m, n in shapes:
            result, layout = check_storage(f"{name}-{m}x{n}", strided_fixture(m, n),
                                          lambda x, y: kernel[(1,)](x, y, **options))
            compiled.append(result); records.append(layout)
        assert all(value is compiled[0] for value in compiled), "ABI layout change recompiled a fixed-block kernel"
        reductions, tail_atomic = [], {}
        if name == "level_2d_tails":
            for label in ("negative-with-ties", "positive"):
                x = torch.arange(1, 38, dtype=torch.float32)
                if label == "negative-with-ties":
                    x = -x
                    x[3] = x[28] = -0.5
                original = x.clone()
                y_storage = torch.full((7,), SENTINEL, dtype=torch.float32)
                y = y_storage[1:5]
                expected = torch.tensor([x.max().item(), x.double().sum().item(), x.min().item(),
                                         x.argmax().item()], dtype=torch.float32)
                snapshot("tail-reduce-" + label, source=original, expected_output=expected)
                reduction_kernel = kernels.tail_reductions[(1,)](x, y, **options)
                snapshot("tail-reduce-" + label, source=original, actual_source=x,
                         expected_output=expected, actual_output=y, output_storage=y_storage)
                close(y, expected)
                assert y[3].item() == expected[3].item(), "tail argmax changed the first winning index"
                assert torch.equal(x, original), "tail reduction changed its input"
                assert bool(torch.all(y_storage[[0, 5, 6]] == SENTINEL)), "tail reduction overwrote output guard"
                reductions.append({"input": label, "N": 37, "block": 64,
                                   "operations": ["max", "sum", "min", "argmax-first-index"],
                                   "expected": expected.tolist(), "native_build": reduction_kernel.metadata.native_build})
            values = np.resize(np.arange(1, 8, dtype=np.int32), 37)
            expected = np.bincount((values + 1 // values) % 4, minlength=4).astype(np.int32)
            x = torch.from_numpy(values.copy())
            storage = torch.full((8,), -999, dtype=torch.int32)
            counts = storage[2:6]
            counts.zero_()
            snapshot("tail-atomic", source=values, expected_output=expected, initial_output=storage)
            atomic_kernel = kernels.tail_atomic_counts[(1,)](x, counts, **options)
            snapshot("tail-atomic", source=values, actual_source=x, expected_output=expected,
                     actual_output=counts, output_storage=storage)
            np.testing.assert_array_equal(counts.numpy(), expected)
            assert counts.sum().item() == 37, "invalid View lanes produced atomic effects"
            assert bool(torch.all(storage[[0, 1, 6, 7]] == -999)), "tail atomic overwrote output guard"
            np.testing.assert_array_equal(x.numpy(), values)
            kinds = {arg["name"]: arg["kind"] for arg in atomic_kernel.metadata.kernel_abi["arguments"]}
            assert kinds["X"] == "view" and kinds["Y"] == "pointer", "mixed View/pointer ABI was not exercised"
            tail_atomic = {"N": 37, "block": 64, "input_range": [1, 7], "buckets": 4,
                           "expected_counts": expected.tolist(), "total_count": 37,
                           "division_result_used_in_address": True, "explicit_pointer_mask": False,
                           "integer_exact": True, "mixed_view_pointer_abi": True,
                           "native_build": atomic_kernel.metadata.native_build}
        return {**details(compiled[0]), "layouts": records, "dynamic_layout_reuses_kernel": True,
                "tile": [2, 8] if name == "level_2d_tails" else [1, 64],
                "tail_reductions": reductions, "tail_atomic": tail_atomic}

    if name == "readonly_zero_stride":
        source = np.arange(65, dtype=np.float32)/2048
        original = source.copy()
        x = np.broadcast_to(source, (3, 65))
        storage = np.full((3, 68), SENTINEL, dtype=np.float32)
        y = storage[:, 1:66]
        expected = storage.copy(); expected[:, 1:66] = x*1.25-0.5
        snapshot(name, source=source, expected_output=expected)
        compiled = kernels.affine_fixed[(1,)](x, y, **options)
        snapshot(name, source=original, actual_source=source, expected_output=expected, actual_output=storage)
        close(storage, expected)
        np.testing.assert_array_equal(source, original)
        np.testing.assert_array_equal(storage[:, [0, 66, 67]], np.full((3, 3), SENTINEL, np.float32))
        y.setflags(write=False)
        diagnostic = reject(lambda: kernels.affine_fixed[(1,)](x, y, **options), ("writ", "readonly", "read-only"))
        close(storage, expected)
        return {**details(compiled), "readonly_input": not x.flags.writeable, "input_byte_strides": list(x.strides),
                "readonly_output_rejected": diagnostic, "no_partial_writes": True}

    if name in ("same_axis_subtiles", "subview_extent"):
        n = 37
        source = torch.arange(n*2+3, dtype=torch.float32)/2048
        original = source.clone()
        x = source[1:1+2*n:2]
        output = torch.full((n*3+2,), SENTINEL, dtype=torch.float32)
        y = output[1:1+3*n:3]
        records, compiled = [], []
        variants = (("direct", kernels.subview_extent), ("branch", kernels.subview_branch)) if name == "subview_extent" else (
                    ("parent-slice", kernels.same_axis_subtiles),)
        for variant, kernel in variants:
            variant_compiled = []
            for start in ((5, 21) if name == "subview_extent" else (0,)):
                output.fill_(SENTINEL)
                expected = output.clone(); active = torch.zeros_like(output, dtype=torch.bool)
                region = slice(start, start+8) if name == "subview_extent" else slice(None)
                expected[1:1+3*n:3][region] = x[region]*1.25-0.5
                active[1:1+3*n:3][region] = True
                label = f"{name}-{variant}-{start}"
                snapshot(label, source=source, expected_output=expected)
                sizes = torch.full((2,), -1, dtype=torch.int32)
                result = (kernel[(1,)](x, y, sizes, start, **options)
                          if name == "subview_extent" else kernel[(1,)](x, y, **options))
                snapshot(label, source=original, actual_source=source, expected_output=expected, actual_output=output, extents=sizes)
                close(output, expected)
                assert bool(torch.all(output[~active] == SENTINEL)), "subregion changed untouched storage"
                assert torch.equal(source, original), "read-only input changed"
                if name == "subview_extent":
                    assert sizes.tolist() == [n, 8], "root/subview extent differs from the declared region"
                variant_compiled.append(result)
                records.append({"variant": variant, "N": n, "start": start,
                                "extents": sizes.tolist() if name == "subview_extent" else None,
                                "native_build": result.metadata.native_build})
            assert all(value is variant_compiled[0] for value in variant_compiled), "runtime subview offset changed compiled code"
            compiled.extend(variant_compiled)
        return {**details(compiled[0]), "scenarios": records, "fixed_value_extent": 8 if name == "subview_extent" else 4,
                "padding_unchanged": True, "runtime_shape_only": True}

    if name == "alias":
        source = torch.arange(3*38, dtype=torch.float32).reshape(3, 38)/1000
        original = source.clone()
        expected = original.clone(); expected[:, :-1] = original[:, :-1]*0.75+original[:, 1:]
        snapshot(name, source=original, expected_output=expected)
        compiled = kernels.state_update_fixed[(1,)](source[:, :-1], source[:, 1:], **options)
        snapshot(name, source=original, expected_output=expected, actual_output=source)
        close(source, expected)
        x = torch.arange(195, dtype=torch.float32).reshape(3, 65)/1000
        saved = x.clone()
        alias = reject(lambda: kernels.affine_fixed[(1,)](x, x, **options), ("alias", "overlap"))
        assert torch.equal(x, saved), "rejected alias invocation wrote input"
        storage = torch.full((65,), SENTINEL, dtype=torch.float32)
        overlap = reject(lambda: kernels.affine_fixed[(1,)](x, storage.as_strided((3, 65), (0, 1)), **options),
                         ("overlap", "stride"))
        assert bool(torch.all(storage == SENTINEL)), "rejected writable overlap changed storage"
        broadcasts = []
        for n in (0, 1, 7, 8, 9, 17):
            input_storage = torch.full((n+2,), SENTINEL, dtype=torch.float32)
            values = input_storage[1:1+n]
            values.copy_(torch.arange(n, dtype=torch.float32) / 16 - 3)
            before = input_storage.clone()
            output = torch.full((n+2, 19), SENTINEL, dtype=torch.float32)
            expected = output.clone()
            expected[1:1+n, 1:17] = values[:, None] + torch.arange(16, dtype=torch.float32)[None, :]
            label = f"broadcast-snapshot-{n}"
            snapshot(label, source=before, expected_output=expected)
            kernel = kernels.broadcast_snapshot[(1,)](values, output[1:1+n, 1:17], **options)
            snapshot(label, source=before, actual_source=input_storage, expected_output=expected, actual_output=output)
            assert torch.equal(output, expected), "broadcast observed a later write or changed output padding"
            assert bool(torch.all(values == 1000)), "source write was lost"
            assert torch.equal(input_storage[[0, -1]], before[[0, -1]]), "source write escaped its View"
            broadcasts.append(kernel)
        assert all(kernel is broadcasts[0] for kernel in broadcasts), "snapshot shape recompiled the kernel"
        (root / "broadcast-snapshot.llir").write_text(broadcasts[0].asm["llir"])
        return {**details(compiled), "permitted_shifted_alias_correct": True,
                "disjoint_alias_rejected": alias, "writable_overlap_rejected": overlap, "no_partial_writes": True,
                "broadcast_snapshot_before_write": True, "broadcast_extents": [0, 1, 7, 8, 9, 17]}

    first, first_layout = check_storage("aot-source", strided_fixture(3, 65),
                                       lambda x, y: kernels.affine_fixed[(1,)](x, y, **options))
    artifact = root / "dense-view-aot"
    first.export(artifact)
    loaded = triton.load(artifact)
    records = []
    for m, n in ((3, 65), (5, 97)):
        _, layout = check_storage(f"aot-loaded-{m}x{n}", strided_fixture(m, n),
                                  lambda x, y: loaded[(1,)](x, y, 1.25, bias=-0.5))
        records.append(layout)
    return {**details(first), "artifact": str(artifact), "source_layout": first_layout,
            "loaded_layouts": records, "aot_dynamic_shape_and_stride_correct": True}
