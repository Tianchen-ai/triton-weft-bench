"""Small exact-integer oracles; candidate execution stays in the runtime worker."""
import numpy as np


def exact(actual, expected):
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as error:
        raise ValueError("numerical mismatch: " + str(error)) from error


def wrap_integer(value, bits, signed=True):
    value = int(value) % (1 << bits)
    return value - (1 << bits) if signed and value >= (1 << (bits-1)) else value


def mac_oracle(left, right, group, bits, signed):
    result = np.empty((2, 3, 2, 16//group), dtype=object)
    for batch, row, head, g in np.ndindex(result.shape):
        total = sum(int(left[batch, row, k])*int(right[batch, head, k])
                    for k in range(g*group, (g+1)*group))
        result[batch, row, head, g] = wrap_integer(total, bits, signed)
    return result


def guarded(size, dtype, sentinel=-12345):
    storage = np.full(size+4, sentinel, dtype=dtype)
    return storage, storage[2:-2]


def save(root, name, **arrays):
    np.savez(root / (name + ".npz"), **arrays)


def native_evidence(compiled, root, name, operations):
    canonical = compiled.asm["weftir"]
    for operation in operations:
        assert "weft_kernel." + operation in canonical, "public frontend did not emit " + operation
    arch = compiled.metadata.target.arch
    stages = ("weftir", "x86ir", "llir") if arch == "x86_64" else ("weftir", "riscvir", "c")
    paths = {}
    for stage in stages:
        path = root / (name + "." + stage)
        path.write_text(compiled.asm[stage])
        paths[stage] = str(path)
    evidence = {"operations": operations, "stages": paths, "kernel_abi": compiled.metadata.kernel_abi,
                "native_build": compiled.metadata.native_build}
    if arch == "x86_64":
        evidence["physical_schedules"] = compiled.metadata.compilation["array_schedules"]
        if "uniform_validity_arrays" in compiled.metadata.compilation:
            evidence["uniform_validity_arrays"] = compiled.metadata.compilation["uniform_validity_arrays"]
    return evidence


def multi_axis_mac(root, options, close):
    from numerical_primitive_kernels import grouped_mac, grouped_mac_float, aligned_integer_pairs, byte_dot_tail, equal_extent_axes

    cases = (
        (0, 2, 16, True, np.int16,
         [-128, -128, -17, -1, 0, 1, 63, 127, 127, 126, -64, 7, 42, -99, 3, 8],
         [255, 255, 128, 1, 0, 254, 127, 255, 2, 129, 128, 11, 253, 17, 64, 250]),
        (1, 4, 16, False, np.uint16,
         [255, 255, 255, 255, 128, 0, 1, 127, 254, 17, 200, 3, 129, 250, 7, 64],
         [255, 255, 255, 255, 1, 128, 254, 0, 127, 3, 201, 9, 250, 64, 128, 17]),
        (2, 4, 32, True, np.int32,
         [-32768, -32768, -32768, -32768, 32767, -1, 0, 1, 16384, -16384, 30000, -30000, 7, 31, -99, 127],
         [-32768, -32768, -32768, -32768, 32767, 1, -1, 0, -16384, 16384, -30000, 30000, 17, -31, 99, -127]),
        (3, 4, 32, True, np.int32,
         [255, 255, 255, 255, 128, 0, 1, 127, 254, 17, 200, 3, 129, 250, 7, 64],
         [-128, -128, -128, -128, 127, 127, 0, -1, -77, 33, -64, 17, 51, -101, 11, -31]),
        (4, 4, 32, True, np.int32,
         [-128, -128, -128, -128, 127, -1, 0, 1, -64, 63, -7, 17, 31, -111, 64, -32],
         [-128, -128, -128, -128, 127, 127, 0, -1, -77, 33, -64, 17, 51, -101, 11, -31]),
        (5, 4, 32, True, np.int32,
         [255, 255, 255, 255, 128, 0, 1, 127, 254, 17, 200, 3, 129, 250, 7, 64],
         [255, 255, 255, 255, 1, 128, 254, 0, 127, 3, 201, 9, 250, 64, 128, 17]),
    )
    records = []
    for mode, group, bits, signed, dtype, a, b in cases:
        left = np.stack([np.stack([np.roll(a, batch*5+row*3) for row in range(3)])
                         for batch in range(2)]).astype(np.int32)
        right = np.stack([np.stack([np.roll(b, batch*2+head*5) for head in range(2)])
                          for batch in range(2)]).astype(np.int32)
        original_left, original_right = left.copy(), right.copy()
        expected = mac_oracle(left, right, group, bits, signed).astype(dtype)
        sentinel = -12345 if signed else 65431
        storage, output = guarded(expected.size, dtype, sentinel)
        name = f"grouped-mac-{mode}"
        save(root, name, left=left, right=right, expected=expected)
        compiled = grouped_mac[(1,)](left, right, output, MODE=mode, **options)
        save(root, name, left=original_left, right=original_right, actual_left=left, actual_right=right,
             expected=expected, actual=output.reshape(expected.shape), output_storage=storage)
        evidence = native_evidence(compiled, root, name, ["mac_groups"])
        exact(output.reshape(expected.shape), expected)
        exact(left, original_left)
        exact(right, original_right)
        assert np.all(storage[[0, 1, -2, -1]] == sentinel), "grouped MAC overwrote guard"
        records.append({"mode": mode, "group": group, "accumulator_bits": bits, "signed": signed,
                        "overflow": "wrap", "shape": list(expected.shape), "evidence": evidence})
    left = ((np.arange(72, dtype=np.float32)-36)/17).reshape(2, 3, 12)
    right = ((np.arange(48, dtype=np.float32)*7 % 53-26)/23).reshape(2, 2, 12)
    original_left, original_right = left.copy(), right.copy()
    expected = np.empty((2, 3, 2, 4), dtype=np.float32)
    for batch, row, head, g in np.ndindex(expected.shape):
        expected[batch, row, head, g] = sum(float(left[batch, row, k])*float(right[batch, head, k])
                                          for k in range(g*3, (g+1)*3))
    storage, output = guarded(expected.size, np.float32)
    save(root, "grouped-mac-f32-n3", left=left, right=right, expected=expected)
    compiled = grouped_mac_float[(1,)](left, right, output, **options)
    save(root, "grouped-mac-f32-n3", left=original_left, right=original_right, actual_left=left,
         actual_right=right, expected=expected, actual=output.reshape(expected.shape), output_storage=storage)
    evidence = native_evidence(compiled, root, "grouped-mac-f32-n3", ["mac_groups"])
    close(output.reshape(expected.shape), expected)
    exact(left, original_left)
    exact(right, original_right)
    assert np.all(storage[[0, 1, -2, -1]] == -12345)
    records.append({"mode": "f32", "group": 3, "accumulator": "f32", "shape": list(expected.shape),
                    "oracle": "float64 products and sum rounded to f32", "tolerance": "existing runtime close",
                    "evidence": evidence})
    for mode, a, b, bits in (
        (0, [15, 15, 0, 1, 14, 7, 3, 8], [-128, -128, 127, 127, -1, 0, 1, 64], 16),
        (1, [255, 255, 255, 255, 128, 0, 1, 127], [-128, -128, 127, 127, -1, 0, 1, 64], 16),
        (2, [-32768, -32768, 32767, 32767, 0, -1, 1, 16384], [-32768, -32768, 32767, 32767, -1, 1, 1, 16384], 32),
        (3, [0, 1, -1, 7, -7, 8, 1, 1], [-32768, -1, 32767, 2, -7, 8, -13, 42], 32),
        (4, [255, 255, 127, 127, 128, 0, 1, 127], [-128, -128, 127, 127, -1, 0, 1, 64], 16),
        (5, [255, 255, 128, 128, 127, 0, 1, 127], [-128, -128, 127, 127, -1, 0, 1, 64], 16),
        (6, [255, 255, 129, 129, 128, 0, 1, 127], [-128, -128, 127, 127, -1, 0, 1, 64], 16),
        (7, [12, 12, 12, 12, 15, 0, 1, 127], [-128, -128, 127, 127, -1, 0, 1, 64], 16),
    ):
        left, right = np.tile(a, 4)[:31].astype(np.int32), np.tile(b, 4)[:31].astype(np.int32)
        mask = {4: 127, 5: 128, 6: 129}.get(mode)
        expected = np.array([wrap_integer(sum((int(left[k] > 0) if mode == 3 else
                                              ((int(left[k]) & 15) * 10) & 127 if mode == 7 else
                                              int(left[k]) & mask if mask is not None else int(left[k])) * int(right[k])
                                             for k in range(i, min(i+2, 31))), bits)
                             for i in range(0, 31, 2)], dtype=np.int32)
        storage, output = guarded(16, np.int32)
        compiled = aligned_integer_pairs[(1,)](left, right, output, MODE=mode, **options)
        name = f"aligned-integer-pairs-{mode}"
        save(root, name, left=left, right=right, expected=expected, actual=output, output_storage=storage)
        evidence = native_evidence(compiled, root, name, ["mac_groups"])
        exact(output, expected)
        assert np.all(storage[[0, 1, -2, -1]] == -12345), "aligned pairs overwrote guard"
        records.append({"mode": name, "shape": [31], "group": 2, "accumulator_bits": bits,
                        "overflow": "wrap", "inactive_last_member": True, "evidence": evidence})
    for rows in (4, 7):
        source = np.arange(rows * 4, dtype=np.int32).reshape(rows, 4) - 13
        storage, output = guarded(rows, np.int32)
        compiled = equal_extent_axes[(1,)](source, output, **options)
        expected = 2 * source.sum(axis=1, dtype=np.int32)
        exact(output, expected)
        exact(storage[[0, 1, -2, -1]], np.full(4, -12345, np.int32))
        name = f"equal-extent-helper-axes-{rows}"
        save(root, name, source=source, expected=expected, actual=output)
        records.append({"mode": name, "shape": [rows, 4],
                        "evidence": native_evidence(compiled, root, name, ["level"])})
    for length in (0, 1, 3, 17, 65, 95, 96):
        left = np.resize(np.array([-128, 127, -1, 0, 19, -67], np.int8), length)
        right = np.resize(np.array([-128, -128, 127, 17, -99], np.int8), length)
        storage, output = guarded(24, np.int32)
        total_storage, total = guarded(1, np.int32)
        expected = np.full(24, -12345, np.int32)
        for start in range(0, length, 4):
            expected[start//4] = np.sum(left[start:start+4].astype(np.int64)*right[start:start+4])
        compiled = byte_dot_tail[(1,)](left, right, output, total, **options)
        expected_total = int(np.dot(left.astype(np.int64), right.astype(np.int64))) if length else -12345
        exact(output, expected); exact(total, np.array([expected_total], np.int32))
        assert np.all(storage[[0, 1, -2, -1]] == -12345)
        assert np.all(total_storage[[0, 1, -2, -1]] == -12345)
        name = f"byte-dot-tail-{length}"
        save(root, name, left=left, right=right, actual=output, expected=expected, total=total)
        records.append({"mode": name, "shape": [length], "group": 4,
                        "evidence": native_evidence(compiled, root, name, ["mac_groups"])})
    return {"shared_batch_and_distinct_free_axes": True, "integer_cases_exact": True, "subcases": records}


def multi_axis_pairs(root, options):
    from numerical_primitive_kernels import fold_pairs

    records = []
    patterns = ([-32768, -32768, 32767, 32767, -1, 1, -127, 128, 0, -99, 30000, -30000, 7, 19, -5, -31],
                [65535, 65535, 32768, 32768, 0, 1, 127, 128, 65431, 99, 60000, 30000, 7, 19, 5, 31])
    for unsigned, pattern in enumerate(patterns):
        source = np.stack([np.stack([np.roll(pattern, batch*3+row) for row in range(3)])
                           for batch in range(2)]).astype(np.int32)
        original = source.copy()
        expected = np.empty((2, 3, 8), dtype=np.int32)
        for batch, row, pair in np.ndindex(expected.shape):
            expected[batch, row, pair] = int(source[batch, row, pair*2])+int(source[batch, row, pair*2+1])
        storage, output = guarded(expected.size, np.int32)
        name = f"sum-pairs-{unsigned}"
        save(root, name, source=source, expected=expected)
        compiled = fold_pairs[(1,)](source, output, UNSIGNED=bool(unsigned), **options)
        save(root, name, source=original, actual_source=source, expected=expected,
             actual=output.reshape(expected.shape), output_storage=storage)
        evidence = native_evidence(compiled, root, name, ["fold2"])
        exact(output.reshape(expected.shape), expected)
        exact(source, original)
        assert np.all(storage[[0, 1, -2, -1]] == -12345)
        records.append({"input": "u16" if unsigned else "i16", "output": "i32", "shape": [2, 3, 8], "evidence": evidence})
    return {"integer_exact": True, "subcases": records}


def partial_groups(root, options):
    from numerical_primitive_kernels import grouped_tail

    n = 35
    x_storage, x = guarded(n, np.int8, 85)
    w_storage, weights = guarded(n, np.uint8, 177)
    x[:] = np.resize(np.array([-128, 127, -17, 9, 31, -63, 1], dtype=np.int8), n)
    weights[:] = np.resize(np.array([255, 128, 1, 127, 254, 3], dtype=np.uint8), n)
    original_x, original_w = x_storage.copy(), w_storage.copy()
    group_storage, groups = guarded(16, np.int16)
    pair_storage, pairs = guarded(32, np.int32)
    count_storage, counts = guarded(4, np.int32)
    counts[:] = 0
    expected_groups, expected_pairs = groups.copy(), pairs.copy()
    active_groups = [wrap_integer(sum(int(x[k])*int(weights[k]) for k in range(start, min(start+4, n))), 16)
                     for start in range(0, n, 4)]
    active_pairs = [sum(int(x[k]) for k in range(start, min(start+2, n))) for start in range(0, n, 2)]
    expected_groups[:len(active_groups)] = active_groups
    expected_pairs[:len(active_pairs)] = active_pairs
    expected_counts = np.zeros(4, dtype=np.int32)
    for value in active_groups: expected_counts[value & 1] += 1
    for value in active_pairs: expected_counts[2+(value & 1)] += 1
    save(root, "grouped-tail", source=original_x, weights=original_w, expected_groups=expected_groups,
         expected_pairs=expected_pairs, expected_counts=expected_counts)
    compiled = grouped_tail[(1,)](x, weights, groups, pairs, counts, **options)
    save(root, "grouped-tail", source=original_x, actual_source=x_storage, weights=original_w,
         actual_weights=w_storage, expected_groups=expected_groups, actual_groups=group_storage,
         expected_pairs=expected_pairs, actual_pairs=pair_storage, expected_counts=expected_counts, actual_counts=count_storage)
    evidence = native_evidence(compiled, root, "grouped-tail", ["mac_groups", "fold2"])
    for actual, expected in ((groups, expected_groups), (pairs, expected_pairs), (counts, expected_counts),
                             (x_storage, original_x), (w_storage, original_w)):
        exact(actual, expected)
    for storage in (group_storage, pair_storage, count_storage):
        assert np.all(storage[[0, 1, -2, -1]] == -12345)
    return {"N": n, "block": 32, "active_groups": 9, "active_pairs": 18,
            "partial_group_terms": 3, "partial_pair_terms": 1, "inactive_terms": "zero identity",
            "output_activity": "any valid term", "invalid_outputs_preserve_sentinel": True,
            "unmasked_pointer_store_and_atomic": True, "counts": counts.tolist(), "evidence": evidence}


def lookup_contract(root, options, compile_target=None):
    from numerical_primitive_kernels import lookup_snapshot, lookup_tail, lookup_byte_bits, lookup_float_bits, named_selection
    from numerical_primitive_kernels import lookup_strided, lookup_effect_barrier, lookup_runs, lookup_record_rows, lookup_record_strided, indexed_wide
    from numerical_primitive_kernels import lookup_repeated

    artifacts = {}
    def execute(kernel, *arguments, **constants):
        if compile_target is None:
            return kernel[(1,)](*arguments, **constants, **options)
        from triton import compile
        from triton.compiler import ASTSource
        from triton.backends.compiler import CPUTarget
        # Pointer-only outputs retain the actual input scalar type in AOT.
        signature = {}
        if kernel in (lookup_byte_bits, lookup_float_bits):
            dtype = "i8" if kernel is lookup_byte_bits else "fp32"
            signature = {"Old": "*" + dtype, "New": "*" + dtype}
        elif kernel is lookup_snapshot:
            signature = {name: "*i32" for name in ("Old", "New", "Scalar")}
        elif kernel is lookup_tail:
            signature = {name: "*i32" for name in ("Out", "Counts")}
        key = kernel.__name__, tuple(sorted(constants.items()))
        if key not in artifacts:
            artifacts[key] = compile(ASTSource(kernel, signature, constants),
                                     target=CPUTarget(backend="weft", **compile_target), options=options)
        compiled = artifacts[key]
        compiled[(1,)](*arguments)
        return compiled

    records, evidence = [], []
    for extent in (16, 128, 8192):
        if extent == 16:
            table = np.array([101, -31, 17, 0, 99, -8, 213, 7, -100, 5, 83, -77, 19, 30, -1, 54], dtype=np.int32)
            indices = np.array([[15, 0, 7, 3, 3, 12, 1, 8], [2, 14, 6, 5, 9, 4, 13, 10]], dtype=np.uint32)
            scalar_index, stride, offset, suffix = 3, 7, 3, ""
        else:
            table = (np.arange(extent, dtype=np.int32)*37 % 509)-254
            indices = np.array([[extent-1, 0, extent//2, 3, 3, extent-4, 1, extent//4],
                                [extent//3, extent-2, extent//2+6, 5, extent//4+9, 4, extent-3, extent//2-1]], dtype=np.uint32)
            scalar_index, stride, offset, suffix = extent-1, 37, extent-1, f"-table{extent}"
        original, initial_indices = table.copy(), indices.copy()
        old_storage, old = guarded(16, np.int32)
        new_storage, new = guarded(16, np.int32)
        scalar_storage, scalar = guarded(1, np.int32)
        expected_old = original[indices]
        expected_new = expected_old + 1000
        snapshot_name, tail_name = "lookup-snapshot"+suffix, "lookup-tail"+suffix
        save(root, snapshot_name, table=original, indices=indices, expected_old=expected_old, expected_new=expected_new)
        compiled = execute(lookup_snapshot, table, indices, old, new, scalar,
                           TABLE_SIZE=extent, SCALAR_INDEX=scalar_index)
        save(root, snapshot_name, table=original, actual_table=table, indices=initial_indices,
             actual_indices=indices, expected_old=expected_old, actual_old=old.reshape(2, 8),
             expected_new=expected_new, actual_new=new.reshape(2, 8), actual_scalar=scalar)
        snapshot_evidence = native_evidence(compiled, root, snapshot_name, ["lookup"])
        exact(old.reshape(2, 8), expected_old)
        exact(new.reshape(2, 8), expected_new)
        exact(table, original+1000)
        exact(indices, initial_indices)
        assert scalar[0] == original[scalar_index]
        for storage in (old_storage, new_storage, scalar_storage):
            assert np.all(storage[[0, 1, -2, -1]] == -12345)

        table = original.copy()
        index_storage, indices = guarded(35, np.uint32, 2**32-1)
        indices[:] = (np.arange(35, dtype=np.uint32)*stride+offset) % extent
        original_indices = index_storage.copy()
        output_storage, output = guarded(64, np.int32)
        count_storage, counts = guarded(4, np.int32)
        counts[:] = 0
        expected = output.copy(); expected[:35] = original[indices]
        expected_counts = np.bincount(original[indices] & 3, minlength=4).astype(np.int32)
        save(root, tail_name, table=original, indices=original_indices, expected=expected, expected_counts=expected_counts)
        compiled = execute(lookup_tail, table, indices, output, counts, TABLE_SIZE=extent)
        save(root, tail_name, table=original, actual_table=table, indices=original_indices,
             actual_indices=index_storage, expected=expected, actual=output_storage,
             expected_counts=expected_counts, actual_counts=count_storage)
        tail_evidence = native_evidence(compiled, root, tail_name, ["lookup"])
        for actual, wanted in ((output, expected), (counts, expected_counts), (table, original), (index_storage, original_indices)):
            exact(actual, wanted)
        for storage in (output_storage, count_storage):
            assert np.all(storage[[0, 1, -2, -1]] == -12345)
        assert int(counts.sum()) == 35, "inactive lookup lane produced an atomic effect"
        evidence.extend((snapshot_evidence, tail_evidence))
        records.append({"table_extent": extent, "table_bits": extent*32, "scalar_index": scalar_index,
                        "pointer_effect_count": int(counts.sum()), "counts": counts.tolist(),
                        "inactive_tail_elements": 29, "evidence": [snapshot_evidence, tail_evidence]})
    sparse_records = []
    for shape in ((3, 37), (16, 4096)):
        dense = (np.arange(np.prod(shape), dtype=np.int32).reshape(shape) * 17) - 997
        backing = np.zeros((shape[0], shape[1]*2+3), np.int32)
        backing[:, 1:shape[1]*2+1:2] = dense
        for layout, table in (("dense", dense), ("gapped", backing[:, 1:shape[1]*2+1:2]),
                              ("transposed", dense.T.copy().T), ("broadcast", np.broadcast_to(dense[:1], shape))):
            indices = ((np.arange(35, dtype=np.uint32)*137 + np.prod(shape)-1) % np.prod(shape)).astype(np.uint32)
            storage, output = guarded(35, np.int32)
            compiled = execute(lookup_strided, table, indices, output)
            exact(output, table.reshape(-1)[indices])
            exact(storage[[0, 1, -2, -1]], np.full(4, -12345, np.int32))
            proof = native_evidence(compiled, root, f"lookup-strided-{shape[0]}-{shape[1]}-{layout}", ["lookup"])
            sparse_records.append({"shape": list(shape), "layout": layout, "evidence": proof})
    # The index read aliases the table. Both reads may share a read-only
    # interval; the existing effect-barrier case below must still retain its
    # snapshot across a write.
    table = ((np.arange(256, dtype=np.int32) * 137) % 256).reshape(8, 32)
    indices = table.reshape(-1)[3:38].view(np.uint32)
    storage, output = guarded(35, np.int32)
    compiled = execute(lookup_strided, table, indices, output)
    exact(output, table.reshape(-1)[indices])
    exact(storage[[0, 1, -2, -1]], np.full(4, -12345, np.int32))
    evidence.append(native_evidence(compiled, root, "lookup-aliasing-index-read", ["lookup"]))
    for step in (1, 3):
        table = (np.arange(257 * step, dtype=np.int32) * 37 - 997)[::step]
        indices = np.array([256, 3], np.uint32)
        storage, output = guarded(35, np.int32)
        compiled = execute(lookup_repeated, table, indices, output)
        exact(output, table[indices[(np.arange(35) % 16) // 8]])
        exact(storage[[0, 1, -2, -1]], np.full(4, -12345, np.int32))
        evidence.append(native_evidence(compiled, root, f"lookup-repeated-stride{step}", ["lookup"]))
    table = np.arange(65536, dtype=np.int32) + 41
    output = np.full(2, -12345, np.int32)
    compiled = execute(lookup_effect_barrier, table, output)
    exact(output, np.array([41, -77], np.int32))
    exact(table[1:], np.arange(1, 65536, dtype=np.int32) + 41)
    evidence.append(native_evidence(compiled, root, "lookup-effect-barrier", ["lookup"]))
    for step in (1, 2):
        backing = (np.arange(256*step) % 251 - 125).astype(np.int8)
        table = backing[::step]
        bases = np.array([0, 47, 241, 249, 255], np.uint8)
        output = np.empty((5, 16), np.int8)
        compiled = execute(lookup_runs, table, bases, output)
        indices = (bases[:, None] + np.arange(16, dtype=np.uint8)[None, :]).astype(np.uint8)
        exact(output, table[indices])
        evidence.append(native_evidence(compiled, root, f"lookup-runs-stride{step}", ["lookup"]))
    for stride in (1, 3):
        storage = (np.arange(1024 * 8 * stride) % 251 - 125).astype(np.int8)
        table = storage[::stride].reshape(1024, 8)
        for wrap in (False, True):
            for rows, columns in ((7, 8), (5, 3), (0, 8)):
                bases = np.array([0, 1023, 255, 8193, 17, 500, 65535], np.uint16)[:rows]
                output = np.full((rows, columns), -123, np.int8)
                kernel = lookup_record_rows if stride == 1 else lookup_record_strided
                compiled = execute(kernel, table, bases, output, WRAP=wrap)
                indices = (bases.astype(np.uint32)[:, None] & 1023) * 8 + np.arange(columns, dtype=np.uint32)
                if wrap:
                    indices = (indices + 253).astype(np.uint8).astype(np.uint32)
                exact(output, table.reshape(-1)[indices])
                evidence.append(native_evidence(compiled, root,
                    f"lookup-record-rows-{stride}-{wrap}-{rows}-{columns}", ["lookup"]))
    source = np.arange(257, dtype=np.float32) / 8 - 7
    bytes_ = (np.arange(257) % 256 - 128).astype(np.int8)
    old_source, old_bytes = source.copy(), bytes_.copy()
    indices = ((np.arange(71) * 53 + 256) % 257).astype(np.uint32)
    output, integer = np.empty(71, np.float32), np.empty(71, np.int32)
    compiled = execute(indexed_wide, source, bytes_, indices, output, integer)
    exact(output, old_source[indices]); exact(integer, old_bytes[indices].astype(np.int32))
    exact(source, old_source+1); exact(bytes_, old_bytes+np.int8(1))
    evidence.append(native_evidence(compiled, root, "indexed-wide-snapshot", ["extract"]))
    matrix = np.arange(15, dtype=np.float32).reshape(3, 5) - 7
    rows = np.array([2, -3, 17], np.int32)
    columns = np.array([0.5, -1.5, 2.5, -3.5, 4.5], np.float32)
    output = np.full((3, 5), -111, np.float32)
    transposed = np.full((5, 3), -111, np.float32)
    compiled = execute(named_selection, matrix, rows, columns, output, transposed)
    exact(output, np.where(matrix >= 0, rows[:, None], columns[None, :]))
    exact(transposed, np.where(matrix >= 0, columns[None, :], rows[:, None]).T)
    evidence.append(native_evidence(compiled, root, "named-selection", ["select"]))
    bit_records = []
    for dtype, kernel in ((np.int8, lookup_byte_bits), (np.float32, lookup_float_bits)):
        for extent in (3, 8, 16, 17):
            if dtype is np.int8:
                original = np.resize(np.array([-128, -127, -1, 0, 1, 17, 63, 127], np.int8), extent)
            else:
                bits = np.array([0x80000000, 0, 1, 0x007fffff, 0x7f800000,
                                 0xff800000, 0x7fc12345, 0x3f800000], np.uint32)
                original = np.resize(bits, extent).view(np.float32)
            # More than one native packet plus a non-power-of-two logical tail.
            n, block = 71, 96
            indices = ((np.arange(n, dtype=np.uint32)*7 + 2) % extent)
            table = original.copy()
            before_storage, before = guarded(n + 13, dtype, 11)
            after_storage, after = guarded(n + 13, dtype, 11)
            compiled = execute(kernel, table, indices, before, after, TABLE_SIZE=extent, BLOCK=block)
            unsigned = np.uint8 if dtype is np.int8 else np.uint32
            exact(before[:n].view(unsigned), original[indices].view(unsigned))
            exact(after[:n], np.zeros(n, dtype))
            exact(table, np.zeros(extent, dtype))
            for storage in (before_storage, after_storage):
                exact(storage[:2], np.full(2, 11, dtype))
                exact(storage[n+2:], np.full(15, 11, dtype))
            name = f"lookup-bits-{np.dtype(dtype).name}-{extent}"
            save(root, name, original=original, indices=indices, before=before_storage, after=after_storage)
            proof = native_evidence(compiled, root, name, ["lookup"])
            evidence.append(proof)
            bit_records.append({"type": np.dtype(dtype).name, "table_extent": extent,
                                "active": n, "block": block, "bits_preserved": True, "evidence": proof})
    return {"bounds": "in_bounds", "active_indices_in_bounds": True, "index_dtype": "u32",
            "snapshot_before_table_write": True, "implicit_admit_after_table_write": True,
            "scalar_and_shaped_indices": True, "index_shape": [2, 8], "tail_N": 35, "block": 32,
            "pointer_effect_count": 35, "lookup_mask_or_fill_arguments": False,
            "table_extents": [16, 128, 8192], "variants": records, "bit_variants": bit_records,
            "sparse_strided_tables": sparse_records, "single_consumer_effect_barrier": True,
            "contiguous_lookup_wrap_and_stride": True, "wide_snapshot_across_write": True,
            "execution_mode": "aot" if compile_target else "jit", "evidence": evidence}


def scalar_validity_contract(root, options):
    from numerical_primitive_kernels import scalar_validity
    from helper_native import capture

    n = 35
    source_storage, source = guarded(n, np.int8, 85)
    source[:] = np.resize(np.array([-7, -3, -1, 0, 1, 3, 7], dtype=np.int8), n)
    original = source_storage.copy()
    output_storage, output = guarded(64, np.int32)
    count_storage, counts = guarded(4, np.int32)
    counts[:] = 0
    increments = np.array([1, 4, 6], dtype=np.int64)[np.arange(n) % 3]
    caller_increments = 6 + np.where(np.arange(n) % 2 == 0, 3, -2)
    expected = output.copy()
    expected[:n] = (source.astype(np.int64) + increments + caller_increments + 5) & 3
    expected_counts = np.bincount(expected[:n], minlength=4).astype(np.int32)
    save(root, "scalar-validity", source=original, expected=expected, expected_counts=expected_counts)
    compiled = scalar_validity[(1,)](source, output, counts, **options)
    save(root, "scalar-validity", source=original, actual_source=source_storage, expected=expected,
         actual=output_storage, expected_counts=expected_counts, actual_counts=count_storage)
    evidence = capture(compiled, root, "scalar-validity", minimum_noinline=1)
    exact(output, expected)
    exact(counts, expected_counts)
    exact(source_storage, original)
    for storage in (output_storage, count_storage):
        assert np.all(storage[[0, 1, -2, -1]] == -12345)
    assert int(counts.sum()) == n, "inactive scalar produced an atomic effect"
    return {"N": n, "block": 32, "scalar_slots": 64, "valid_scalar_effects": n,
            "invalid_tail_elements": 29, "helper_paths": ["if-return", "for-return", "while-return"],
            "source_dtype": "i8", "helper_dtype": "i16", "caller_dtype": "i32",
            "caller_operations": ["widen", "for-carry", "while-carry", "runtime-if", "add", "and", "store", "atomic_add"],
            "explicit_effect_mask": False, "counts": counts.tolist(), "integer_exact": True,
            "oracle": "helper [1,4,6] + caller 6+[3,-2] + 5; valid-input histogram", "evidence": evidence}


def streaming_store_contract(root, options, compile_target):
    from numerical_primitive_kernels import streaming_copy, streaming_overlap
    from triton import compile
    from triton.compiler import ASTSource
    from triton.backends.compiler import CPUTarget

    target = CPUTarget(backend="weft", **compile_target)
    compiled = compile(ASTSource(streaming_copy, {}, {"BLOCK": 64}), target=target, options=options)
    shared = compile(ASTSource(streaming_overlap, {}, {}), target=target, options=options)
    lengths = (0, 1, 15, 16, 31, 64, 257, 511, 512, 513, 576, 1025, 4097)
    for n in lengths:
        for shift in (0, 4, 8, 1):
            source = np.resize(np.array([0, 0x80000000, 1, 0x7f800000, 0xff800000,
                                         0x7fc12345, 0x3f800000], np.uint32), n).view(np.float32)
            backing = np.full(n + 48, -777, np.float32)
            start = ((-backing.ctypes.data) % 64) // 4 + 16 + shift
            output = backing[start:start+n]
            compiled[(1,)](source, output)
            exact(output.view(np.uint32), source.view(np.uint32))
            exact(backing[:start], np.full(start, -777, np.float32))
            exact(backing[start+n:], np.full(backing.size-start-n, -777, np.float32))
    backing = np.arange(1026, dtype=np.float32)
    expected = backing.copy()
    for start in range(0, 1025, 64):
        end = min(start+64, 1025)
        expected[start+1:end+1] = expected[start:end].copy()
    shared[(1,)](backing[:-1], backing[1:])
    exact(backing, expected)
    windows = []
    for block in (256, 1024, 4096):
        wide = compile(ASTSource(streaming_copy, {}, {"BLOCK": block}), target=target, options=options)
        for n in (0, block - 1, block, block + 1, 3 * block, 3 * block + 1, 5 * block + 1):
            source = (np.arange(n, dtype=np.uint32) * 65537).view(np.float32)
            backing = np.full(n + 2, -777, np.float32)
            wide[(1,)](source, backing[1:-1])
            exact(backing[1:-1].view(np.uint32), source.view(np.uint32))
            exact(backing[[0, -1]], np.array([-777, -777], np.float32))
        ir = wide.asm["x86ir"]
        distance = min(4, 32768 // (4 * block))
        assert f"lookahead_iterations = {distance} : i64" in ir, "wide stream scheduling was not selected"
        windows.append({"block": block, "lookahead_iterations": distance,
                        "evidence": native_evidence(wide, root, f"streaming-window-{block}", [])})
    return {"lengths": list(lengths), "aligned_and_unaligned": True,
            "stream_prefetch_selected": compiled.metadata.target.arch == "x86_64" and
                                        "weft.x86.prefetched_streams" in compiled.asm["x86ir"],
            "overlapping_input_preserves_order": True, "exact_payload_bits": True, "wide_windows": windows,
            "evidence": [native_evidence(compiled, root, "streaming-copy", []),
                         native_evidence(shared, root, "streaming-overlap", [])]}


def cpu_tuning_contract(root, options, compile_target):
    from triton import compile
    from triton.compiler import ASTSource
    from triton.backends.compiler import CPUTarget
    from numerical_primitive_kernels import byte_contraction_unsigned
    target = CPUTarget(backend="weft", **compile_target)
    source = ASTSource(byte_contraction_unsigned, {}, {"BM": 4, "BN": 16, "BK": 32})
    selected = {**options, "backend_options": {"tune_cpu": "sapphirerapids"}}
    compiled = compile(source, target=target, options=selected)
    text = compiled.asm["llir"]
    assert '"tune-cpu"="sapphirerapids"' in text, "tuning selection was lost before LLVM code generation"
    base = compile_target["march"].split("+")[0]
    assert f'"target-cpu"="{base}"' in text, "tuning changed the ISA target"
    if base == "x86-64-v3":
        assert "+avx512" not in text and "+amx" not in text, "tuning enabled undeclared ISA"
    for k in (0, 1, 32, 37):
        a = (np.arange(4*k) * 37 % 256).astype(np.uint8).reshape(4, k)
        b = (np.arange(16*k) * 71 % 256 - 128).astype(np.int8).reshape(16, k)
        y = np.full((4, 16), -777, np.int32)
        compiled[(1,)](a, b, y)
        exact(y, a.astype(np.int32) @ b.astype(np.int32).T)
    try:
        compile(source, target=target, options={**options, "backend_options": {"tune_cpu": "not-an-x86-cpu"}})
    except Exception as error:
        if "unknown LLVM x86 tuning CPU" not in str(error):
            raise
    else:
        raise AssertionError("unknown tuning CPU was silently accepted")
    return {"tuning_cpu": "sapphirerapids", "isa_target": compile_target,
            "unknown_cpu_rejected": True, "tails_and_empty_reduction": True,
            "evidence": native_evidence(compiled, root, "cpu-tuning", ["contract"])}


def indexed_packet_contract(root, options, compile_target):
    from triton import compile
    from triton.compiler import ASTSource
    from triton.backends.compiler import CPUTarget
    from numerical_primitive_kernels import indexed_row_packets, indexed_loop_packets

    evidence, records = [], []
    target = CPUTarget(backend="weft", **compile_target)
    for k in (64, 256):
        compiled = compile(ASTSource(indexed_row_packets, {}, {"K": k, "N": 32}), target=target, options=options)
        assert "weft.x86.indexed_packets" in compiled.asm["x86ir"], "snapshot packet selection was not exercised"
        for rows, first in ((0, 0), (4, 0), (5, 32), (5, 50), (4, 240), (5, 250)):
            source = (np.arange(rows*k)*37 % 256).astype(np.uint8).reshape(rows,k)
            saved = source.copy()
            storage, output = guarded(rows*32, np.int32)
            output = output.reshape(rows,32)
            indices = (np.arange(32, dtype=np.uint32) + first) % 256
            active = indices < k
            expected = output.copy()
            expected[:,active] = saved[:,indices[active]]
            compiled[(1,)](source, output, first)
            exact(output, expected)
            exact(source, saved ^ np.uint8(17))
            exact(storage[[0,1,-2,-1]], np.full(4,-12345,np.int32))
            records.append({"rows":rows,"columns":k,"first":first})
        evidence.append(native_evidence(compiled, root, f"indexed-row-{k}", ["extract"]))
    loop = compile(ASTSource(indexed_loop_packets, {}, {"K":256,"N":32}), target=target, options=options)
    for rows in (0,4,5):
        source = (np.arange(rows*256)*37 % 256).astype(np.uint8).reshape(rows,256)
        saved = source.copy()
        output = np.full((rows,32), -12345, np.int32)
        loop[(1,)](source, output)
        exact(output, saved.astype(np.int32).reshape(rows,8,32).sum(axis=1,dtype=np.int32))
        exact(source, saved ^ np.uint8(17))
    evidence.append(native_evidence(loop, root, "indexed-loop", ["extract"]))
    return {"cases":records,"saved_value_after_write":True,"narrow_wrap_and_inactive_lanes":True,
            "bounded_induction_variable":True,"evidence":evidence}


def run_scenario(name, root, options, close, compile_target=None):
    if name == "indexed_packets":
        return indexed_packet_contract(root, options, compile_target)
    if name == "byte_planes":
        return byte_plane_contract(root, options, compile_target)
    if name == "cpu_tuning":
        return cpu_tuning_contract(root, options, compile_target)
    if name == "streaming_stores":
        return streaming_store_contract(root, options, compile_target)
    if name == "bfloat_contractions":
        return bfloat_contraction_contract(root, options, compile_target)
    if name == "byte_contractions":
        return byte_contraction_contract(root, options, compile_target)
    if name == "amx_snapshots":
        return amx_snapshot_contract(root, options, compile_target)
    if name == "word_byte_contractions":
        return word_byte_contraction_contract(root, options, compile_target)
    if name == "amx_mixed_storage":
        return {"bytes": byte_contraction_contract(root, options, compile_target, mixed_storage=True),
                "words": word_byte_contraction_contract(root, options, compile_target, mixed_storage=True)}
    if name == "byte_sum":
        return byte_sum_contract(root, options, compile_target)
    if name == "extrema_and_half":
        return extrema_and_half(root, options, compile_target)
    if name == "lookup":
        return lookup_contract(root, options, compile_target)
    if name == "array_contractions":
        return array_contractions(root, options, close, compile_target)
    if name == "array_packets":
        return array_packets(root, options, close, compile_target)
    if name == "grouped_mac":
        return multi_axis_mac(root, options, close)
    operations = {"sum_pairs": multi_axis_pairs,
                  "grouped_tail": partial_groups,
                  "scalar_validity": scalar_validity_contract}
    if name not in operations:
        raise ValueError("unknown numerical primitive scenario " + name)
    return operations[name](root, options)


def byte_plane_contract(root, options, compile_target):
    from triton import compile
    from triton.compiler import ASTSource
    from triton.backends.compiler import CPUTarget
    from numerical_primitive_kernels import byte_plane_fields
    records = []
    for width in (1, 2, 4):
        for step in (1, 4):
            compiled = compile(ASTSource(byte_plane_fields, {}, {"WIDTH": width, "STEP": step}),
                               target=CPUTarget(backend="weft", **compile_target), options=options)
            for n in (0, 1, 3, 17, 255, 256, 257, 512):
                lane = np.arange(n) % 256
                start = (lane // step) % (9 - width)
                end = ((lane // step) // (9 - width)) % (9 - width)
                storage, output = guarded(n, np.uint8, 0xa5)
                # Every input byte at every demanded field placement, including
                # bits crossing SIMD word boundaries and an inactive final tile.
                for value in range(256):
                    x = np.full(n, value, np.uint8)
                    expected = (((x.astype(np.uint16) >> start) & ((1 << width) - 1)) << end).astype(np.uint8)
                    compiled[(1,)](x, output)
                    exact(output, expected)
                exact(storage[[0, 1, -2, -1]], np.full(4, 0xa5, np.uint8))
                records.append({"width": width, "group": step, "N": n, "input_bytes": 256})
            native_evidence(compiled, root, f"byte-plane-{width}-{step}", ["cast"])
    return {"subcases": records, "integer_exact": True, "tails_and_empty": True}


def bfloat_contraction_contract(root, options, compile_target):
    import torch
    from triton import compile
    from triton.compiler import ASTSource
    from triton.backends.compiler import CPUTarget
    from numerical_primitive_kernels import bfloat_contraction
    matrix = "amx-bf16" in compile_target["march"]
    bm, bn, bk = (32, 32, 512) if matrix else (2, 16, 32)
    compiled = compile(ASTSource(bfloat_contraction, {}, {"BM": bm, "BN": bn, "BK": bk}),
                       target=CPUTarget(backend="weft", **compile_target), options=options)
    records = []
    for m, n, k in ((bm, bn, bk), (bm + 1, bn + 1, bk + 3), (0, bn, bk), (bm, bn, 0)):
        a = ((torch.arange(m*k, dtype=torch.float32) % 13 - 6) / 7).reshape(m, k).bfloat16()
        b = ((torch.arange(n*k, dtype=torch.float32) % 11 - 5) / 9).reshape(n, k).bfloat16()
        storage, output = guarded(m*n, np.float32)
        compiled[(1,)](a, b, output.reshape(m, n))
        expected = a.float().numpy().astype(np.float64) @ b.float().numpy().astype(np.float64).T
        np.testing.assert_allclose(output.reshape(m, n), expected, rtol=3e-5, atol=3e-6)
        exact(storage[[0, 1, -2, -1]], np.full(4, -12345, np.float32))
        records.append({"shape": [m, n, k], "kind": "normal-and-tail"})
    for name, left, right in (("subnormal-input", 1, 227 << 7),
                               ("subnormal-product", 63 << 7, 63 << 7),
                               ("large-and-small", 230 << 7, 30 << 7),
                               ("infinity", 0x7f80, 127 << 7),
                               ("nan", 0x7fc1, 127 << 7)):
        a = torch.zeros((bm, bk), dtype=torch.int16)
        b = torch.zeros((bn, bk), dtype=torch.int16)
        a[:, 0] = left
        b[:, 0] = right
        a, b = a.view(torch.bfloat16), b.view(torch.bfloat16)
        storage, output = guarded(bm * bn, np.float32)
        compiled[(1,)](a, b, output.reshape(bm, bn))
        with np.errstate(invalid="ignore", over="ignore"):
            expected = (a.float().numpy().astype(np.float64) @ b.float().numpy().astype(np.float64).T).astype(np.float32)
        np.testing.assert_array_equal(output.reshape(bm, bn), expected)
        exact(storage[[0, 1, -2, -1]], np.full(4, -12345, np.float32))
        records.append({"shape": [bm, bn, bk], "kind": name})
    return {"cases": records, "evidence": native_evidence(compiled, root, "bfloat-contraction", ["contract"])}


def byte_contraction_contract(root, options, compile_target, mixed_storage=False):
    from numerical_primitive_kernels import byte_contraction_unsigned, byte_contraction_signed, byte_contraction_both_unsigned, byte_contraction_signed_unsigned, partial_axis_identity
    records = []
    matrix = compile_target is not None and "amx-int8" in compile_target["march"]
    tiles = (({"BM": 16, "BN": 16, "BK": 16}, {"BM": 272, "BN": 272, "BK": 16},
              {"BM": 9, "BN": 16, "BK": 528},
              {"BM": 8, "BN": 32, "BK": 32}, {"BM": 32, "BN": 32, "BK": 32},
              {"BM": 16, "BN": 48, "BK": 32},
              {"BM": 16, "BN": 16, "BK": 32}, {"BM": 32, "BN": 32, "BK": 512}) if matrix else
             ({"BM": 4, "BN": 4, "BK": 32}, {"BM": 4, "BN": 16, "BK": 32},
              {"BM": 1, "BN": 128, "BK": 32}))
    if mixed_storage:
        tiles = ({"BM": 272, "BN": 8, "BK": 16}, {"BM": 64, "BN": 32, "BK": 128})
    for kernel, lhs_type, rhs_type in ((byte_contraction_unsigned, np.uint8, np.int8),
                                       (byte_contraction_signed, np.int8, np.int8),
                                       (byte_contraction_both_unsigned, np.uint8, np.uint8),
                                       (byte_contraction_signed_unsigned, np.int8, np.uint8)):
      for tile in tiles:
        compiled = None
        shapes = ((4, 4, 32), (3, 5, 37), (8, 9, 64), (1, 1, 1), (0, 4, 32), (4, 4, 0))
        bm, bn, bk = tile["BM"], tile["BN"], tile["BK"]
        shapes += ((bm, bn, bk), (bm + 1, bn + 1, bk + 3))
        if matrix and bk >= 32:
            shapes += ((bm, bn, 131072),)
        for m, n, k in shapes:
            for uniform in (False, True):
                a = (np.arange(m*k, dtype=np.int64)*37 % 256).astype(lhs_type).reshape(m, k)
                b = (np.arange(n*k, dtype=np.int64)*71 % 256).astype(rhs_type).reshape(n, k)
                if uniform:
                    a.fill(255 if lhs_type == np.uint8 else -128)
                    b.fill(255 if rhs_type == np.uint8 else -128)
                expected = (a.astype(np.int64) @ b.astype(np.int64).T).astype(np.int32)
                storage, output = guarded(m*n, np.int32)
                output = output.reshape(m, n)
                if compile_target is None:
                    current = kernel[(1,)](a, b, output, **tile, **options)
                    if compiled is None:
                        compiled = current
                    assert current is compiled, "byte contraction specialized runtime extents"
                else:
                    if compiled is None:
                        from triton import compile
                        from triton.compiler import ASTSource
                        from triton.backends.compiler import CPUTarget
                        compiled = compile(ASTSource(kernel, {}, tile), target=CPUTarget(backend="weft", **compile_target), options=options)
                    compiled[(1,)](a, b, output)
                save(root, f"{kernel.__name__}-{bm}-{bn}-{bk}-{m}-{n}-{k}-{uniform}", lhs=a, rhs=b, expected=expected, actual=output)
                exact(output, expected)
                exact(storage[[0, 1, -2, -1]], np.full(4, -12345, np.int32))
                records.append({"kernel": kernel.__name__, "tile": tile, "M": m, "N": n, "K": k, "uniform_extreme": uniform})
        native_evidence(compiled, root, f"{kernel.__name__}-{bm}-{bn}-{bk}", ["contract"])
        if matrix:
            assert "amx-int8" in compiled.metadata.single_instance_abi["used_extensions"], "integer matrix route was not selected"
    source = np.arange(17*17, dtype=np.float32).reshape(17, 17)
    storage, output = guarded(source.size, np.float32)
    output = output.reshape(source.shape)
    if compile_target is None:
        partial_axis_identity[(1,)](source, output, **options)
    else:
        from triton import compile
        from triton.compiler import ASTSource
        from triton.backends.compiler import CPUTarget
        partial = compile(ASTSource(partial_axis_identity, {}, {}), target=CPUTarget(backend="weft", **compile_target), options=options)
        partial[(1,)](source, output)
    exact(output, source*2)
    exact(storage[[0, 1, -2, -1]], np.full(4, -12345, np.float32))
    return {"subcases": records, "oracle": "independent int64 matrix product", "tails_and_guards": True,
            "equal_extents_keep_distinct_local_axes": True}


def word_byte_contraction_contract(root, options, compile_target, mixed_storage=False):
    from numerical_primitive_kernels import word_byte_contraction, uword_byte_contraction, byte_word_contraction, byte_uword_contraction, word_word_contraction, byte_scaled_word_contraction
    from triton import compile
    from triton.compiler import ASTSource
    from triton.backends.compiler import CPUTarget
    records = []
    for kernel, lhs, rhs in ((word_byte_contraction, np.int16, np.int8), (uword_byte_contraction, np.uint16, np.uint8),
                             (byte_word_contraction, np.int8, np.int16), (byte_uword_contraction, np.uint8, np.uint16),
                             (word_word_contraction, np.int16, np.int16)):
        if mixed_storage and kernel not in (byte_word_contraction, byte_uword_contraction):
            continue
        words = kernel is word_word_contraction
        tile = ({"BM": 32, "BN": 16, "BK": 256} if mixed_storage else
                {"BM": 16 if words else 8, "BN": 16 if words else 32, "BK": 16 if words else 128})
        compiled = compile(ASTSource(kernel, {}, tile), target=CPUTarget(backend="weft", **compile_target), options=options)
        bm, bn, bk = tile["BM"], tile["BN"], tile["BK"]
        for m,n,k in ((bm,bn,bk),(bm+1,bn+1,bk*2+3),(bm,bn,131072),(0,bn,bk),(bm,bn,0)):
            for extreme in (False,True):
                a=(np.arange(m*k,dtype=np.int64)*8213 % (np.iinfo(lhs).max+1)).astype(lhs).reshape(m,k)
                b=(np.arange(n*k,dtype=np.int64)*371 % (np.iinfo(rhs).max+1)).astype(rhs).reshape(n,k)
                if np.issubdtype(lhs,np.signedinteger): a[::2] *= -1
                if np.issubdtype(rhs,np.signedinteger): b[::2] *= -1
                if extreme:
                    a.fill(np.iinfo(lhs).min if np.issubdtype(lhs,np.signedinteger) else np.iinfo(lhs).max)
                    b.fill(np.iinfo(rhs).min if np.issubdtype(rhs,np.signedinteger) else np.iinfo(rhs).max)
                storage,output=guarded(m*n,np.int32)
                compiled[(1,)](a,b,output.reshape(m,n))
                expected=(a.astype(np.int64) @ b.astype(np.int64).T).astype(np.int32)
                exact(output.reshape(m,n),expected)
                exact(storage[[0,1,-2,-1]],np.full(4,-12345,np.int32))
                records.append({"kernel":kernel.__name__,"shape":[m,n,k],"extreme":extreme})
        native_evidence(compiled,root,kernel.__name__,["contract"])
        if not words:
            assert "amx-int8" in compiled.metadata.single_instance_abi["used_extensions"]
    rows = 32 if mixed_storage else 8
    for mutate in ((False, True) if mixed_storage else (False,)):
        scaled = compile(ASTSource(byte_scaled_word_contraction, {}, {"BM": rows, "BN": 16, "BK": 256, "MUTATE": mutate}),
                         target=CPUTarget(backend="weft", **compile_target), options=options)
        for m, n, k in ((rows, 16, 256), (rows-1, 16, 255), (rows+1, 17, 263), (0, 16, 256), (rows, 16, 0)):
            a = (np.arange(m*k, dtype=np.int64)*71 % 256).astype(np.int8).reshape(m,k)
            b = (np.arange(n*k, dtype=np.int64)*8213 % 65536).astype(np.int16).reshape(n,k)
            s = (np.arange(n*k, dtype=np.int64)*37 + 32768).astype(np.int16).reshape(n,k)
            original = b.copy()
            weights = (b.astype(np.int64)*s.astype(np.int64)).astype(np.int16)
            storage, output = guarded(m*n, np.int32)
            scaled[(1,)](a, b, s, output.reshape(m,n))
            expected = (a.astype(np.int64) @ weights.astype(np.int64).T).astype(np.int32)
            if mutate:
                expected[rows:] = 0
            exact(output.reshape(m,n), expected)
            exact(b, np.zeros_like(b) if mutate and m else original)
            exact(storage[[0,1,-2,-1]], np.full(4,-12345,np.int32))
            records.append({"kernel": "byte_scaled_word_contraction", "shape": [m,n,k],
                            "i16_product_wrap": True, "saved_value_after_source_write": mutate})
        native_evidence(scaled, root, f"byte-scaled-word-{mutate}", ["contract"])
        assert "amx-int8" in scaled.metadata.single_instance_abi["used_extensions"]
    return {"subcases":records,"oracle":"int64 product with exact i16 scale and i32 accumulation wrap"}


def amx_snapshot_contract(root, options, compile_target):
    from numerical_primitive_kernels import byte_contraction_snapshot, word_snapshot_reuse
    from triton import compile
    from triton.compiler import ASTSource
    from triton.backends.compiler import CPUTarget
    records = []
    for mutate, low_mask in ((False, 255), (False, 127), (True, 255), (True, 127)):
        compiled = compile(ASTSource(byte_contraction_snapshot, {}, {"MUTATE": mutate, "BM": 16, "BN": 16,
                                                                  "LOW_MASK": low_mask}),
                           target=CPUTarget(backend="weft", **compile_target), options=options)
        for m, k, stride in ((16, 32, 1), (16, 64, 1), (13, 37, 1), (16, 32, 2), (0, 32, 1), (16, 0, 1)):
            backing = (np.arange(m*(k*stride+17), dtype=np.int64)*37 % 256).astype(np.int8).reshape(m, k*stride+17)
            a = backing[:, :k*stride:stride]
            original, before = a.copy(), backing.copy()
            b = (np.arange(16*k, dtype=np.int64)*71 % 256).astype(np.int8).reshape(16, k)
            storage, output = guarded(m*16, np.int32)
            compiled[(1,)](a, b, output.reshape(m, 16))
            transformed = (original.astype(np.int16) & low_mask).astype(np.int8)
            expected = (transformed.astype(np.int64) @ b.astype(np.int64).T).astype(np.int32)
            exact(output.reshape(m, 16), expected)
            exact(storage[[0, 1, -2, -1]], np.full(4, -12345, np.int32))
            if mutate:
                before[:, :k*stride:stride] = 0
            exact(backing, before)
            records.append({"M": m, "K": k, "row_gap": 17, "element_stride": stride,
                            "mutation": mutate, "low_byte_mask": low_mask})
        native_evidence(compiled, root, f"amx-snapshot-{mutate}-{low_mask}", ["contract"])
        if low_mask != 255:
            continue
        reused = compile(ASTSource(word_snapshot_reuse, {}, {"MUTATE": mutate, "K": 128, "N": 16, "BM": 8}),
                         target=CPUTarget(backend="weft", **compile_target), options=options)
        for m in (0, 8, 16, 17):
            a = (np.arange(m*128,dtype=np.int64)*71 % 256).astype(np.int8).reshape(m,128)
            b = ((np.arange(16*128,dtype=np.int64)*8213 % 65536)-32768).astype(np.int16).reshape(16,128)
            prior = b.copy()
            storage, output = guarded(m*16,np.int32)
            reused[(1,)](a,b,output.reshape(m,16))
            exact(output.reshape(m,16),(a.astype(np.int64) @ prior.astype(np.int64).T).astype(np.int32))
            exact(b,np.zeros_like(b) if mutate else prior)
            exact(storage[[0,1,-2,-1]],np.full(4,-12345,np.int32))
            records.append({"M":m,"K":128,"snapshot_reuse":True,"mutation":mutate})
        native_evidence(reused,root,f"amx-packing-reuse-{mutate}",["contract"])
        assert "weft.x86.reused_amx_packings" in reused.asm["x86ir"], "packing reuse was not exercised"
    return {"subcases": records, "oracle": "saved-input int64 product and untouched storage gaps"}


def byte_sum_contract(root, options, compile_target):
    from numerical_primitive_kernels import byte_sum_tail, byte_sum_rows, integer_sum_rows
    compiled = None
    records = []
    for n in (0, 1, 7, 8, 17, 33, 255, 256, 257):
        for uniform in (False, True):
            x = np.full(n, -128, np.int8) if uniform else ((np.arange(n)*37 % 256)-128).astype(np.int8)
            u = np.full(n, 255, np.uint8) if uniform else (np.arange(n)*71 % 256).astype(np.uint8)
            storage, output = guarded(4, np.int32)
            expected = output.copy()
            if n:
                sx, su = sum(map(int, x)), sum(map(int, u))
                expected[:] = [wrap_integer(sx,16),sx,wrap_integer(su,16),su]
            if compile_target is None:
                current = byte_sum_tail[(1,)](x,u,output,**options)
                if compiled is None: compiled = current
                assert current is compiled, "byte sum specialized the runtime extent"
            else:
                if compiled is None:
                    from triton import compile
                    from triton.compiler import ASTSource
                    from triton.backends.compiler import CPUTarget
                    compiled = compile(ASTSource(byte_sum_tail,{},{}),target=CPUTarget(backend="weft",**compile_target),options=options)
                compiled[(1,)](x,u,output)
            save(root,f"byte-sum-{n}-{uniform}",signed=x,unsigned=u,expected=expected,actual=output)
            exact(output,expected)
            exact(storage[[0,1,-2,-1]],np.full(4,-12345,np.int32))
            records.append({"N":n,"uniform_extreme":uniform})
    for n in (7, 33, 257):
        backing = (np.arange(272)*71 % 256).astype(np.uint8)
        u = backing[:n]
        x = ((np.arange(n)*37 % 256)-128).astype(np.int8)
        output = backing.view(np.int32)[2:6]
        expected_storage = backing.copy()
        sx, su = sum(map(int, x)), sum(map(int, u))
        expected_storage.view(np.int32)[2:6] = [wrap_integer(sx,16),sx,wrap_integer(su,16),su]
        compiled[(1,)](x, u, output)
        exact(backing, expected_storage)
        records.append({"N": n, "unsigned_input_overlaps_output": True})
    evidence=native_evidence(compiled,root,"byte-sum",["reduce"])
    rows_evidence = []
    for tile_rows, block in ((1, 32), (1, 257), (4, 257)):
        rows_compiled = None
        shapes = ((0, 7), (4, 0), (1, 7), (3, 32)) if block == 32 else ((0, 33), (4, 0), (1, 7), (3, 32), (4, 33), (5, 257))
        for m, n in shapes:
            x = ((np.arange(m*n)*37 % 256)-128).astype(np.int8).reshape(m, n)
            u = (np.arange(m*n)*71 % 256).astype(np.uint8).reshape(m, n)
            if m:
                x[-1].fill(-128); u[-1].fill(255)
            storage, output = guarded(m*4, np.int32)
            output = output.reshape(m, 4)
            expected = output.copy()
            if n:
                for row in range(m):
                    sx, su = sum(map(int, x[row])), sum(map(int, u[row]))
                    expected[row] = [wrap_integer(sx, 16), sx, wrap_integer(su, 16), su]
            if compile_target is None:
                current = byte_sum_rows[(1,)](x, u, output, ROWS=tile_rows, BLOCK=block, **options)
                if rows_compiled is None: rows_compiled = current
                assert current is rows_compiled, "row byte sums specialized runtime shapes"
            else:
                if rows_compiled is None:
                    from triton import compile
                    from triton.compiler import ASTSource
                    from triton.backends.compiler import CPUTarget
                    rows_compiled = compile(ASTSource(byte_sum_rows, {}, {"ROWS": tile_rows, "BLOCK": block}),
                        target=CPUTarget(backend="weft", **compile_target), options=options)
                rows_compiled[(1,)](x, u, output)
            exact(output, expected)
            exact(storage[[0, 1,-2,-1]], np.full(4, -12345, np.int32))
            records.append({"M": m, "N": n, "row_tile": tile_rows, "block": block})
        rows_evidence.append(native_evidence(rows_compiled, root, f"byte-sum-rows-{tile_rows}-{block}", ["reduce"]))
        if tile_rows == 1 and block == 32 and "riscvir" in rows_compiled.asm:
            assert '"weft_riscv.rvv_widen_reduce"' in rows_compiled.asm["riscvir"], "singleton row widening reduction was not selected"
    for bits, unsigned in ((8, False), (8, True), (64, False), (64, True)):
        source = np.resize(np.array([-(1 << 63), (1 << 63)-1, -128, 255, 1], np.int64), (5, 257))
        storage, output = guarded(5, np.int64)
        expected = np.array([wrap_integer(wrap_integer(sum(wrap_integer(int(x), bits, not unsigned)
                             for x in row), bits, not unsigned), 64) for row in source], np.int64)
        if compile_target is None:
            width_compiled = integer_sum_rows[(1,)](source, output, BITS=bits, UNSIGNED=unsigned, **options)
        else:
            from triton import compile
            from triton.compiler import ASTSource
            from triton.backends.compiler import CPUTarget
            width_compiled = compile(ASTSource(integer_sum_rows, {}, {"BITS": bits, "UNSIGNED": unsigned}),
                target=CPUTarget(backend="weft", **compile_target), options=options)
            width_compiled[(1,)](source, output)
        exact(output, expected)
        exact(storage[[0,1,-2,-1]], np.full(4, -12345, np.int64))
        rows_evidence.append(native_evidence(width_compiled, root, f"integer-sum-{bits}-{unsigned}", ["reduce"]))
        records.append({"bits": bits, "unsigned": unsigned, "modular_overflow": True, "tail_rows": True})
    return {"subcases":records,"signed_and_unsigned":True,"i16_wrap":True,"empty_preserves_output":True,
            "projected_rows_and_tails": True, "evidence":[evidence, *rows_evidence]}


def extrema_and_half(root, options, compile_target):
    from numerical_primitive_kernels import fixed_number_extrema, scalar_half_widen, array_special_reduce, independent_half_pair, snapshot_half_pair, absolute_maxima, broadcast_half_snapshot
    artifacts = {}
    def execute(kernel, *args):
        if compile_target is None:
            return kernel[(1,)](*args, **options)
        from triton import compile
        from triton.compiler import ASTSource
        from triton.backends.compiler import CPUTarget
        if kernel not in artifacts:
            constants = {"C": 32} if kernel is broadcast_half_snapshot else {}
            artifacts[kernel] = compile(ASTSource(kernel, {}, constants),
                                       target=CPUTarget(backend="weft", **compile_target), options=options)
        compiled = artifacts[kernel]
        compiled[(1,)](*args)
        return compiled

    def number_extremum(a, b, maximum):
        result = np.fmax(a, b) if maximum else np.fmin(a, b)
        zero = (a == 0) & (b == 0)
        bits = np.bitwise_and(a.view(np.uint32), b.view(np.uint32)) if maximum else np.bitwise_or(a.view(np.uint32), b.view(np.uint32))
        return np.where(zero, bits.view(np.float32), result).astype(np.float32)

    def same(actual, wanted):
        exact(np.isnan(actual), np.isnan(wanted))
        numeric = ~np.isnan(wanted)
        unsigned = np.dtype(f"u{wanted.dtype.itemsize}")
        exact(actual.view(unsigned)[numeric], wanted.view(unsigned)[numeric])

    sources = [[], [-0., 0.], [0., -0.], [np.nan], [np.nan]*9, [np.nan]*17,
               [-3.]*17, [2.]*17, [np.inf, -np.inf, np.nan, -0., 0.],
               [-0., 0.]*8+[-0.], [np.nan, -0., 0.]*5+[1., -1.]]
    records = []
    for i, values in enumerate(sources):
        x = np.array(values, np.float32); other = x[::-1].copy()
        original = x.copy()
        storage, output = guarded(x.size*4, np.float32, -777)
        output = output.reshape(x.size, 4)
        stats_storage, stats = guarded(3, np.float32, -777)
        expected_pair = np.stack((number_extremum(x, other, True), number_extremum(x, other, False),
                                  number_extremum(x, np.full_like(x, 0.), True),
                                  number_extremum(x, np.full_like(x, -0.), False)), axis=1)
        expected_stats = np.full(3, -777, np.float32)
        if x.size:
            expected_stats[:] = x[0]
            expected_stats[2] = np.abs(x[0])
            for value in x[1:]:
                expected_stats[0] = number_extremum(expected_stats[:1], np.array([value], np.float32), True)[0]
                expected_stats[1] = number_extremum(expected_stats[1:], np.array([value], np.float32), False)[0]
                expected_stats[2] = number_extremum(expected_stats[2:], np.array([abs(value)], np.float32), True)[0]
        compiled = execute(fixed_number_extrema, x, other, output, stats)
        if i == 0:
            native_evidence(compiled, root, "fixed-number-extrema", ["reduce"])
        save(root, f"extrema-{i}", source=original, expected_pair=expected_pair, actual_pair=output,
             expected_stats=expected_stats, actual_stats=stats)
        same(output, expected_pair); same(stats, expected_stats); same(x, original)
        exact(storage[[0, 1, -2, -1]], np.full(4, -777, np.float32))
        exact(stats_storage[[0, 1, -2, -1]], np.full(4, -777, np.float32))
        records.append({"N":x.size,"signed_zero_bits_checked":True})
    for m, k in ((4, 32), (3, 17), (4, 0), (0, 32)):
        x = np.zeros((m, k), np.float32)
        if m and k:
            x[0] = np.nan
            x[1] = -0.
            x[2] = -np.arange(k, dtype=np.float32)
            x[2, 0] = -np.inf
            if m == 4:
                x[3] = -3.
                x[3, 0] = np.nan
        storage, y = guarded(m, np.float32, -777)
        compiled = execute(absolute_maxima, x, y)
        expected = np.full(m, -777, np.float32)
        if k:
            for row in range(m):
                expected[row] = np.abs(x[row, 0])
                for value in x[row, 1:]:
                    expected[row] = number_extremum(expected[row:row+1], np.array([abs(value)], np.float32), True)[0]
        same(y, expected)
        exact(storage[[0, 1, -2, -1]], np.full(4, -777, np.float32))
        save(root, f"absolute-maxima-{m}-{k}", source=x, expected=expected, actual=y)
    native_evidence(compiled, root, "absolute-maxima", [])
    array = np.array([[np.nan]*17, [-0., 0.]*8+[-0.], [0., -0.]*8+[0.]], np.float32)
    array_result = np.full((3,3), -777, np.float32)
    execute(array_special_reduce, array, array_result)
    same(array_result[:,1:], np.array([[np.nan,np.nan],[0.,-0.],[0.,-0.]],np.float32))
    raw = np.arange(65536, dtype=np.uint16)
    source = raw.view(np.float16)
    storage, output = guarded(65536, np.float32, -777)
    with np.errstate(invalid="ignore"):
        expected = source.astype(np.float32)
    compiled = execute(scalar_half_widen, source, output)
    same(output, expected); exact(raw, np.arange(65536,dtype=np.uint16))
    exact(storage[[0,1,-2,-1]],np.full(4,-777,np.float32))
    save(root,"scalar-half",source_bits=raw,expected=expected,actual=output)
    native_evidence(compiled, root, "scalar-half", [])
    other = source[::-1].copy()
    packed = np.stack((source, other), axis=1)
    with np.errstate(invalid="ignore"):
        expected_pair = packed.astype(np.float32)
        expected_wide = packed.astype(np.float64)
    for kernel, arguments in ((independent_half_pair, (source, other)), (snapshot_half_pair, (packed,))):
        storage, output = guarded(131072, np.float32, -777)
        wide_storage, wide = guarded(131072, np.float64, -777)
        output, wide = output.reshape(65536, 2), wide.reshape(65536, 2)
        compiled = execute(kernel, *arguments, output, wide)
        same(output, expected_pair); same(wide, expected_wide)
        exact(storage[[0, 1, -2, -1]], np.full(4, -777, np.float32))
        exact(wide_storage[[0, 1, -2, -1]], np.full(4, -777, np.float64))
        save(root, kernel.__name__, source_bits=raw, expected=expected_pair, actual=output,
             expected_wide=expected_wide, actual_wide=wide)
        native_evidence(compiled, root, kernel.__name__, [])
    for n, stride in ((0, 3), (1, 3), (31, 3), (33, 1), (65536, 3)):
        backing = np.full(n * stride + 2, 7, np.float16)
        x = backing[1:1 + n * stride:stride]
        x[:] = source[:n]
        before = backing.copy()
        expected_backing = backing.copy()
        expected_backing[1:1 + n * stride:stride] = 1
        output_storage, output = guarded(n * 32, np.float32, -777)
        wide_storage, wide = guarded(n * 32, np.float64, -777)
        output, wide = output.reshape(n, 32), wide.reshape(n, 32)
        with np.errstate(invalid="ignore"):
            wanted = np.broadcast_to(x.astype(np.float32)[:, None], (n, 32)).copy()
            wanted_wide = np.broadcast_to(x.astype(np.float64)[:, None], (n, 32)).copy()
        compiled = execute(broadcast_half_snapshot, x, output, wide)
        same(output, wanted); same(wide, wanted_wide)
        exact(backing.view(np.uint16), expected_backing.view(np.uint16))
        exact(output_storage[[0, 1, -2, -1]], np.full(4, -777, np.float32))
        exact(wide_storage[[0, 1, -2, -1]], np.full(4, -777, np.float64))
        save(root, f"broadcast-half-{n}-{stride}", source=before, actual_source=backing,
             expected=wanted, actual=output, expected_wide=wanted_wide, actual_wide=wide)
    native_evidence(compiled, root, "broadcast-half-snapshot", [])
    return {"extrema":records,"array_extrema_signed_zero_and_nan":True,"half_patterns":65536,
            "strided_half_broadcast_snapshot":True,
            "half_pairs_f32_and_f64":True,"half_nan_payload_unspecified":True,"output_guards_unchanged":True}


def array_contractions(root, options, close, compile_target):
    from numerical_primitive_kernels import array_contraction, array_contraction_integer, array_contraction_batched, indexed_subbyte, pointer_packet_wrap, named_standard_reduction
    from triton import compile
    from triton.compiler import ASTSource
    from triton.backends.compiler import CPUTarget
    artifacts, records = {}, []

    def execute(kernel, *args, signature=None, **constants):
        key = kernel.__name__, tuple(sorted(constants.items()))
        if key not in artifacts:
            artifacts[key] = compile(ASTSource(kernel, signature or {}, constants),
                                     target=CPUTarget(backend="weft", **compile_target), options=options)
        compiled = artifacts[key]
        compiled[(1,)](*args)
        return compiled

    for m, n, k in [(0, 3, 17), (3, 0, 17), (3, 5, 0), (3, 5, 1), (3, 5, 7),
                    (3, 5, 17), (2, 3, 257), (3, 5, 5003)]:
        x = ((np.arange(m*k).reshape(m, k) % 31) - 15).astype(np.float32) / 16
        w = ((np.arange(n*k).reshape(n, k) % 23) - 11).astype(np.float32) / 16
        y = np.full((m, n), -777, np.float32)
        execute(array_contraction, x, w, y)
        close(y, x @ w.T if k else np.full_like(y, -777))
        scale = np.arange(m, dtype=np.float32) + 0.5
        sums = np.full(m, -777, np.float32)
        execute(named_standard_reduction, x, scale, sums)
        close(sums, x.sum(axis=1) * scale if k else np.full_like(sums, -777))
        xi = (x * 32768).astype(np.int16)
        wi = (np.arange(n*k).reshape(n, k) % 256).astype(np.uint8)
        yi = np.full((m, n), -777, np.int32)
        execute(array_contraction_integer, xi, wi, yi)
        exact(yi, xi.astype(np.int32) @ wi.astype(np.int32).T if k else np.full_like(yi, -777))
        records.append({"shape": [m, n, k], "f32": True, "i16_u8_i32_wrap": True})
    for layout in ("gapped", "transposed", "broadcast"):
        x = (np.arange(3*74).reshape(3, 74) % 19 - 9).astype(np.float32)[:, ::2]
        w = (np.arange(5*37).reshape(37, 5) % 13 - 6).astype(np.float32).T
        if layout == "broadcast": w = np.broadcast_to(w[:1], (5, 37))
        if layout == "gapped": w = np.repeat(w, 2, axis=1)[:, ::2]
        storage = np.full((3, 10), -777, np.float32)
        execute(array_contraction, x, w, storage[:, ::2])
        close(storage[:, ::2], x @ w.T)
        exact(storage[:, 1::2], np.full((3, 5), -777, np.float32))
    for k, l in [(0, 5), (3, 7), (7, 17)]:
        x = (np.arange(2*3*k*l).reshape(2, 3, k, l) % 17 - 8).astype(np.float32) / 8
        w = (np.arange(2*5*k*l).reshape(2, 5, k, l) % 13 - 6).astype(np.float32) / 8
        y = np.full((2, 3, 5), -777, np.float32)
        execute(array_contraction_batched, x, w, y)
        close(y, np.einsum("bmkl,bnkl->bmn", x, w) if k*l else np.full_like(y, -777))
    for kind in range(4):
        x = (np.arange(256) - 128).astype(np.int8)
        original = x.copy()
        indices = ((np.arange(67) * 53) % 256).astype(np.uint32)
        y = np.full(67, -777, np.int32)
        execute(indexed_subbyte, x, indices, y, KIND=kind)
        logical = (original.astype(np.int32) & 3 if kind == 0 else original.astype(np.int32) & 15 if kind == 1
                   else (original.astype(np.int32) + 8) % 16 - 8 if kind == 2 else original < 0)
        exact(y, logical[indices].astype(np.int32))
        exact(x, original + np.int8(1))
    for base in (0, 63, 110, 126, -128, -45):
        x = np.arange(256, dtype=np.float32) / 7
        y = np.full(64, -777, np.float32)
        execute(pointer_packet_wrap, x, y, base, signature={"X": "*fp32", "Y": "*fp32", "Base": "i32"})
        index = (np.arange(64) + base).astype(np.int8).astype(np.int64) + 128
        expected = x[index]; expected[57:] = -13
        exact(y, expected)
    evidence = [native_evidence(compiled, root, f"contraction-{index}", [])
                for index, compiled in enumerate(artifacts.values())]
    return {"records": records, "layouts": ["gapped", "transposed", "broadcast"],
            "shared_batch_and_two_reduction_axes": True, "indexed_subbyte_snapshot": True,
            "execution_mode": "aot", "evidence": evidence}


def array_packets(root, options, close, compile_target):
    from numerical_primitive_kernels import array_map_reduce, array_tail_reduce, array_special_reduce, array_scalar_reduce, array_storage
    from numerical_primitive_kernels import array_forward_snapshot, array_forward_integer, array_forward_stage, array_resident_reduce
    from numerical_primitive_kernels import array_bfloat_storage, array_bfloat_rounding
    from numerical_primitive_kernels import array_project_snapshot, array_project_bits, array_project_bfloat
    from numerical_primitive_kernels import specialized_project_snapshot

    artifacts = {}
    def execute(kernel, *arguments, **constants):
        if compile_target is None:
            return kernel[(1,)](*arguments, **constants, **options)
        from triton import compile
        from triton.compiler import ASTSource
        from triton.backends.compiler import CPUTarget
        key = kernel.__name__, tuple(sorted(constants.items()))
        if key not in artifacts:
            artifacts[key] = compile(ASTSource(kernel, {}, constants),
                                     target=CPUTarget(backend="weft", **compile_target), options=options)
        compiled = artifacts[key]
        compiled[(1,)](*arguments)
        return compiled

    import torch
    bfloat_evidence = []
    # Independent boundary cases: signed zeros, tiny values, both tie parities,
    # largest finite values and infinities. Narrowing must not depend on a libcall.
    source_bits = np.array([0, 0x80000000, 1, 0x00008000, 0x00008001, 0x80008001,
                            0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000,
                            0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000], np.uint32)
    nearest_bits = np.array([0, 0x8000, 0, 0, 1, 0x8001, 0x3f80, 0x3f82,
                             0xbf80, 0xbf82, 0x7f80, 0xff80, 0x7f80, 0xff80], np.uint16)
    truncated_bits = np.array([0, 0x8000, 0, 0, 0, 0x8000, 0x3f80, 0x3f81,
                               0xbf80, 0xbf81, 0x7f7f, 0xff7f, 0x7f80, 0xff80], np.uint16)
    source = torch.from_numpy(source_bits.view(np.float32))
    nearest = torch.empty(len(source), dtype=torch.bfloat16)
    truncated = torch.empty_like(nearest)
    compiled = execute(array_bfloat_rounding, source, nearest, truncated)
    exact(nearest.view(torch.int16).numpy().view(np.uint16), nearest_bits)
    exact(truncated.view(torch.int16).numpy().view(np.uint16), truncated_bits)
    bfloat_evidence.append(native_evidence(compiled, root, "array-bfloat-rounding", ["narrow"]))
    for count in (0, 17, 4097):
        storage = torch.zeros(count * 2 + 4, dtype=torch.bfloat16)
        source = storage[2:-2:2]
        values = ((torch.arange(count, dtype=torch.float32) % 31) - 17) / 8
        source.copy_(values)
        original = source.clone()
        saved_storage = torch.full_like(storage, 11)
        saved = saved_storage[2:-2:2]
        wide_storage = torch.full((count * 2 + 4,), 13, dtype=torch.float32)
        wide = wide_storage[2:-2:2]
        compiled = execute(array_bfloat_storage, source, saved, wide)
        exact(saved.view(torch.int16).numpy(), original.view(torch.int16).numpy())
        exact(wide.numpy(), original.float().numpy())
        exact(source.view(torch.int16).numpy(), (original.float() + 1).bfloat16().view(torch.int16).numpy())
        for backing, sentinel in ((saved_storage, 11), (wide_storage, 13)):
            exact(backing[:2].float().numpy(), np.full(2, sentinel))
            exact(backing[-2:].float().numpy(), np.full(2, sentinel))
            exact(backing[3:-2:2].float().numpy(), np.full(count, sentinel))
        bfloat_evidence.append(native_evidence(compiled, root, f"array-bfloat-{count}", ["helper"]))
    projected_records = []
    for m, n in ((0, 17), (5, 0), (1, 1), (5, 31), (5, 4097)):
        original = (np.arange(m*n, dtype=np.float32).reshape(m, n) % 101 - 57) / 8
        backing = np.zeros((m, n*2+3), np.float32)
        source = backing[:, 1:n*2+1:2]
        source[:] = original
        destination = np.full_like(backing, -777)
        output = destination[:, 1:n*2+1:2]
        compiled = execute(array_project_snapshot, source, output)
        exact(output, original)
        exact(source, original+1)
        untouched = np.ones(destination.shape, np.bool_); untouched[:, 1:n*2+1:2] = False
        exact(destination[untouched], np.full(untouched.sum(), -777, np.float32))
        proof = native_evidence(compiled, root, f"array-project-{m}-{n}", ["extract"])
        projected_records.append({"shape": [m, n], "snapshot_across_write": True, "evidence": proof})
    for n in (1, 31, 257):
        source = np.arange(5*n, dtype=np.float32).reshape(5, n) / 8
        original, output = source.copy(), np.full_like(source, -777)
        compiled = execute(specialized_project_snapshot, source, output, N=n)
        exact(output, original); exact(source, original+1)
        projected_records.append({"shape": [5, n], "specialized_helper_dimension": True,
            "evidence": native_evidence(compiled, root, f"array-project-specialized-{n}", ["extract"])})
    for kind in range(3):
        for n in (0, 17, 4097):
            source = (np.arange(n) % 256 - 128).astype(np.int8)
            original = source.copy()
            storage, output = guarded(n, np.int8, 37)
            compiled = execute(array_project_bits, source, output, KIND=kind)
            logical = original & 15 if kind == 0 else ((original.astype(np.int32) + 8) % 16 - 8) if kind == 1 else original < 0
            exact(output, logical.astype(np.int8)); exact(source, original+np.int8(1))
            exact(storage[[0, 1, -2, -1]], np.full(4, 37, np.int8))
            projected_records.append({"shape": [n], "kind": kind, "evidence":
                native_evidence(compiled, root, f"array-project-bits-{kind}-{n}", ["extract"])})
    for m, n in ((5, 0), (5, 31), (5, 4097)):
        storage = torch.zeros((m, n*2+3), dtype=torch.bfloat16)
        source = storage[:, 1:n*2+1:2]
        source.copy_((torch.arange(m*n).reshape(m, n) % 31 - 17).float() / 8)
        original = source.clone()
        target_storage = torch.full_like(storage, 11)
        output = target_storage[:, 1:n*2+1:2]
        compiled = execute(array_project_bfloat, source, output)
        exact(output.view(torch.int16).numpy(), original.view(torch.int16).numpy())
        exact(source.float().numpy(), (original.float()+1).bfloat16().float().numpy())
        untouched = np.ones(tuple(target_storage.shape), np.bool_); untouched[:, 1:n*2+1:2] = False
        exact(target_storage.float().numpy()[untouched], np.full(untouched.sum(), 11, np.float32))
        projected_records.append({"shape": [m, n], "type": "bf16", "evidence":
            native_evidence(compiled, root, f"array-project-bfloat-{m}-{n}", ["extract"])})
    records, first = [], None
    for m, n in [(3, 1), (3, 0), (3, 7), (19, 17), (3, 31), (3, 257), (3, 4095), (3, 4097), (3, 5003)]:
        x = ((np.arange(m*n).reshape(m, n) % 37) - 23).astype(np.float32) / 8
        y = np.full_like(x, -777)
        flags = np.zeros((m, n), dtype=np.bool_)
        rows = np.full((m, 3), -777, np.float32)
        columns = np.full(n, -777, np.float32)
        wide = np.full(m, -777, np.float64)
        integer = np.full(m, -777, np.int32)
        if n == 257:
            x[:] = np.float32(2**27)
        expected = np.where(x < 0, x * 0.5, x - 2.0)
        compiled = execute(array_map_reduce, x, y, flags, rows, columns, wide, integer)
        if first is None:
            first = compiled
        else:
            assert compiled is first, "runtime shape changed the compiled array program"
        close(y, expected)
        exact(flags, x < 0)
        wanted = np.stack((expected.sum(axis=1), expected.max(axis=1), expected.min(axis=1)), axis=1) if n else np.full((m, 3), -777, np.float32)
        close(rows, wanted)
        close(columns, expected.sum(axis=0))
        close(wide, x.astype(np.float64).sum(axis=1) if n else np.full(m, -777, np.float64))
        exact(integer, (x * 8).astype(np.int32).sum(axis=1, dtype=np.int32) if n else np.full(m, -777, np.int32))
        records.append({"shape": [m, n], "row_reductions": ["sum", "max", "min"], "column_sum": True,
                        "boolean_storage": True, "reused_kernel": compiled is first})
    # Reuse the same compiled program across true runtime storage relations.
    # Output holes must stay untouched; changing stride cannot imply dense access.
    m, n = 3, 37
    dense = (np.arange(m*n).reshape(m, n) % 31 - 20).astype(np.float32) / 8
    gapped = np.zeros((m, 2*n+3), np.float32)
    gapped[:, 1:2*n+1:2] = dense
    layouts = {"unit": dense, "gapped": gapped[:, 1:2*n+1:2],
               "transpose": dense.T.copy().T, "broadcast": np.broadcast_to(dense[:1], (m, n))}
    for layout, x in layouts.items():
        storage = np.full((m, 2*n+3), -777, np.float32)
        y = storage[:, 1:2*n+1:2]
        flags = np.zeros((m, n), np.bool_)
        rows = np.empty((m, 3), np.float32)
        columns, wide, integer = np.empty(n, np.float32), np.empty(m, np.float64), np.empty(m, np.int32)
        compiled = execute(array_map_reduce, x, y, flags, rows, columns, wide, integer)
        assert compiled is first, "runtime stride changed the compiled array program"
        expected = np.where(x < 0, x * 0.5, x - 2.0)
        close(y, expected); exact(flags, x < 0)
        close(rows, np.stack((expected.sum(axis=1), expected.max(axis=1), expected.min(axis=1)), axis=1))
        close(columns, expected.sum(axis=0)); close(wide, x.astype(np.float64).sum(axis=1))
        exact(integer, (x * 8).astype(np.int32).sum(axis=1, dtype=np.int32))
        untouched = np.ones(storage.shape, np.bool_); untouched[:, 1:2*n+1:2] = False
        exact(storage[untouched], np.full(untouched.sum(), -777, np.float32))
        records.append({"shape": [m, n], "layout": layout, "reused_kernel": True, "output_holes_unchanged": True})
    m, n, block = 3, 5003, 5008
    x = -np.ones((m, n), np.float32) * (np.arange(n, dtype=np.float32) % 13 + 1)
    # Guard complete contiguous allocations without requiring strided Views.
    storage, y = guarded(m*n, np.float32, -777)
    y = y.reshape(m, n)
    rows = np.full((m, 3), -777, np.float32)
    compiled = execute(array_tail_reduce, x, y, rows, BLOCK=block)
    expected = x * 0.5 - 2.0
    close(y, expected)
    close(rows, np.stack((expected.sum(axis=1), expected.max(axis=1), expected.min(axis=1)), axis=1))
    exact(storage[[0, 1, -2, -1]], np.full(4, -777, np.float32))
    evidence = native_evidence(compiled, root, "array-tail", ["reduce"])
    layout_evidence = native_evidence(first, root, "array-layouts", ["reduce"])
    scalar_first = None
    for n in (0, 1, 31, 5003):
        x = (np.arange(2*n, dtype=np.float32) % 17 - 10)[::2]
        y = np.full(3, -777, np.float32)
        current = execute(array_scalar_reduce, x, y)
        if scalar_first is None:
            scalar_first = current
        assert current is scalar_first, "scalar reduction specialized runtime length"
        wanted = np.array([(x*x).sum(), x.max(), x.min()], np.float32) if n else np.full(3, -777, np.float32)
        close(y, wanted)
    storage_first = None
    for n, step in ((0, 1), (65, 1), (65, 2), (5003, 2)):
        arguments, backing = [], []
        for dtype in (np.int8, np.int16, np.float16, np.float64):
            x = (np.arange(n*step) % 31 - 20).astype(dtype)[::step]
            storage = np.full(n*step+2, -77, dtype)
            y = storage[1:1+n*step:step]
            arguments.extend((x, y)); backing.append(storage)
        current = execute(array_storage, *arguments)
        if storage_first is None:
            storage_first = current
        assert current is storage_first, "array storage specialized runtime size/stride"
        for i, storage in enumerate(backing):
            exact(arguments[2*i+1], arguments[2*i])
            untouched = np.ones(storage.shape, np.bool_); untouched[1:1+n*step:step] = False
            exact(storage[untouched], np.full(untouched.sum(), -77, storage.dtype))
    forward_first, integer_first = None, None
    for n in (0, 1, 31, 5003):
        original = (np.arange(n, dtype=np.float32) % 17 - 10) / 4
        x, stats = original.copy(), np.full(3, -777, np.float32)
        current = execute(array_forward_snapshot, x, stats)
        if forward_first is None:
            forward_first = current
        assert current is forward_first, "snapshot forwarding specialized runtime length"
        scaled = original * 2 + 1
        wanted = np.array([(scaled*scaled).sum(), scaled.sum(), original.max()], np.float32) if n else np.full(3, -777, np.float32)
        close(stats, wanted); close(x, original - 3)
        integers = np.resize(np.array([32768, -32769, 65535, 65536, -65537, 101], np.int32), n)
        output = np.full(1, -777, np.int32)
        current = execute(array_forward_integer, integers, output)
        if integer_first is None:
            integer_first = current
        assert current is integer_first, "integer forwarding specialized runtime length"
        expected = (integers.astype(np.int16).astype(np.int32) * np.int32(131071)).sum(dtype=np.int32) if n else np.int32(-777)
        exact(output, np.array([expected], np.int32))
    forwarding_evidence = native_evidence(forward_first, root, "array-forwarding", ["reduce"])
    stage_first = None
    for n in (0, 33, 5003):
        x = np.full(n, np.nextafter(np.float32(1), np.float32(2)), np.float32)
        storage, y = guarded(n, np.float32, -777)
        current = execute(array_forward_stage, x, y)
        if stage_first is None:
            stage_first = current
        assert current is stage_first, "stage boundary specialized runtime length"
        # The stage freezes the rounded F32 product. Cross-birth FMA would leave
        # a residual which the following exact power-of-two scaling exposes.
        expected = (x*x - np.float32(1.000000238418579)) * np.float32(1099511627776.0)
        close(y, expected)
        exact(storage[[0, 1, -2, -1]], np.full(4, -777, np.float32))
    stage_evidence = native_evidence(stage_first, root, "array-forward-stage", ["level"])
    resident_evidence = []
    for width in (16, 256):
        resident_first = None
        for n in (0, 1, 31, 65, 5003):
            x = (np.arange(n, dtype=np.float32) % 17 - 8) / 16
            original = np.arange(width, dtype=np.float32) / 32 - 2
            keep, stats = original.copy(), np.full(1, -777, np.float32)
            storage, output = guarded(width, np.float32, -777)
            current = execute(array_resident_reduce, x, keep, output, stats, RESIDENT=width)
            if resident_first is None:
                resident_first = current
            assert current is resident_first, "surrounding residency specialized runtime reduction length"
            total = (x*x).sum(dtype=np.float32)
            close(stats, np.array([total if n else -777], np.float32))
            close(keep, original - 1)
            close(output, original + total if n else np.full(width, -777, np.float32))
            exact(storage[[0, 1, -2, -1]], np.full(4, -777, np.float32))
        resident_evidence.append(native_evidence(resident_first, root, f"array-resident-{width}", ["reduce"]))
    special = np.full((4, 17), -3, np.float32)
    special[0] = np.nan
    special[1, 0] = np.nan
    special[2, 1] = np.inf
    special[3, :2] = [np.inf, -np.inf]
    rows = np.full((4, 3), -777, np.float32)
    execute(array_special_reduce, special, rows)
    with np.errstate(invalid="ignore"):
        wanted = np.stack((special.sum(axis=1), np.fmax.reduce(special, axis=1),
                           np.fmin.reduce(special, axis=1)), axis=1)
    exact(np.isnan(rows), np.isnan(wanted))
    exact(np.isposinf(rows), np.isposinf(wanted))
    exact(np.isneginf(rows), np.isneginf(wanted))
    finite = np.isfinite(wanted)
    close(rows[finite], wanted[finite])
    return {"shapes": records, "masked_array_shape": [2, block], "active_tail": [m % 2, n % block],
            "output_guards_unchanged": True, "source_parameters_changed": False,
            "f64_reduction": True, "i32_modular_reduction": True,
            "empty_domain_skips_output": True, "nan_and_infinity_policy": True,
            "runtime_strides_reuse_kernel": True, "scalar_reduction_empty_validity": True,
            "array_storage_types": ["i8", "i16", "f16", "f64"],
            "bfloat_snapshot_helper": bfloat_evidence,
            "projected_array_snapshots": projected_records,
            "producer_snapshot_across_write": True, "fanout_preserved": True,
            "producer_integer_cast_wrap": True, "forwarding_evidence": forwarding_evidence,
            "stage_rounding_boundary": True, "stage_evidence": stage_evidence,
            "surrounding_snapshot_widths": [16, 256], "resident_evidence": resident_evidence,
            "layout_evidence": layout_evidence,
            "execution_mode": "aot" if compile_target else "jit", "evidence": evidence}
