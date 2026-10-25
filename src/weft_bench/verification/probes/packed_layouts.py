"""Independent byte fixtures for declared layouts; no compiler metadata as oracle."""
import numpy as np


def dense_bits(values, width):
    word = sum((int(value) & ((1 << width)-1)) << (width*index) for index, value in enumerate(values))
    return word.to_bytes((len(values)*width+7)//8, "little")


def serialize(name, tag, q, s, m, flags):
    if name == "orders":
        return int(tag).to_bytes(2, "big") + bytes([sum(int(value) << (7-i) for i, value in enumerate(flags))])
    prefix = int(tag).to_bytes(2, "little")
    if name == "natural":
        return prefix + dense_bits(q, 4) + dense_bits(s, 6)
    if name != "layered_joined":
        raise ValueError("unknown packed layout " + name)
    # Four positions per layer, first logical layer in the high nibble.
    layered = bytes(((int(q[i]) & 15) << 4) | (int(q[i+4]) & 15) for i in range(4))
    # The same published two-field/six-bit byte relation used by GGML Q4_K.
    joined = bytes([int(s[i]) | ((int(s[i+4]) >> 4) << 6) for i in range(4)] +
                   [int(m[i]) | ((int(m[i+4]) >> 4) << 6) for i in range(4)] +
                   [(int(s[i+4]) & 15) | ((int(m[i+4]) & 15) << 4) for i in range(4)])
    return prefix + layered + joined


def fixture(name, rows=3, records=2):
    elements = 8
    position = np.arange(rows*records*elements).reshape(rows, records, elements)
    q = (position*3+2) % 16-8
    s, m = (position*7+11) % 64, (position*13+5) % 64
    flags = (position//3+position) % 2
    tag = 0x1234 + np.arange(rows)[:, None]*0x101 + np.arange(records)[None, :]*0x201
    record_bytes = {"natural": 12, "layered_joined": 18, "orders": 3}[name]
    raw = np.empty((rows, records, record_bytes), np.uint8)
    changed = np.empty_like(raw)
    output = np.empty_like(raw)
    observed = np.empty((rows, records, elements), np.int32)
    for row in range(rows):
        for record in range(records):
            values = [q[row, record], s[row, record], m[row, record], flags[row, record]]
            updated = [values[0] ^ 1, values[1] ^ 32, values[2] ^ 16, values[3] ^ 1]
            raw[row, record] = np.frombuffer(serialize(name, tag[row, record], *values), np.uint8)
            changed[row, record] = np.frombuffer(serialize(name, tag[row, record], *updated), np.uint8)
            output[row, record] = np.frombuffer(serialize(name, tag[row, record]+1, *updated), np.uint8)
            if name == "orders":
                observed[row, record] = tag[row, record]+values[3]
            else:
                observed[row, record] = tag[row, record]+values[0]+16*values[1]
                if name == "layered_joined":
                    observed[row, record] += 1024*values[2]
    return raw, changed, output, observed.reshape(rows, records*elements)


def nibble_contraction_contract(root, options, native_i4=False, narrow_handoff=False):
    import triton.language as tl
    import packed_layout_kernels as kernels
    from helper_native import capture

    def storage(rows, k, cohort, multiplier):
        position = np.arange(rows*k).reshape(rows, k)
        unsigned = ((position*multiplier + position//7 + 3) % 16).astype(np.int8)
        signed = (((position*(multiplier+2) + position//5 + 1) % 16)-8).astype(np.int8)
        choices = np.array([-2147483648, 2147483647, -1, 0, 1, 65537, -65539], np.int32)
        bias = np.resize(choices, rows*(k//32)).reshape(rows, k//32)
        raw = bytearray()
        for first in range(0, rows, cohort):
            for block in range(0, k, 32):
                for values in (signed, unsigned):
                    for row in range(first, first+cohort):
                        raw.extend(dense_bits(values[row, block:block+32], 4))
                for row in range(first, first+cohort):
                    raw.extend(int(bias[row, block//32]).to_bytes(4, "little", signed=True))
        backing = np.full(len(raw)+258, 0xa5, np.uint8)
        payload = backing[129:-129]
        payload[:] = np.frombuffer(raw, np.uint8)
        return backing, payload, signed, unsigned, bias

    records = []
    nr = 16 if narrow_handoff else 32
    shapes = ((1, 4, 32, 32), (4, 4, 32, 96), (8, 16, 64, 96), (8, 12, 32, 64))
    if narrow_handoff:
        shapes += ((16, 16, 32, 32), (16, 20, 64, 64))
    for mr, m, n, k in shapes:
        xb, x, xs, xu, bias_x = storage(m, k, 4, 7)
        wb, w, ws, wu, bias_w = storage(n, k, 32, 11)
        original_x, original_w = xb.copy(), wb.copy()
        choices = ((False, False, False), (True, False, False), (True, True, False))
        if native_i4:
            choices += ((False, False, True), (True, False, True))
        for unsigned, biased, mixed in choices:
            output = np.full(m*n+64, -12345, np.int32)
            y = output[32:-32].reshape(m, n)
            left = xu if unsigned else xs
            right = ws if unsigned and mixed else wu if unsigned or mixed else ws
            expected = left.astype(np.int64) @ right.astype(np.int64).T
            if biased:
                expected = (expected + bias_x.astype(np.int64) @ bias_w.astype(np.int64).T).astype(np.int32)
            name = f"nibble-{mr}-{m}-{n}-{k}-{'unsigned' if unsigned else 'signed'}" + ('-mixed' if mixed else '') + ('-mac' if biased else '')
            compiled = kernels.nibble_contraction[(1,)](
                tl.View(x, encoding=kernels.NibbleRows[4], shape=(m, k)),
                tl.View(w, encoding=kernels.NibbleRows[32], shape=(n, k)),
                y, MR=mr, NR=nr, UNSIGNED=unsigned, BIAS=biased, NATIVE_I4=native_i4, MIXED=mixed, **options)
            np.savez(root/(name+'.npz'), left=left, right=right, actual=y, expected=expected)
            np.testing.assert_array_equal(y, expected)
            np.testing.assert_array_equal(xb, original_x)
            np.testing.assert_array_equal(wb, original_w)
            np.testing.assert_array_equal(output[:32], np.full(32, -12345, np.int32))
            np.testing.assert_array_equal(output[-32:], np.full(32, -12345, np.int32))
            leaf = "spacemit-ime2-i4-" if native_i4 else "ime.fragment-nibbles-unsigned" if unsigned else "ime.fragment-nibbles-signed"
            assert leaf in compiled.asm["riscvir"]
            if biased:
                assert any(leaf in compiled.asm["riscvir"] for leaf in
                           ("rvv.vmacc.row.vx", "rvv.vmacc.vv"))
            records.append({"MR": mr, "M": m, "N": n, "K": k, "unsigned": unsigned, "modular_mac": biased, "native_i4": native_i4, "mixed": mixed,
                            "evidence": capture(compiled, root, name)})
        if native_i4:
            positions = np.arange(m*k).reshape(m, k)
            byte_values = ((positions*47 + positions//11 + 19) % 256 - 128).astype(np.int8)
            raw = bytearray()
            for first in range(0, m, 4):
                for block in range(0, k, 32):
                    raw.extend(np.ones(4, np.float32).tobytes())
                    raw.extend(byte_values[first:first+4, block:block+32].tobytes())
            xb = np.full(len(raw)+260, 0xa5, np.uint8)
            x = xb[130:-130]
            x[:] = np.frombuffer(raw, np.uint8)
            original_x = xb.copy()
            for shift, unsigned, coefficient in ((0, True, 0), (4, False, 0), (0, False, 0), (4, True, 0),
                                                 (4, False, 16), (4, False, -2147483648)):
                output = np.full(m*n+64, -12345, np.int32)
                y = output[32:-32].reshape(m, n)
                left = (byte_values.astype(np.int32) >> shift) & 15
                if not unsigned:
                    left = (left ^ 8) - 8
                expected = left @ wu.astype(np.int32).T
                if coefficient:
                    low = (byte_values.astype(np.int32) & 15) @ wu.astype(np.int32).T
                    expected = (expected.astype(np.int64) * coefficient + low).astype(np.int32)
                name = f"byte-nibble-{mr}-{m}-{n}-{k}-{shift}-{unsigned}-{coefficient}"
                compiled = kernels.byte_nibble_contraction[(1,)](
                    tl.View(x, encoding=kernels.FloatScaleRows[4], shape=(m, k)),
                    tl.View(w, encoding=kernels.NibbleRows[32], shape=(n, k)),
                    y, MR=mr, NR=nr, SHIFT=shift, UNSIGNED=unsigned, COEFFICIENT=coefficient, **options)
                np.savez(root/(name+'.npz'), left=left, right=wu, actual=y, expected=expected)
                np.testing.assert_array_equal(y, expected)
                np.testing.assert_array_equal(xb, original_x)
                np.testing.assert_array_equal(wb, original_w)
                np.testing.assert_array_equal(output[:32], np.full(32, -12345, np.int32))
                np.testing.assert_array_equal(output[-32:], np.full(32, -12345, np.int32))
                assert "ime.field-nibble-pack" in compiled.asm["riscvir"]
                if coefficient:
                    assert 'kind = "shl"' in compiled.asm["riscvir"]
                records.append({"MR": mr, "M": m, "N": n, "K": k,
                                "unsigned": unsigned, "shift": shift, "coefficient": coefficient,
                                "byte_to_nibble": True,
                                "evidence": capture(compiled, root, name)})
    return {"integer_exact": True, "guards_and_inputs_preserved": True, "cases": records}


def masked_record_contraction_contract(root, options):
    import triton.language as tl
    import packed_layout_kernels as kernels
    from helper_native import capture

    raw, _, _, _ = fixture("natural", rows=1, records=4)
    raw = raw.reshape(-1)
    positions = np.arange(32).reshape(4, 8)
    products = ((positions * 3 + 2) % 16 - 8) * ((positions * 7 + 11) % 64)
    records, first = [], None
    for left in (0, 8, 16, 24, 32):
        for right in (0, 8, 24, 32):
            lhs = np.full(raw.size + 64, 0xa5, np.uint8)
            rhs = np.full(raw.size + 64, 0x5a, np.uint8)
            lhs[32:-32], rhs[32:-32] = raw, raw
            guard = np.full(13, -12345, np.int32)
            output = guard[4:9]
            a = tl.View(lhs[32:32 + left // 8 * 12], encoding=kernels.NaturalLE, shape=(left,))
            c = tl.View(rhs[32:32 + right // 8 * 12], encoding=kernels.NaturalLE, shape=(right,))
            compiled = kernels.masked_record_contract[(1,)](a, c, output, **options)
            if first is None:
                first = compiled
            assert compiled is first, "encoded contraction specialized runtime valid extents"
            active = min(left, right) // 8
            wanted = np.full(5, -12345, np.int32)
            wanted[:active] = products[:active].sum(axis=1, dtype=np.int32)
            if active:
                wanted[4] = products[:active].sum(dtype=np.int32)
            np.testing.assert_array_equal(output, wanted)
            np.testing.assert_array_equal(guard[:4], np.full(4, -12345, np.int32))
            np.testing.assert_array_equal(guard[9:], np.full(4, -12345, np.int32))
            np.testing.assert_array_equal(lhs[32:-32], raw)
            np.testing.assert_array_equal(rhs[32:-32], raw)
            records.append({"left": left, "right": right, "active_rows": active})
    return {"cases": records, "integer_exact": True, "two_reduction_axes": True,
            "invalid_outputs_untouched": True, "evidence": capture(first, root, "masked-record-contract")}


def run_scenario(name, root, options):
    if name == "masked_contract":
        return masked_record_contraction_contract(root, options)
    if name == "unaligned_windows":
        return unaligned_windows_contract(root, options)
    if name == "numeric_bitplanes":
        return numeric_bitplanes_contract(root, options)
    if name == "snapshot_storage":
        return snapshot_storage_contract(root, options)
    if name == "translated_windows":
        return translated_windows_contract(root, options)
    if name == "pipeline_carriers":
        from numerical_primitives import pipeline_carrier_contract
        return pipeline_carrier_contract(root, options)
    if name == "mixed_integer_mac":
        from numerical_primitives import mixed_integer_mac_contract
        return mixed_integer_mac_contract(root, options)
    import triton.language as tl
    import packed_layout_kernels as kernels
    from helper_native import capture

    if name == "cohort":
        return cohort_contract(root, options)
    if name == "cohort_byte_writes":
        return cohort_byte_write_contract(root, options)
    if name == "cohort_float_scales":
        return cohort_float_scale_contract(root, options)
    if name == "nibble_contraction":
        return nibble_contraction_contract(root, options)
    if name == "native_nibble_contraction":
        return nibble_contraction_contract(root, options, native_i4=True)
    if name == "nibble_handoff":
        return nibble_contraction_contract(root, options, native_i4=True, narrow_handoff=True)

    choices = {"natural": (kernels.NaturalLE, kernels.natural_fields),
               "layered_joined": (kernels.LayeredJoined, kernels.layered_joined_fields),
               "orders": (kernels.BigOrder, kernels.ordered_fields)}
    encoding, kernel = choices[name]
    raw, expected_input, expected_output, expected_observed = fixture(name)
    guard = 128
    input_storage = np.full(raw.size+2*guard, 0xa5, np.uint8)
    output_storage = np.full(raw.size+2*guard, 0x3c, np.uint8)
    source = input_storage[guard:-guard].reshape(raw.shape)
    output = output_storage[guard:-guard].reshape(raw.shape)
    source[:] = raw
    observed_storage = np.full(expected_observed.size+64, -12345, np.int32)
    observed = observed_storage[32:-32].reshape(expected_observed.shape)
    path = root / ("packed-" + name + ".npz")
    np.savez(path, input=raw, expected_input=expected_input, expected_output=expected_output,
             expected_observed=expected_observed)
    compiled = kernel[(1,)](tl.View(source, encoding=encoding, shape=(3, 16)),
                           tl.View(output, encoding=encoding, shape=(3, 16)), observed, **options)
    np.savez(path, input=raw, expected_input=expected_input, actual_input=input_storage,
             expected_output=expected_output, actual_output=output_storage,
             expected_observed=expected_observed, actual_observed=observed_storage)
    evidence = capture(compiled, root, "packed-" + name)
    try:
        np.testing.assert_array_equal(source, expected_input)
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(observed, expected_observed)
        for storage, width, sentinel in ((input_storage, guard, 0xa5), (output_storage, guard, 0x3c),
                                         (observed_storage, 32, -12345)):
            np.testing.assert_array_equal(storage[:width], np.full(width, sentinel, storage.dtype))
            np.testing.assert_array_equal(storage[-width:], np.full(width, sentinel, storage.dtype))
    except AssertionError as error:
        raise ValueError("numerical mismatch: " + str(error)) from error
    reshape_kernel = {"natural": kernels.natural_reshape,
                      "layered_joined": kernels.layered_reshape,
                      "orders": kernels.ordered_reshape}[name]
    reshape_evidence = []
    for records, swap in ((0, False), (1, False), (3, False), (4, False),
                          (1, True), (2, True), (3, True), (4, True)):
        original, _, _, expected = fixture(name, rows=1, records=records)
        result_storage = np.full(original.size + 2*guard, 0x3c, np.uint8)
        result = result_storage[guard:-guard].reshape(original.shape)
        values_storage = np.full(records*8 + 64, -12345, np.int32)
        values = values_storage[32:-32].reshape(records, 8)
        source_storage = np.full(original.size + 2*guard, 0xa5, np.uint8)
        source = source_storage[guard:-guard].reshape(original.shape)
        source[:] = original
        current = reshape_kernel[(1,)](tl.View(source, encoding=encoding, shape=(records*8,)),
                                        tl.View(result, encoding=encoding, shape=(records, 8)),
                                        values, SWAP=swap, **options)
        order = [0, 2, 1, 3] if swap else list(range(records))
        expected_result = np.full_like(result, 0x3c)
        expected_values = np.full_like(values, -12345)
        for destination, source_record in enumerate(order[:records]):
            if source_record < records:
                expected_result[0, destination] = original[0, source_record]
                expected_values[destination] = expected.reshape(records, 8)[source_record]
        np.testing.assert_array_equal(result, expected_result)
        np.testing.assert_array_equal(values, expected_values)
        np.testing.assert_array_equal(source, original)
        for storage, width, sentinel in ((source_storage, guard, 0xa5),
                                         (result_storage, guard, 0x3c), (values_storage, 32, -12345)):
            np.testing.assert_array_equal(storage[:width], np.full(width, sentinel, storage.dtype))
            np.testing.assert_array_equal(storage[-width:], np.full(width, sentinel, storage.dtype))
        reshape_evidence.append({"records": records, "swap": swap,
                                 "evidence": capture(current, root, f"reshape-{name}-{records}-{swap}")})
    alias_kernel = {"natural": kernels.natural_alias_reshape,
                    "layered_joined": kernels.layered_alias_reshape,
                    "orders": kernels.ordered_alias_reshape}[name]
    original, _, _, expected = fixture(name, rows=1, records=4)
    aliased = np.full(original.size + 2*guard, 0xa5, np.uint8)
    payload = aliased[guard:-guard].reshape(original.shape)
    payload[:] = original
    numeric = np.full(32 + 64, -12345, np.int32)
    values = numeric[32:-32].reshape(4, 8)
    current = alias_kernel[(1,)](tl.View(payload, encoding=encoding, shape=(32,)),
                                 tl.View(payload, encoding=encoding, shape=(4, 8)), values, **options)
    np.testing.assert_array_equal(payload[0], original[0, [0, 2, 1, 3]])
    np.testing.assert_array_equal(values, expected.reshape(4, 8)[[0, 2, 1, 3]])
    for storage, width, sentinel in ((aliased, guard, 0xa5), (numeric, 32, -12345)):
        np.testing.assert_array_equal(storage[:width], np.full(width, sentinel, storage.dtype))
        np.testing.assert_array_equal(storage[-width:], np.full(width, sentinel, storage.dtype))
    reshape_evidence.append({"records": 4, "swap": True, "input_output_alias": True,
                             "evidence": capture(current, root, f"reshape-{name}-alias")})
    nested_kernel = {"natural": kernels.natural_nested_reshape,
                     "layered_joined": kernels.layered_nested_reshape,
                     "orders": kernels.ordered_nested_reshape}[name]
    original, _, _, expected = fixture(name, rows=10, records=2)
    source_storage = np.full(original.size + 2*guard, 0xa5, np.uint8)
    destination_storage = np.full(original.size + 2*guard, 0x3c, np.uint8)
    source = source_storage[guard:-guard].reshape(original.shape)
    destination = destination_storage[guard:-guard].reshape(original.shape)
    source[:] = original
    numeric = np.full(160 + 64, -12345, np.int32)
    values = numeric[32:-32].reshape(10, 16)
    current = nested_kernel[(1,)](tl.View(source, encoding=encoding, shape=(10, 16)),
                                  tl.View(destination, encoding=encoding, shape=(10, 16)), values, **options)
    expected_result = np.full_like(destination, 0x3c)
    expected_values = np.full_like(values, -12345)
    for destination_row in range(10):
        source_row = destination_row // 4 * 4 + [0, 2, 1, 3][destination_row % 4]
        if source_row < 10:
            expected_result[destination_row] = original[source_row]
            expected_values[destination_row] = expected[source_row]
    np.testing.assert_array_equal(destination, expected_result)
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(source, original)
    for storage, width, sentinel in ((source_storage, guard, 0xa5), (destination_storage, guard, 0x3c),
                                     (numeric, 32, -12345)):
        np.testing.assert_array_equal(storage[:width], np.full(width, sentinel, storage.dtype))
        np.testing.assert_array_equal(storage[-width:], np.full(width, sentinel, storage.dtype))
    reshape_evidence.append({"shape": [10, 16], "nested_row_groups": [8, 4], "swap": True,
                             "evidence": capture(current, root, f"reshape-{name}-nested")})
    return {"layout": name, "shape": [3, 16], "row_group": 4, "logical_record_elements": 8,
            "record_bytes": raw.shape[-1], "field_reads_and_writes": True,
            "whole_record_stage_snapshot": True, "partial_row_guard": True,
            "signed_i4_unsigned_u6": name != "orders", "oracle": "independent declared byte layout",
            "snapshot": str(path), "evidence": evidence, "record_reshapes": reshape_evidence}


def unaligned_windows_contract(root, options):
    import triton.language as tl
    import packed_layout_kernels as kernels
    from helper_native import capture

    records = []
    for update, whole in ((False, False), (True, False), (False, True), (True, True)):
        for count in (1, 3):
            words = (np.arange(count * 8, dtype=np.uint64) * 0x9e3779b9 + 0xfffffff0).astype(np.uint32).reshape(count, 8)
            source = np.full(count * 34 + 132, 0xa5, np.uint8)
            raw = source[66:-66].reshape(count, 34)
            raw[:, :2] = np.array([0x53, 0x29], np.uint8)
            raw[:, 2:] = words.astype('<u4').view(np.uint8).reshape(count, 32)
            before = source.copy()
            output = np.full(count + 16, 0xdeadbeef, np.uint32)
            compiled = kernels.unaligned_word_windows[(1,)](
                tl.View(raw.reshape(-1), encoding=kernels.UnalignedWords, shape=(count * 8,)),
                output[:count], UPDATE=update, WHOLE=whole, **options)
            expected = words.sum(axis=1, dtype=np.uint64).astype(np.uint32)
            np.testing.assert_array_equal(output[:count], expected)
            np.testing.assert_array_equal(output[count:], np.full(16, 0xdeadbeef, np.uint32))
            expected_source = before.copy()
            if update:
                expected_source[66:-66].reshape(count, 34)[:, 2:] = (words ^ np.uint32(0x13579bdf)).astype('<u4').view(np.uint8).reshape(count, 32)
            np.testing.assert_array_equal(source, expected_source)
            name = f"unaligned-words-{update}-{whole}-{count}"
            np.savez(root / (name + '.npz'), actual=output, expected=expected, source=source)
            records.append({"update": update, "whole": whole, "records": count,
                            "evidence": capture(compiled, root, name)})
    return {"integer_exact": True, "read_snapshot_and_guards_preserved": True, "cases": records}


def numeric_bitplanes_contract(root, options):
    import triton.language as tl
    import packed_layout_kernels as kernels
    from helper_native import capture

    records = []
    for select, keep in ((False, False), (True, False), (False, True)):
        for n in (64, 192):
            bits = ((np.arange(n) * 7 + np.arange(n) // 5) % 2).astype(np.uint8)
            packed = np.packbits(bits, bitorder="little")
            values = np.resize(np.array([-128, -127, -1, 0, 1, 63, 127], np.int8), n)
            output = np.full(n + 16, 77, np.int8)
            observed = np.full(n, -99, np.int8)
            wide = np.full(n + 16, 0xdeadbeef, np.uint32)
            compiled = kernels.encoded_sign_product[(1,)](
                tl.View(packed, encoding=kernels.BitSigns, shape=(n,)),
                values, output[:n], observed, wide[:n], SELECT=select, KEEP=keep, **options)
            expected = (values.astype(np.int16) * (1 - 2 * bits.astype(np.int16))).astype(np.int8)
            np.testing.assert_array_equal(output[:n], expected)
            np.testing.assert_array_equal(output[n:], np.full(16, 77, np.int8))
            np.testing.assert_array_equal(observed, bits.astype(np.int8) if keep else np.full(n, -99, np.int8))
            np.testing.assert_array_equal(wide[:n], bits.astype(np.uint32) if keep else np.full(n, 0xdeadbeef, np.uint32))
            np.testing.assert_array_equal(wide[n:], np.full(16, 0xdeadbeef, np.uint32))
            np.testing.assert_array_equal(packed, np.packbits(bits, bitorder="little"))
            if "riscvir" in compiled.asm and not select and not keep:
                assert "rvv.masked-negate" in compiled.asm["riscvir"], "numeric predicate materialization lost masked negation"
            name = f"numeric-bitplane-{select}-{keep}-{n}"
            np.savez(root / (name + '.npz'), output=output, expected=expected, bits=bits, wide=wide)
            records.append({"size": n, "select": select, "extra_consumer": keep,
                            "evidence": capture(compiled, root, name)})
    for base in (-128, -1, 0, 1, 127):
        n = 192
        bits = ((np.arange(n) * 7 + np.arange(n) // 5) % 2).astype(np.uint8)
        packed = np.packbits(bits, bitorder="little")
        values = np.resize(np.array([-128, -127, -1, 0, 1, 63, 127], np.int8), n)
        output = np.full(9, -12345, np.int32)
        compiled = kernels.encoded_sign_reduction[(1,)](
            tl.View(packed, encoding=kernels.BitSigns, shape=(n,)),
            values, output[:1], BASE=base, **options)
        signed = (base * (1 - 2 * bits.astype(np.int16))).astype(np.int8)
        expected = (signed.astype(np.int64) * values.astype(np.int64)).sum()
        np.testing.assert_array_equal(output[:1], np.array([expected], np.int32))
        np.testing.assert_array_equal(output[1:], np.full(8, -12345, np.int32))
        name = f"numeric-bitplane-reduction-{base}"
        np.savez(root / (name + '.npz'), output=output, expected=expected, signed=signed)
        records.append({"base": base, "reduction": True, "evidence": capture(compiled, root, name)})
    return {"integer_exact": True, "signed_min_wrap_preserved": True, "cases": records}


def snapshot_storage_contract(root, options):
    import triton.language as tl
    import packed_layout_kernels as kernels
    from helper_native import capture

    records, first = [], None
    for m, k, overlap in ((1, 32, False), (5, 96, False), (5, 96, True),
                          (4, 1024, True), (4, 32768, False)):
        signed = ((np.arange(m * k) * 7 + 3) % 16 - 8).reshape(m, k)
        raw = bytearray()
        for row in range(m):
            for block in range(k // 32):
                raw.extend(dense_bits(signed[row, block * 32:(block + 1) * 32], 4))
                raw.extend(bytes([0xa5] * 16))
                raw.extend((12345 + row + block).to_bytes(4, "little", signed=True))
        source_storage = np.full(len(raw) + 128, 0xc3, np.uint8)
        x = source_storage[64:-64]
        x[:] = np.frombuffer(raw, np.uint8)
        destination_storage = source_storage if overlap else np.full_like(source_storage, 0x5a)
        y = destination_storage[64:-64]
        before = y.copy().reshape(m, k // 32, 36)
        expected = before.copy()
        for row in range(m):
            for block in range(k // 32):
                expected[row, block, :16] = np.frombuffer(dense_bits(
                    signed[row, block * 32:(block + 1) * 32] ^ 1, 4), np.uint8)
        output_storage = np.full(m * k + 64, -12345, np.int32)
        output = output_storage[32:-32].reshape(m, k)
        compiled = kernels.encoded_snapshot_rows[(1,)](
            tl.View(x, encoding=kernels.NibblePair, shape=(m, k)),
            tl.View(y, encoding=kernels.NibblePair, shape=(m, k)), output, **options)
        if first is None:
            first = compiled
        else:
            assert compiled is first, "record snapshot JIT specialized dynamic shape or alias addresses"
        np.testing.assert_array_equal(output, signed.astype(np.int32) + 1000)
        np.testing.assert_array_equal(y.reshape(expected.shape), expected)
        np.testing.assert_array_equal(source_storage[:64], np.full(64, 0xc3, np.uint8))
        np.testing.assert_array_equal(source_storage[-64:], np.full(64, 0xc3, np.uint8))
        if not overlap:
            np.testing.assert_array_equal(x, np.frombuffer(raw, np.uint8))
            np.testing.assert_array_equal(destination_storage[:64], np.full(64, 0x5a, np.uint8))
            np.testing.assert_array_equal(destination_storage[-64:], np.full(64, 0x5a, np.uint8))
        np.testing.assert_array_equal(output_storage[:32], np.full(32, -12345, np.int32))
        np.testing.assert_array_equal(output_storage[-32:], np.full(32, -12345, np.int32))
        name = f"record-snapshot-{m}-{k}-{'overlap' if overlap else 'disjoint'}"
        proof = capture(compiled, root, name)
        if "riscvir" in compiled.asm:
            assert 'snapshot_may_borrow = true' in compiled.asm["riscvir"]
            assert 'weft_riscv.storage_disjoint' in compiled.asm["riscvir"]
        np.savez(root / (name + '.npz'), actual=output, expected=signed + 1000, bytes=y)
        records.append({"M": m, "K": k, "overlap": overlap, "evidence": proof})
    return {"integer_exact": True, "same_jit_for_shapes_and_aliases": True,
            "guards_preserved": True, "cases": records}


def cohort_byte_write_contract(root, options):
    import triton.language as tl
    import packed_layout_kernels as kernels
    from helper_native import capture

    records = []
    compiled = {}
    cases = [(m, k, first, 8 if first is None else 4) for m, k, first in
             ((0, 8, None), (4, 8, None), (8, 16, None), (12, 16, None),
              (12, 0, None), (12, 8, 0), (12, 8, 1), (12, 8, 4), (12, 8, 9))]
    cases += [(12, 16, None, 2), (0, 8, None, 2)]
    cases += [(12, 8, first, 2) for first in (0, 1, 2, 3, 4, 11)]
    for m, k, first, width in cases:
        source = (np.arange(m*k, dtype=np.uint32)*37 % 256).astype(np.uint8).reshape(m,k)
        size = m*(k//8)*18
        storage = np.full(size+256, 0xa5, np.uint8)
        output = storage[128:-128]
        expected = output.copy()
        rows = range(m) if first is None else range(first, min(first+width,m))
        for row in rows:
            for record in range(k//8):
                for element in range(2,7):
                    address = ((row//4)*(k//8)+record)*72 + 8 + (element//4)*16 + (row%4)*4 + element%4
                    expected[address] = source[row,record*8+element]
        view = tl.View(output, encoding=kernels.ByteRows[4], shape=(m,k))
        if first is None:
            current = kernels.cohort_byte_write[(1,)](source, view, ROWS=width, **options)
            assert compiled.setdefault(width, current) is current, "cohort store specialized runtime dimensions"
        else:
            current = kernels.cohort_byte_slice[(1,)](source, view, first, ROWS=width, **options)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(storage[:128], np.full(128,0xa5,np.uint8))
        np.testing.assert_array_equal(storage[-128:], np.full(128,0xa5,np.uint8))
        records.append({"M":m,"K":k,"first":first,"rows":width})
        if first is None:
            storage.fill(0xa5)
            indexed = kernels.cohort_byte_index_write[(1,)](source, view, ROWS=width, **options)
            np.testing.assert_array_equal(output, expected)
            np.testing.assert_array_equal(storage[:128], np.full(128,0xa5,np.uint8))
            np.testing.assert_array_equal(storage[-128:], np.full(128,0xa5,np.uint8))
            capture(indexed, root, f"cohort-byte-index-{m}-{k}-{width}")
            records.append({"M":m,"K":k,"first":first,"rows":width,"indexed":True})
    return {"cases":records,"partial_field_and_neighbor_bytes_exact":True,
            "cross_cohort_and_partial_rows":True,
            "evidence":[capture(compiled[8],root,"cohort-byte-write"),
                        capture(compiled[2],root,"cohort-half-byte-write"),capture(current,root,"cohort-byte-slice")]}


def translated_windows_contract(root, options):
    import triton.language as tl
    import packed_layout_kernels as kernels
    from helper_native import capture

    cases = []
    for records in (1, 2):
        rows, k = 8, records * 32
        coordinates = np.arange(rows * k).reshape(rows, records, 32)
        low, high = (coordinates * 7 + 3) % 16, (coordinates // 3 + 1) % 4
        scales = (np.arange(rows * records * 8).reshape(rows, records, 8) * 13 % 255 - 127).astype(np.int8)
        raw = np.zeros((rows, records, 32), np.uint8)
        for row in range(rows):
            for record in range(records):
                for index in range(32):
                    raw[row, record, index // 8 * 4 + index % 4] |= int(low[row, record, index]) << (index % 8 // 4 * 4)
                    raw[row, record, 16 + index // 16 * 4 + index % 4] |= int(high[row, record, index]) << (index % 16 // 4 * 2)
                raw[row, record, 24:] = scales[row, record].view(np.uint8)
        storage = np.full(raw.size + 2, 0xa5, np.uint8)
        packed = storage[1:-1].reshape(2, records, 128)
        for cohort in range(2):
            for record in range(records):
                for row in range(4):
                    for begin, size, chunk in ((0, 16, 4), (16, 8, 4), (24, 8, 1)):
                        for byte in range(size):
                            address = begin * 4 + byte // chunk * chunk * 4 + row * chunk + byte % chunk
                            packed[cohort, record, address] = raw[cohort * 4 + row, record, begin + byte]
        packed_bytes = packed.copy()
        for interleaved in (False, True):
            encoding = kernels.WindowRows[4] if interleaved else kernels.WindowFields
            kernel = kernels.translated_field_windows if interleaved else kernels.translated_plain_windows
            view = tl.View(storage[1:-1], encoding=encoding, shape=(rows, k))
            for width, row_tile in ((16, 4), (16, 2), (16, 3), (8, 4)):
                for update in (False, True):
                    storage[1:-1] = (packed_bytes if interleaved else raw).reshape(-1)
                    expected_storage = storage.copy()
                    if update:
                        for row in range(rows):
                            for record in range(records):
                                for index in range(0, 8, width // 4):
                                    address = ((row // 4 * records + record) * 128 + 24 * 4 + index * 4 + row % 4
                                               if interleaved else (row * records + record) * 32 + 24 + index)
                                    expected_storage[1 + address] ^= 1
                    guard = np.full(rows * k + 8, -12345, np.int32)
                    output = guard[4:-4].reshape(rows, k)
                    selected_scales = np.repeat(scales[:, :, np.arange(0, 32, width) // 4].astype(np.int32), width, axis=2)
                    expected = (low + high * 16 + selected_scales * 64).reshape(rows, k).astype(np.int32)
                    compiled = kernel[(1,)](view, output, ROWS=row_tile, WIDTH=width, UPDATE=update, **options)
                    name = f"translated-{records}-{width}-{row_tile}-{int(interleaved)}-{int(update)}"
                    np.savez(root / (name + ".npz"), input=raw, actual_storage=storage,
                             expected_storage=expected_storage, actual=output, expected=expected)
                    np.testing.assert_array_equal(output, expected)
                    np.testing.assert_array_equal(storage, expected_storage)
                    np.testing.assert_array_equal(guard[:4], np.full(4, -12345, np.int32))
                    np.testing.assert_array_equal(guard[-4:], np.full(4, -12345, np.int32))
                    cases.append({"records": records, "width": width, "row_tile": row_tile,
                                  "interleaved": interleaved, "update": update,
                                  "evidence": capture(compiled, root, name)})
    return {"integer_exact": True, "input_and_output_guards_preserved": True,
            "dynamic_windows_and_indices": True, "cross_cohort_and_tail_rows": True, "cases": cases}


def cohort_float_scale_contract(root, options):
    import triton.language as tl
    import packed_layout_kernels as kernels
    from helper_native import capture

    scales = np.array([0.25, -2.0, 1.5, -0.125, 8.0, 0.0, -4.0, 0.5], np.float32)
    evidence = []
    for records in (1, 2):
        k = records * 32
        # The two-byte offset is deliberately insufficient for an F32 load.
        storage = np.full(8 * records * 36 + 4, 0xa5, np.uint8)
        packed = storage[2:-2].reshape(2, records, 144)
        for cohort in range(2):
            for block in range(records):
                for row in range(4):
                    value = np.array([scales[cohort*4+row] * (block+1)], np.float32)
                    packed[cohort, block, 4*row:4*row+4] = value.view(np.uint8)
        original = storage.copy()
        view = tl.View(packed, encoding=kernels.FloatScaleRows[4], shape=(8, k))
        for first, rows in ((0, 8), (0, 4), (1, 4), (4, 4), (3, 2), (6, 2), (7, 1)):
            source = ((np.arange(rows*k).reshape(rows, k) % 29 - 14) / 8).astype(np.float32)
            guarded = np.full(rows*k+8, -12345, np.float32)
            output = guarded[4:-4].reshape(rows, k)
            expected = source * scales[first:first+rows, None] * np.repeat(np.arange(1, records+1), 32)[None, :]
            for saved, update in ((False, False), (True, False), (True, True)):
                storage[:] = original
                guarded.fill(-12345)
                compiled = kernels.cohort_float_scale[(1,)](
                    view, source, output, First=first, ROWS=rows, RECORDS=records,
                    SAVED=saved, UPDATE=update, **options)
                name = f"float-scales-{records}-{first}-{rows}-{saved}-{update}"
                evidence.append(capture(compiled, root, name))
                np.savez(root / (name + ".npz"), source=source, actual=output,
                         expected=expected, first=first, rows=rows, records=records)
                np.testing.assert_array_equal(output, expected.astype(np.float32))
                expected_storage = original.copy()
                if update:
                    expected_packed = expected_storage[2:-2].reshape(2, records, 144)
                    for row in range(first, first + rows):
                        for block in range(records):
                            value = np.array([scales[row] * (block + 1) + 1], np.float32)
                            start = (row % 4) * 4
                            expected_packed[row // 4, block, start:start + 4] = value.view(np.uint8)
                np.testing.assert_array_equal(storage, expected_storage)
                np.testing.assert_array_equal(guarded[[0, 1, -2, -1]], np.full(4, -12345, np.float32))
    return {"two_byte_alignment": True, "saved_read_and_update_preserved": True,
            "partial_and_cross_cohort_rows": True, "evidence": evidence}


def cohort_contract(root, options):
    import ctypes
    import triton.language as tl
    import packed_layout_kernels as kernels
    from helper_native import capture

    raw, _, _, expected = fixture("natural", rows=8)
    packed = np.empty((2, 2, 48), np.uint8)
    for group in range(2):
        for block in range(2):
            for row in range(4):
                for start, length, chunk in ((0, 2, 2), (2, 4, 1), (6, 6, 2)):
                    for index in range(length):
                        offset = start*4 + index//chunk*chunk*4 + row*chunk + index%chunk
                        packed[group, block, offset] = raw[group*4+row, block, start+index]
    storage = np.full(expected.size+8, -12345, np.int32)
    output = storage[4:-4].reshape(expected.shape)
    view = tl.View(packed, encoding=kernels.NaturalRows[4], shape=(8, 16))
    compiled = kernels.cohort_fields[(1,)](view, output, **options)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(storage[[0, 1, -2, -1]], np.full(4, -12345, np.int32))
    evidence = capture(compiled, root, "packed-cohort")
    output.fill(-12345)
    half = kernels.cohort_fields[(1,)](view, output, ROWS=2, **options)
    np.testing.assert_array_equal(output, expected)
    capture(half, root, "packed-half-cohort")
    indexed_evidence = []
    for rows in (8,):
        for width in (4, 2, 1):
            indexed_storage = np.full(rows*16 + 8, -12345, np.int32)
            indexed = indexed_storage[4:-4].reshape(rows, 16)
            current = kernels.cohort_indexed_fields[(1,)](
                tl.View(packed, encoding=kernels.NaturalRows[4], shape=(rows, 16)),
                indexed, ROWS=width, **options)
            np.testing.assert_array_equal(indexed, expected[:rows])
            np.testing.assert_array_equal(indexed_storage[[0, 1, -2, -1]], np.full(4, -12345, np.int32))
            indexed_evidence.append(capture(current, root, f"packed-indexed-{rows}-{width}"))
    artifact = root / "cohort-artifact"
    compiled.export(artifact)
    library = ctypes.CDLL(str(artifact / "kernel.so"))
    size = library.cohort_fields_X_packed_size
    size.argtypes, size.restype = [ctypes.c_size_t, ctypes.c_size_t], ctypes.c_size_t
    pack = library.cohort_fields_X_pack
    pack.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    pack.restype = None
    assert size(8, 16) == packed.nbytes
    actual = np.full(packed.shape, 0xa5, np.uint8)
    pack(raw.ctypes.data, actual.ctypes.data, 8, 16)
    np.testing.assert_array_equal(actual, packed)
    # One record per row makes a four-row slice cross a cohort at First=1.
    raw_one, _, _, expected_one = fixture("natural", rows=8, records=1)
    packed_one = np.empty(raw_one.size, np.uint8)
    pack(raw_one.ctypes.data, packed_one.ctypes.data, 8, 8)
    for first, width in ((0, 2), (1, 2), (3, 2), (6, 2), (7, 1)):
        guard = np.full(width*8+8, -12345, np.int32)
        result = guard[4:-4].reshape(width, 8)
        current = kernels.cohort_index_slice[(1,)](
            tl.View(packed_one, encoding=kernels.NaturalRows[4], shape=(8, 8)),
            result, First=first, ROWS=width, **options)
        np.testing.assert_array_equal(result, expected_one[first:first+width])
        np.testing.assert_array_equal(guard[[0, 1, -2, -1]], np.full(4, -12345, np.int32))
        indexed_evidence.append(capture(current, root, f"packed-index-slice-{first}-{width}"))
    for first in (0, 1, 4):
        result = np.full((4, 8), -12345, np.int32)
        kernels.cohort_slice[(1,)](tl.View(packed_one, encoding=kernels.NaturalRows[4], shape=(8, 8)), result, first, **options)
        np.testing.assert_array_equal(result, expected_one[first:first+4])
    # subview requires offset + extent <= base extent; it does not clamp.
    for first, rows in [(first, 2) for first in (0, 1, 2, 3, 4, 6)] + [(7, 1)]:
        result = np.full((rows, 8), -12345, np.int32)
        kernels.cohort_slice[(1,)](tl.View(packed_one, encoding=kernels.NaturalRows[4], shape=(8, 8)),
                                   result, first, ROWS=rows, **options)
        np.testing.assert_array_equal(result, expected_one[first:first+rows])
    np.savez(root / "packed-cohort.npz", source=raw, expected_packed=packed, actual_packed=actual,
             expected=expected, actual=output)
    return {"independent_byte_layout": True, "exported_builder_exact": True,
            "aligned_and_cross_cohort_slices": [0, 1, 4], "evidence": evidence,
            "indexed_fields": indexed_evidence}
