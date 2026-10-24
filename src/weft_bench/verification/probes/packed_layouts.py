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


def nibble_contraction_contract(root, options):
    import triton.language as tl
    import packed_layout_kernels as kernels
    from helper_native import capture

    def storage(rows, k, cohort, multiplier):
        position = np.arange(rows*k).reshape(rows, k)
        unsigned = ((position*multiplier + position//7 + 3) % 16).astype(np.int8)
        signed = (((position*(multiplier+2) + position//5 + 1) % 16)-8).astype(np.int8)
        raw = bytearray()
        for first in range(0, rows, cohort):
            for block in range(0, k, 32):
                for values in (signed, unsigned):
                    for row in range(first, first+cohort):
                        raw.extend(dense_bits(values[row, block:block+32], 4))
        backing = np.full(len(raw)+258, 0xa5, np.uint8)
        payload = backing[129:-129]
        payload[:] = np.frombuffer(raw, np.uint8)
        return backing, payload, signed, unsigned

    records = []
    for mr, m, n, k in ((1, 4, 32, 32), (4, 4, 32, 96),
                         (8, 16, 64, 96), (8, 12, 32, 64)):
        xb, x, xs, xu = storage(m, k, 4, 7)
        wb, w, ws, wu = storage(n, k, 32, 11)
        original_x, original_w = xb.copy(), wb.copy()
        for unsigned in (False, True):
            output = np.full(m*n+64, -12345, np.int32)
            y = output[32:-32].reshape(m, n)
            left, right = (xu, wu) if unsigned else (xs, ws)
            expected = left.astype(np.int64) @ right.astype(np.int64).T
            name = f"nibble-{mr}-{m}-{n}-{k}-{'unsigned' if unsigned else 'signed'}"
            compiled = kernels.nibble_contraction[(1,)](
                tl.View(x, encoding=kernels.NibbleRows[4], shape=(m, k)),
                tl.View(w, encoding=kernels.NibbleRows[32], shape=(n, k)),
                y, MR=mr, UNSIGNED=unsigned, **options)
            np.savez(root/(name+'.npz'), left=left, right=right, actual=y, expected=expected)
            np.testing.assert_array_equal(y, expected)
            np.testing.assert_array_equal(xb, original_x)
            np.testing.assert_array_equal(wb, original_w)
            np.testing.assert_array_equal(output[:32], np.full(32, -12345, np.int32))
            np.testing.assert_array_equal(output[-32:], np.full(32, -12345, np.int32))
            leaf = "ime.fragment-nibbles-unsigned" if unsigned else "ime.fragment-nibbles-signed"
            assert leaf in compiled.asm["riscvir"]
            records.append({"MR": mr, "M": m, "N": n, "K": k, "unsigned": unsigned,
                            "evidence": capture(compiled, root, name)})
    return {"integer_exact": True, "guards_and_inputs_preserved": True, "cases": records}


def run_scenario(name, root, options):
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
    for records, swap in ((0, False), (1, False), (3, False), (4, False), (4, True)):
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
        np.testing.assert_array_equal(result[0], original[0, order])
        np.testing.assert_array_equal(values, expected.reshape(records, 8)[order])
        np.testing.assert_array_equal(source, original)
        for storage, width, sentinel in ((source_storage, guard, 0xa5),
                                         (result_storage, guard, 0x3c), (values_storage, 32, -12345)):
            np.testing.assert_array_equal(storage[:width], np.full(width, sentinel, storage.dtype))
            np.testing.assert_array_equal(storage[-width:], np.full(width, sentinel, storage.dtype))
        reshape_evidence.append({"records": records, "swap": swap,
                                 "evidence": capture(current, root, f"reshape-{name}-{records}-{swap}")})
    return {"layout": name, "shape": [3, 16], "row_group": 4, "logical_record_elements": 8,
            "record_bytes": raw.shape[-1], "field_reads_and_writes": True,
            "whole_record_stage_snapshot": True, "partial_row_guard": True,
            "signed_i4_unsigned_u6": name != "orders", "oracle": "independent declared byte layout",
            "snapshot": str(path), "evidence": evidence, "record_reshapes": reshape_evidence}


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
            compiled = kernels.cohort_float_scale[(1,)](
                view, source, output, First=first, ROWS=rows, RECORDS=records, **options)
            evidence.append(capture(compiled, root, f"float-scales-{records}-{first}-{rows}"))
            expected = source * scales[first:first+rows, None] * np.repeat(np.arange(1, records+1), 32)[None, :]
            np.savez(root / "float-scales-current.npz", source=source, actual=output,
                     expected=expected, first=first, rows=rows, records=records)
            np.testing.assert_array_equal(output, expected.astype(np.float32))
            np.testing.assert_array_equal(storage, original)
            np.testing.assert_array_equal(guarded[[0, 1, -2, -1]], np.full(4, -12345, np.float32))
    return {"two_byte_alignment": True, "read_only_input_preserved": True,
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
