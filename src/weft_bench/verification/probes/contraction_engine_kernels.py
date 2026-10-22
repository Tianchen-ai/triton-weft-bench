"""Author engine constraints exercised by Bench on actual native code."""
from __future__ import annotations
import triton
import triton.language as tl


@triton.jit(model="weft")
def positional(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               ENGINE: tl.constexpr = "auto"):
    m = tl.arange(0, 16)
    n = tl.arange(0, 16)
    k = tl.arange(0, 64)
    a = tl.load(A + m[:, None] * K + k[None, :],
                (m[:, None] < M) & (k[None, :] < K), other=0)
    b = tl.load(B + k[:, None] * N + n[None, :],
                (k[:, None] < K) & (n[None, :] < N), other=0)
    c = tl.dot(a, b, engine=ENGINE)
    tl.store(C + m[:, None] * N + n[None, :], c,
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit(model="weft")
def named(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i8, (N, K)], C: tl.View[tl.i32, (M, N)],
          M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, ENGINE: tl.constexpr = "auto"):
    a = tl.load(A)
    b = tl.load(B)
    c = tl.reduce_dot(a, b, over="K", acc_dtype=tl.i32, engine=ENGINE)
    tl.store(C, c)


@triton.jit(model="weft")
def named_word(A: tl.View[tl.i16, (M, K)], B: tl.View[tl.i8, (N, K)], C: tl.View[tl.i32, (M, N)],
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, ENGINE: tl.constexpr = "auto"):
    a = tl.load(A)
    b = tl.load(B)
    c = tl.reduce_dot(a, b, over="K", acc_dtype=tl.i32, engine=ENGINE)
    tl.store(C, c)


@triton.jit(model="weft")
def mixed(A, B, C):
    m = tl.arange(0, 16)
    n = tl.arange(0, 16)
    k = tl.arange(0, 64)
    a = tl.load(A + m[:, None] * 64 + k[None, :])
    b = tl.load(B + k[:, None] * 16 + n[None, :])
    c = tl.dot(a, b, engine="matrix")
    d = tl.dot(a, b, engine="vector")
    tl.store(C + m[:, None] * 16 + n[None, :], c + d)
