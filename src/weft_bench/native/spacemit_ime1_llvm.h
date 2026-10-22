#pragma once

// LLVM uses the upstream SpacemiT mnemonic prefix. Keep the vendor source's
// three-operand IME1 spelling while assembling the same instruction with
// +xsmtvdot. The macro deliberately does not accept IME2 operand forms.
#if defined(__clang__) && defined(__riscv)
__asm__(".macro vmadot vd, vs1, vs2\n"
        "smt.vmadot \\vd, \\vs1, \\vs2\n"
        ".endm\n");
#endif
