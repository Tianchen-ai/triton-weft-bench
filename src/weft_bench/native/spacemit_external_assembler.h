#pragma once

// Upstream Clang can compile the standard RVV portions while a vendor GNU
// assembler encodes the inline IME instructions. Clang's file-level arch
// attribute resets assembler command-line extensions, so enable the extension
// after that attribute. Use only with -fno-integrated-as and an IME-capable as.
#if defined(__clang__) && defined(__riscv)
__asm__(".option arch,+xsmtvdotii");
#endif
