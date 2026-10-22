# Contributing

Keep case definitions, implementation adapters and transport separate. Compiler
optimizations and ISA support belong in Weft. Machine addresses, credentials,
models, build products and results must remain outside the tracked tree.

Add a small representative input with a native correctness reference. Define the
timed boundary explicitly, especially for quantization, packing, fused work and
mutable state. Keep a slower or failing implementation visible; do not silently
substitute another function or input. Baseline timing is optional.

Run `python -m unittest discover -s tests -v` for runner changes and a targeted real
case for driver changes. Do not rerun the full hardware matrix for documentation
or formatting edits. Mention actual device checks and unresolved failures in the
change description. Build a wheel with `python -m pip wheel . --no-deps`; the
declared build environment supplies the required setuptools version.
