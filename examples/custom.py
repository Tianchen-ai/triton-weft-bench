"""A standalone native implementation; no Weft/GGML/compiler installation needed."""
from pathlib import Path

from weft_bench import Case, PreparedRun

case = Case("custom_affine", "custom", {"N": 4096}, "native-affine", input_policy="i-mod17")


class Adapter:
    def __init__(self, options=None):
        self.options = options or {}

    def prepare(self, case, target, executor):
        from weft_bench.adapters.native import NATIVE, flags_for
        tools, flags = flags_for(target)
        executor.upload(NATIVE / "timer.h", "timer.h")
        source = executor.write("custom.cpp", r'''
#include "timer.h"
int main() {
  std::vector<float> x(WB_N), y(WB_N);
  for(size_t i=0;i<x.size();++i) x[i]=float(i%17);
  auto run=[&] { for(size_t i=0;i<x.size();++i) y[i]=2*x[i]+1; };
  run();
  for(size_t i=0;i<x.size();++i) if(y[i]!=2*x[i]+1) return 1;
  wb::measure(run, "exact");
}
''')
        binary = executor.workspace + "/custom"
        executor.command([*tools.get("cxx", ["c++"]), *flags, "-std=c++17", "-DWB_N="+str(case.shape["N"]),
                          source, "-o", binary], stage="compile")
        return PreparedRun([binary], executor.workspace)
