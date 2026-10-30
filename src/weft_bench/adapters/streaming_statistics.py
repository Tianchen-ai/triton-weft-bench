"""Native streaming-statistics references in the common benchmark worker."""
from ..core import NativeCall, PreparedNative


class StatisticsReference:
    def prepare_native(self, case, target, options, directory, executor, workload, name):
        blocked=options.get("blocked",False)
        block=int(options.get("block",256))
        if block<=0:raise ValueError("block must be positive")
        function="wb_welford_blocked" if blocked else "wb_welford_sequential"
        args="data.input.data(),output.data(),WB_M,WB_N"+(f",{block},{target.settings.get('threads',1)}" if blocked else "")
        return PreparedNative(NativeCall(name,"",function+"("+args+");",whole_workload=True),
                              details={"provider":"native-c++","algorithm":"block-merge" if blocked else "sequential-welford","block":block})
