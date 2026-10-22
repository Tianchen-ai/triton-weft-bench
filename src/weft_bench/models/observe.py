"""Optional instrumentation of the llama-bench harness, outside its timed region."""

from pathlib import Path
import re


def counter_capabilities(header):
    """Observe the supplied SDK API; an absent counter is not a zero count."""
    return {name: bool(re.search(r"\b" + function + r"\s*\(", header))
            for name, function in (("per_type", "ggml_weft_counters_get"),
                                   ("ime1", "ggml_weft_ime1_calls"),
                                   ("ime2", "ggml_weft_ime2_calls"),
                                   ("amx", "ggml_weft_amx_calls"))}


def instrument(source, *, weft_include=None):
    root = Path(source)
    path = root / "tools/llama-bench/llama-bench.cpp"
    original = text = path.read_text()
    capabilities = None
    if weft_include:
        header = Path(weft_include) / "weft/ggml.h"
        capabilities = counter_capabilities(header.read_text())
        if not capabilities["per_type"] or not capabilities["ime1"]:
            raise ValueError(f"integration SDK lacks the required row/IME counter API: {header}")
        if "WEFT_CALLS type=" in text or "WB_COUNTER sample=" in text:
            pattern = r'^#include "[^"\n]*/weft/ggml\.h"$'
            text, count = re.subn(pattern, lambda _: '#include "' + str(header.resolve()) + '"',
                                  text, flags=re.MULTILINE)
            if not count:
                raise ValueError("instrumented llama-bench has no explicit integration SDK header")
    if "PROFILE_WINDOW " not in text:
        begin = "            uint64_t t_start = get_time_ns();"
        end = "            uint64_t t_ns = get_time_ns() - t_start;"
        if text.count(begin) != 1 or text.count(end) != 1:
            raise ValueError(
                "llama-bench timing boundary changed; update the observer anchors"
            )
        text = "#include <time.h>\n" + text
        text = text.replace(
            begin,
            "            struct timespec wb_begin, wb_end;\n"
            "            clock_gettime(CLOCK_MONOTONIC, &wb_begin);\n" + begin,
        )
        text = text.replace(
            end,
            end + """
            clock_gettime(CLOCK_MONOTONIC, &wb_end);
            fprintf(stderr, "PROFILE_WINDOW %ld.%09ld %ld.%09ld pp=%d tg=%d sample=%d\\n",
                    wb_begin.tv_sec, wb_begin.tv_nsec, wb_end.tv_sec, wb_end.tv_nsec,
                    t.n_prompt, t.n_gen, i);""",
        )
    if weft_include and "WEFT_CALLS type=" not in text:
        header = Path(weft_include) / "weft/ggml.h"
        if not header.is_file():
            raise ValueError(f"missing integration header {header}")
        anchor = "            t.samples_ns.push_back(t_ns);"
        if text.count(anchor) != 1:
            raise ValueError("llama-bench sample boundary changed")
        text = '#include "' + str(header.resolve()) + '"\n' + text
        text = text.replace(
            anchor,
            """            for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
                ggml_weft_counters counts{};
                if (ggml_weft_counters_get(static_cast<ggml_type>(type), &counts) &&
                    (counts.vec_dot_calls || counts.weft_quantize_calls))
                    fprintf(stderr, "WEFT_CALLS type=%s dots=%llu quantizes=%llu\\n", ggml_type_name(static_cast<ggml_type>(type)),
                            (unsigned long long)counts.vec_dot_calls, (unsigned long long)counts.weft_quantize_calls);
            }
            fprintf(stderr, "WEFT_IME_CALLS %llu\\n", (unsigned long long)ggml_weft_ime1_calls());
""" + anchor,
        )
    if weft_include and "WB_COUNTER sample=" not in text:
        header = Path(weft_include) / "weft/ggml.h"
        helper = """
static void wb_measurement_counts(const char *point, int sample) {
  for(int type=0;type<GGML_TYPE_COUNT;++type) {
    ggml_weft_counters counts{};
    if(ggml_weft_counters_get(static_cast<ggml_type>(type), &counts))
      fprintf(stderr,"WB_COUNTER sample=%d point=%s type=%s dots=%llu quantizes=%llu ime=0\\n",sample,point,
              ggml_type_name(static_cast<ggml_type>(type)),(unsigned long long)counts.vec_dot_calls,(unsigned long long)counts.weft_quantize_calls);
  }
  fprintf(stderr,"WB_COUNTER sample=%d point=%s type=ime dots=0 quantizes=0 ime=%llu\\n",sample,point,(unsigned long long)ggml_weft_ime1_calls());
}
"""
        text = (
            "#include <cstdio>\n#include "
            + repr(str(header.resolve())).replace("'", '"')
            + "\n"
            + helper
            + text
        )
        starts = [
            "            struct timespec wb_begin, wb_end;",
            "            struct timespec profile_begin, profile_end;",
        ]
        start = next((s for s in starts if s in text), None)
        if not start:
            raise ValueError("missing timing-window declaration")
        text = text.replace(
            start, '            wb_measurement_counts("begin", i);\n' + start
        )
        anchor = "            t.samples_ns.push_back(t_ns);"
        text = text.replace(
            anchor, '            wb_measurement_counts("end", i);\n' + anchor
        )
    amx_line = '  fprintf(stderr,"WB_AMX_COUNTER sample=%d point=%s calls=%llu\\n",sample,point,(unsigned long long)ggml_weft_amx_calls());'
    if weft_include and not capabilities["amx"] and "WB_AMX_COUNTER sample=" in text:
        if text.count(amx_line) != 1:
            raise ValueError("unrecognized AMX observer; cannot remove an unavailable SDK call")
        text = text.replace(amx_line + "\n", "")
    if weft_include and capabilities["amx"] and "WB_AMX_COUNTER sample=" not in text:
        anchor = 'static void wb_measurement_counts(const char *point, int sample) {'
        if text.count(anchor) != 1:
            raise ValueError("missing measurement counter boundary")
        text = text.replace(anchor, anchor + "\n" + amx_line)
    ime2_line = '  fprintf(stderr,"WB_COUNTER sample=%d point=%s type=ime2 dots=0 quantizes=0 ime=%llu\\n",sample,point,(unsigned long long)ggml_weft_ime2_calls());'
    if weft_include and capabilities["ime2"] and "type=ime2 dots=" not in text:
        anchor = 'static void wb_measurement_counts(const char *point, int sample) {'
        if text.count(anchor) != 1:
            raise ValueError("missing IME2 measurement counter boundary")
        text = text.replace(anchor, anchor + "\n" + ime2_line)
    if weft_include and not capabilities["ime2"] and "type=ime2 dots=" in text:
        if text.count(ime2_line) != 1:
            raise ValueError("unrecognized IME2 observer for unavailable SDK counter")
        text = text.replace(ime2_line + "\n", "")
    if text != original:
        path.write_text(text)
    return {
        "source": str(path),
        "changed": text != original,
        "counters": "WEFT_CALLS type=" in text,
        "counter_capabilities": capabilities,
    }


if __name__ == "__main__":
    import json
    import sys

    request = json.loads(Path(sys.argv[1]).read_text())
    print(
        json.dumps(
            instrument(request["source"], weft_include=request.get("weft_include"))
        )
    )
