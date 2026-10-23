// Untimed validation interposer: compare the two functions on identical model rows.
#include "weft/ggml.h"
#include "ggml-quants.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#ifndef WB_DOT_ATOL
#define WB_DOT_ATOL 0.0001
#endif
#ifndef WB_DOT_RTOL
#define WB_DOT_RTOL 0.002
#endif

namespace {
thread_local ggml_weft_dot_binding selected{};
thread_local ggml_type weight_type = GGML_TYPE_COUNT;
std::atomic_flag captured = ATOMIC_FLAG_INIT;
std::atomic<uint64_t> checked_calls{0};

__attribute__((destructor)) void report_checked_calls() {
    std::fprintf(stderr, "WB_DOT_CHECK_CALLS %llu\n",
                 static_cast<unsigned long long>(checked_calls.load(std::memory_order_relaxed)));
}

void save_bytes(const std::string &path, const void *data, size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(static_cast<const char *>(data), bytes);
    if (!stream) {
        std::fprintf(stderr, "WB_DOT_CHECK cannot write %s\n", path.c_str());
        std::_Exit(87);
    }
}

void checked_dot(const uint8_t *w, const uint8_t *x, float *y, size_t n) {
    checked_calls.fetch_add(1, std::memory_order_relaxed);
    selected.function(w, x, y, n);
    float reference = 0;
    selected.original(static_cast<int>(n), &reference, 0, w, 0, x, 0, 1);
    // Same float tolerance as the ordinary quantized operator checker.
    if (std::isfinite(*y) && std::isfinite(reference) &&
        std::abs(double(*y) - reference) <= WB_DOT_ATOL + WB_DOT_RTOL * std::abs(double(reference)))
        return;
    if (captured.test_and_set()) return;
    const char *prefix = std::getenv("WB_DOT_CAPTURE");
    if (!prefix) {
        std::fprintf(stderr, "WB_DOT_CHECK missing capture destination\n");
        std::_Exit(87);
    }
    const auto activation = ggml_get_type_traits_cpu(weight_type)->vec_dot_type;
    save_bytes(std::string(prefix) + ".W.bin", w, ggml_row_size(weight_type, n));
    save_bytes(std::string(prefix) + ".X.bin", x, ggml_row_size(activation, n));
    std::vector<float> wf(n), xf(n);
    const auto decode_w = ggml_get_type_traits(weight_type)->to_float;
    const auto decode_x = ggml_get_type_traits(activation)->to_float;
    bool oracle_available = decode_w && (decode_x || activation == GGML_TYPE_Q8_K);
    double oracle = 0;
    if (oracle_available) {
        decode_w(w, wf.data(), n);
        if (decode_x) decode_x(x, xf.data(), n);
        else dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(x), xf.data(), n);
        for (size_t i = 0; i < n; ++i) oracle += double(wf[i]) * xf[i];
    }
    std::ofstream report(std::string(prefix) + ".json");
    report << std::setprecision(17) << "{\"type\":\"" << ggml_type_name(weight_type)
           << "\",\"K\":" << n << ",\"native\":";
    if (std::isfinite(reference)) report << reference; else report << "null";
    report << ",\"candidate\":";
    if (std::isfinite(*y)) report << *y; else report << "null";
    report << ",\"dequantized_dot\":";
    if (oracle_available && std::isfinite(oracle)) report << oracle; else report << "null";
    report << ",\"atol\":" << WB_DOT_ATOL << ",\"rtol\":" << WB_DOT_RTOL << "}\n";
    report.close();
    if (!report) std::_Exit(87);
    std::fprintf(stderr, "WB_DOT_CHECK mismatch type=%s K=%zu candidate=%.9g native=%.9g; captured %s\n",
                 ggml_type_name(weight_type), n, *y, reference, prefix);
    std::_Exit(86);
}
} // namespace

extern "C" ggml_weft_dot_binding ggml_weft_select_vec_dot(ggml_type type, ggml_vec_dot_t original) {
    using Select = ggml_weft_dot_binding (*)(ggml_type, ggml_vec_dot_t);
    static auto select = reinterpret_cast<Select>(dlsym(RTLD_NEXT, "ggml_weft_select_vec_dot"));
    if (!select) {
        std::fprintf(stderr, "WB_DOT_CHECK cannot resolve the native selection entry\n");
        std::_Exit(87);
    }
    auto binding = select(type, original);
    static const char *filter = std::getenv("WB_DOT_CHECK_TYPE");
    if (binding.function && (!filter || std::strcmp(filter, ggml_type_name(type)) == 0)) {
        selected = binding;
        weight_type = type;
        binding.function = checked_dot;
    }
    return binding;
}
