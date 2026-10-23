#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <time.h>
#include <sched.h>

namespace wb {
using Clock = std::chrono::steady_clock;
inline double option(const char *key, double fallback) {
  const char *value = std::getenv(key);
  if (!value) return fallback;
  char *end = nullptr;
  double parsed = std::strtod(value, &end);
  if (end == value || *end || !std::isfinite(parsed)) throw std::runtime_error(key);
  return parsed;
}
inline void barrier() { asm volatile("" ::: "memory"); }
inline double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
inline double calling_thread_ms() {
  timespec now;
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now) != 0)
    throw std::runtime_error("CLOCK_THREAD_CPUTIME_ID unavailable");
  return double(now.tv_sec) * 1000.0 + double(now.tv_nsec) * 0.000001;
}
inline long observed_frequency_khz() {
  const int cpu = sched_getcpu();
  if (cpu < 0) return -1;
  char path[128];
  std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
  FILE *file = std::fopen(path, "r");
  if (!file) return -1;
  long value = -1;
  const int read = std::fscanf(file, "%ld", &value);
  std::fclose(file);
  return read == 1 && value > 0 ? value : -1;
}
inline void evict(std::vector<std::uint8_t> &data) {
  for (std::size_t i = 0; i < data.size(); i += 64) data[i] += 1;
  barrier();
}

inline void describe_output_storage(const void *data, std::size_t bytes) {
  if (!data) return;
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  const auto alignment = std::min<std::uintptr_t>(4096, address & (~address + 1));
  std::printf(",\"output_storage\":{\"bytes\":%zu,\"alignment_bytes\":%zu,\"address_mod_256\":%zu}",
              bytes, std::size_t(alignment), std::size_t(address & 255));
}

// reset() restores mutable input between invocations, outside the measured region.
// Algorithm-required output initialization belongs inside run().
template<class Run, class Reset>
void measure(Run run, Reset reset, const char *correctness = "within-tolerance", bool stateful = false, const char *implementation = "",
             const void *output = nullptr, std::size_t outputBytes = 0) {
  const int warmup = static_cast<int>(option("WB_WARMUP", 3));
  const int samples = static_cast<int>(option("WB_SAMPLES", 5));
  const double budget = option("WB_BATCH_MS", 10);
  const char *mode = std::getenv("WB_CACHE");
  const bool cold = mode && std::strcmp(mode, "cold") == 0;
  if (warmup < 0 || samples < 1 || budget <= 0) throw std::runtime_error("invalid timing settings");
  std::vector<std::uint8_t> eviction(cold ? static_cast<std::size_t>(option("WB_EVICTION_BYTES", 67108864)) : 0, 1);
  for (int i = 0; i < warmup; ++i) { reset(); run(); barrier(); }
  Clock::time_point start;
  std::size_t iterations = 1;
  double calibration_ms = 0;
  if (!cold) {
    // A single cold invocation includes timer and dispatch latency and can
    // underestimate the required batch by an order of magnitude. Calibrate
    // actual batches geometrically, outside the samples, using the same run
    // boundary and restoring mutable inputs before each invocation.
    constexpr std::size_t maximum_iterations = 1000000;
    for (;;) {
      double elapsed = 0;
      if (stateful) {
        for (std::size_t i = 0; i < iterations; ++i) {
          reset();
          start = Clock::now();
          run(); barrier();
          elapsed += milliseconds(start);
        }
      } else {
        reset();
        start = Clock::now();
        for (std::size_t i = 0; i < iterations; ++i) { run(); barrier(); }
        elapsed = milliseconds(start);
      }
      calibration_ms += elapsed;
      if (elapsed >= budget || iterations == maximum_iterations) break;
      double factor = std::clamp(budget / std::max(elapsed, 0.000001), 2.0, 16.0);
      iterations = std::min<std::size_t>(maximum_iterations,
                                       std::ceil(double(iterations) * factor));
    }
  }
  std::vector<double> timings, cpu_timings;
  const long frequency_before = observed_frequency_khz();
  for (int sample = 0; sample < samples; ++sample) {
    double elapsed = 0;
    double cpu_elapsed = 0;
    if (stateful || cold) {
      for (std::size_t i = 0; i < iterations; ++i) {
        reset();
        if (cold) evict(eviction);
        const double cpu_start = calling_thread_ms();
        start = Clock::now();
        run(); barrier();
        elapsed += milliseconds(start);
        cpu_elapsed += calling_thread_ms() - cpu_start;
      }
    } else {
      reset();
      const double cpu_start = calling_thread_ms();
      start = Clock::now();
      for (std::size_t i = 0; i < iterations; ++i) { run(); barrier(); }
      elapsed = milliseconds(start);
      cpu_elapsed = calling_thread_ms() - cpu_start;
    }
    timings.push_back(elapsed / iterations);
    cpu_timings.push_back(cpu_elapsed / iterations);
  }
  const long frequency_after = observed_frequency_khz();
  std::printf("WB_RESULT {\"implementation\":\"%s\",\"samples_ms\":[", implementation);
  for (std::size_t i = 0; i < timings.size(); ++i) std::printf("%s%.12g", i ? "," : "", timings[i]);
  // Diagnostic only: this clock excludes descheduling, and excludes other
  // library/OpenMP workers. It never replaces the operator's wall-clock score.
  std::printf("],\"calling_thread_samples_ms\":[");
  for (std::size_t i = 0; i < cpu_timings.size(); ++i) std::printf("%s%.12g", i ? "," : "", cpu_timings[i]);
  std::printf("],\"observed_frequency_khz\":{\"before\":");
  if (frequency_before > 0) std::printf("%ld", frequency_before); else std::printf("null");
  std::printf(",\"after\":");
  if (frequency_after > 0) std::printf("%ld", frequency_after); else std::printf("null");
  std::printf("},\"cpu_time_scope\":\"calling-thread-only\",\"iterations\":%zu,\"calibration_ms\":%.12g,\"correctness\":\"%s\",\"cache\":\"%s\"", iterations, calibration_ms, correctness, cold ? "cold" : "reuse");
  describe_output_storage(output, outputBytes);
  std::printf("}\n");
}
template<class Run> void measure(Run run, const char *correctness = "within-tolerance") {
  measure(run, [] {}, correctness);
}
}
