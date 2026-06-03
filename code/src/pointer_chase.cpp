// pointer_chase.cpp — synthetic random-access benchmark for NUMA latency.
// Allocates an array of N indices forming a single random Hamiltonian cycle,
// then walks the cycle for ITERS hops. Each hop is one cache-line miss with
// no compute to hide it. Run under numactl --membind to measure local-vs-
// remote DRAM latency cleanly.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char **argv) {
    size_t bytes  = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1ULL << 30);
    size_t iters  = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 200000000ULL;
    size_t n = bytes / sizeof(size_t);
    std::vector<size_t> arr(n);

    // Build a single random cycle through arr indices.
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), std::mt19937_64(42));
    for (size_t i = 0; i < n; ++i) arr[order[i]] = order[(i + 1) % n];

    auto t0 = std::chrono::high_resolution_clock::now();
    size_t pos = 0;
    for (size_t i = 0; i < iters; ++i) pos = arr[pos];
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ns = ms * 1e6 / iters;
    printf("BENCH_PC bytes=%zu iters=%zu wall_ms=%.1f ns_per_chase=%.2f final=%zu\n",
        bytes, iters, ms, ns, pos);
    return 0;
}
