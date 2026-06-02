
// Ultimate benchmark for CnnInCpp v1.0 

#include "include/CnnInCpp/CnnInCpp.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <numeric>
#include <omp.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
double get_peak_ram_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
}
#else
double get_peak_ram_mb() { return 0.0; } 
#endif

using namespace CnnInCpp;

Model build_vgg_cifar() {
    Model model;
    model << conv2d(3, 64, 3, 1, 1) << batchnorm2d(64) << leaky_relu(0.01f)
          << conv2d(64, 64, 3, 1, 1) << batchnorm2d(64) << leaky_relu(0.01f)
          << maxpool2d(2, 2) 
          << conv2d(64, 128, 3, 1, 1) << batchnorm2d(128) << leaky_relu(0.01f)
          << conv2d(128, 128, 3, 1, 1) << batchnorm2d(128) << leaky_relu(0.01f)
          << maxpool2d(2, 2) 
          << flatten() << dense(128 * 8 * 8, 10);
    return model;
}

int main() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    std::cout << "====================================================\n";
    std::cout << "  CNN In C++ v1.0 - ULTIMATE ENGINE BENCHMARK\n";
    std::cout << "====================================================\n\n";

    // --- TEST 1: GRAPH INITIALIZATION ---
    std::cout << "--- TEST 1: ENGINE INITIALIZATION ---\n";
    auto init_start = std::chrono::high_resolution_clock::now();
    Model m = build_vgg_cifar();
    m.compile({128, 3, 32, 32});
    m.eval(); // Evaluates and strips backprop memory natively
    auto init_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> init_dur = init_end - init_start;
    std::cout << "Graph Construction, Compile & Eval Mode Setup: " << init_dur.count() << " ms\n\n";

    // --- TEST 2: BATCH SIZE SCALING ---
    std::cout << "--- TEST 2: BATCH SIZE SCALING (16 Threads) ---\n";
    omp_set_num_threads(16);
    std::vector<int> batch_sizes = {16, 64, 128, 256};
    for (int b : batch_sizes) {
        m.compile({b, 3, 32, 32});
        m.eval();
        Tensor x(b, 3, 32, 32); x.randomize();
        m.forward(x); // Warmup

        auto start = std::chrono::high_resolution_clock::now();
        for (int i=0; i<20; i++) m.forward(x);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;
        printf("Batch Size: %-3d | Throughput: %.2f img/sec\n", b, (b * 20) / dur.count());
    }
    std::cout << "\n";

    // --- TEST 3: THREAD SCALING ---
    std::cout << "--- TEST 3: THREAD SCALING (Batch 128) ---\n";
    m.compile({128, 3, 32, 32});
    m.eval();
    Tensor x128(128, 3, 32, 32); x128.randomize();
    
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    for (int t : thread_counts) {
        omp_set_num_threads(t);
        m.forward(x128); // Warmup
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i=0; i<20; i++) m.forward(x128);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;
        printf("Threads: %02d | Throughput: %.2f img/sec\n", t, (128 * 20) / dur.count());
    }
    std::cout << "\n";

    // --- TEST 4: MICRO-LATENCY & JITTER ---
    std::cout << "--- TEST 4: MICRO-LATENCY & JITTER (Batch 1, 1 Thread) ---\n";
    omp_set_num_threads(1);
    m.compile({1, 3, 32, 32});
    m.eval();
    Tensor single_img(1, 3, 32, 32); single_img.randomize();
    
    for(int i=0; i<50; i++) m.forward(single_img); // Warmup
    
    std::vector<double> latencies;
    latencies.reserve(500);
    for(int i=0; i<500; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        m.forward(single_img);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> d = end - start;
        latencies.push_back(d.count());
    }
    
    std::sort(latencies.begin(), latencies.end());
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    double p95 = latencies[(int)(latencies.size() * 0.95)];
    
    printf("Average Latency : %.4f ms\n", avg);
    printf("Min Latency     : %.4f ms\n", latencies.front());
    printf("Max Latency     : %.4f ms\n", latencies.back());
    printf("95th Percentile : %.4f ms\n\n", p95);

    // --- TEST 5: FULL BACKPROPAGATION ---
    std::cout << "--- TEST 5: FULL TRAINING STEP (Batch 128, 16 Threads) ---\n";
    omp_set_num_threads(16);
    m.compile({128, 3, 32, 32});
    m.train(); // Restores gradients natively!
    
    Adam opt(0.001f);
    CrossEntropyLoss criterion;
    Tensor y128(128, 10); y128.fill(0.0f); 
    for(int i=0; i<128; i++) y128(i, i%10) = 1.0f; // Fake one-hot labels

    // Warmup step
    Tensor preds = m.forward(x128);
    float loss = criterion.forward(preds, y128);
    Tensor dL = criterion.backward(preds, y128);
    m.backward(dL);
    opt.step(m.nodes);

    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<10; i++) {
        preds = m.forward(x128);
        loss = criterion.forward(preds, y128);
        dL = criterion.backward(preds, y128);
        m.backward(dL);
        opt.step(m.nodes);
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t_dur = t_end - t_start;
    double avg_step_ms = t_dur.count() / 10.0;
    
    printf("Avg Time per Full Training Step (Fwd+Bwd+Opt): %.2f ms\n", avg_step_ms);
    printf("Training Throughput: %.2f img/sec\n\n", (128.0 / (avg_step_ms / 1000.0)));

    // --- TEST 6: PEAK RAM USAGE ---
    std::cout << "--- TEST 6: PEAK RAM USAGE ---\n";
    printf("Peak Working Set Size: %.2f MB\n", get_peak_ram_mb());
    
    std::cout << "\n====================================================\n";
    return 0;
}


