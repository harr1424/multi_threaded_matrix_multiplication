#include <math.h>
#include <thread>

#include "../Clock.hpp"

// Conventional matrix multiplication, used for benchmarking my improvements:
__attribute__((noinline)) void naive_mult(
        double* __restrict C,
        double* __restrict B,
        double* __restrict A,
        const unsigned int N)
{
    for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j) {
            C[i * N + j] = 0.0;
            for (unsigned int k = 0; k < N; ++k)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
        }
}

// Print a given matrix 
void print(double* matrix, int N)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j != 0) {
                std::cout << ",";
            }
            std::cout << matrix[i*N + j];
        }
        std::cout << std::endl;
    }
}

// This is identical to naive_mult, but will be called in a multi-threaded context
__attribute__((noinline)) void thread_mult(
        double* __restrict C,
        double* __restrict B,
        double* __restrict A,
        const unsigned int N,
        unsigned int start,
        unsigned int end)
{
    for (unsigned int i = start; i < end; ++i)
        for (unsigned int j = 0; j < N; ++j) {
            C[i * N + j] = 0.0;
            for (unsigned int k = 0; k < N; ++k)
                C[i * N + j] += A[i * N + k] * B[j * N + k];
        }
}

// 27 X speed up with N = 10
// This function makes use of multithreading (see thread_mult() above)
__attribute__((noinline)) void mult(
        double* __restrict C,
        double* __restrict B,
        double* __restrict A, // A is never written to but passing by reference gives negligible speedup anyways
        const unsigned int N)
{
    // Transpose B: (by swapping half of the elements only)
    for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < i; ++j)
            std::swap(B[i * N + j], B[i + j * N]);

    // Set up for multithreading
    unsigned int num_threads = 12; // number of cores on my machine is 6, each core supports 2 threads
    std::thread threads[num_threads];
    unsigned int rows_thread = ceil(N/num_threads);

    // Create threads
    // This should partition into size N / num_threads and work on each problem concurrently
    for (unsigned int i = 0; i < num_threads; ++i) {
        unsigned int start = rows_thread * i;
        unsigned int end = (rows_thread * i) + rows_thread;
        if (i == (num_threads - 1) && end != (N - 1)) end = N;
        threads[i] = std::thread(&thread_mult, C, B, A, N, start, end);
    }

    // Activate threads
    for (unsigned int i = 0; i < num_threads; ++i)  threads[i].join();

    // The work of multiplying A nad B is now finished
    // Before passing B to naive_mult(), recover original state:
    // This is essential to avoid numeric error when comparing to C_naive
    // Transpose B: (by swapping half of the elements only)
    for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < i; ++j)
            std::swap(B[i * N + j], B[i + j * N]);
}

int main(int argc, char** argv)
{
    if (argc != 3)
        std::cerr << "usage: <log_problem_size> <seed>" << std::endl;
    else {
        const unsigned LOG_N = atoi(argv[1]);
        const unsigned long N = 1ul << LOG_N;
        const unsigned seed = atoi(argv[2]);
        srand(seed);

        double* A = new double[N * N];
        double* B = new double[N * N];

        for (unsigned long i = 0; i < N * N; ++i)
            A[i] = (rand() % 1000) / 999.0;
        for (unsigned long i = 0; i < N * N; ++i)
            B[i] = (rand() % 1000) / 999.0;

        double* C = new double[N * N];

        Clock c;
        mult(C, B, A, N);
        float mult_time = c.tock();

        double* C_naive = new double[N * N];
        c.tick();
        naive_mult(C_naive, B, A, N);
        float naive_mult_time = c.tock();

        double max_l1_error = 0.0;
        for (unsigned long i = 0; i < N * N; ++i)
            max_l1_error = std::max(max_l1_error, fabs(C_naive[i] - C[i]));
        std::cout << "Numeric error: " << max_l1_error << std::endl;

        float speedup = naive_mult_time / mult_time;
        std::cout << "Speedup: " << speedup << std::endl;

        std::cout << std::endl;

        // Verify numeric error:
        bool pass = true;
        if (max_l1_error >= 1e-4) {
            std::cerr << "FAIL: Numeric error is too high" << std::endl;
            pass = false;
        } else
            std::cout << "PASS: Numeric error" << std::endl;

        // Verify speedup:
        if (speedup < 5) {
            std::cerr << "FAIL: Speedup is too low" << std::endl;
            pass = false;
        } else
            std::cout << "PASS: Speedup" << std::endl;

        if (pass)
            std::cout << "OVERALL PASS" << std::endl;
        else {
            std::cout << "OVERALL FAIL" << std::endl;
            exit(1);
        }
    }

    return 0;
}
