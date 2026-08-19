#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                    cudaGetErrorName(err_), __FILE__, __LINE__,           \
                    cudaGetErrorString(err_));                            \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

__global__ void saxpy(const float *x, float *y, const int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n)
		y[idx] += 2.0 * x[idx];
}

int main(int argc, char **argv)
{
	int n = std::stoi(argv[1]);
	if(n == 0)
	{
		puts("SUM=0");
		return 0;
	}

	size_t bytes = (size_t)n * sizeof(float);
	float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);

	for(int i = 0; i < n; i++)
	{
		h_x[i] = ((i % 2048) - 1024) * 0.5f;
		h_y[i] = (i % 1024) - 512;
	}

	float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

	CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

	int threadsPerBlock = 1024;
	int blocksPerGrid = (n - 1) / threadsPerBlock + 1;

	GpuTimer timer;
	timer.start();

	saxpy<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, n);
	CUDA_CHECK(cudaGetLastError());

	float kernel_time = timer.stop_ms();

	CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

	double sum = 0;
	for(int i = 0; i < n; i++)
		sum += h_y[i];
	printf("SUM=%.0f KERNEL_TIME=%0.3f\n", sum, kernel_time);
	return 0;
}
