

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>

#define PI 3.14159265358979323846

#define N 100000
#define BLOCK_SIZE 1024

__device__ void BoxMuller(float u1, float u2, float *n1, float *n2)
{
	float r = sqrtf(-2*logf(u1));
	float theta = 2*PI*(u2);
	*n1 = r*sinf(theta);
	*n2 = r*cosf(theta);
}

__global__ void norm_transform(float *dev_u1, float *dev_u2, 
		float *dev_n1, float *dev_n2, int size)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < size)
	{
		float res1;
		float res2;
		BoxMuller(dev_u1[tid], dev_u2[tid], &res1, &res2);
		dev_n1[tid] = res1;
		dev_n2[tid] = res2;
	}
	
}


int main()
{
	int dev_count;
	cudaGetDeviceCount(&dev_count);
	printf("Number of CUDA-capable devices: %d.\n", dev_count);
	cudaDeviceProp dev_prop;
	for (int j=0; j<dev_count; j++)
	{
		cudaGetDeviceProperties(&dev_prop, j);
		printf("Device number %d has max %d threads per block.\n", j, dev_prop.maxThreadsPerBlock);
		printf("Device number %d has %d multiprocessors.\n", j, dev_prop.multiProcessorCount);
	}
	int i;
	
	
	curandGenerator_t gen1, gen2;
	float *dev_u1, *dev_u2, *host_u1, *host_u2;
	float *dev_n1, *dev_n2, *host_n1, *host_n2;

	// allocate memory on the host
	//host_u1 = (float*)calloc(N, sizeof(float));
	//host_u2 = (float*)calloc(N, sizeof(float));
	
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	host_n1 = (float*)calloc(N, sizeof(float));
	host_n2 = (float*)calloc(N, sizeof(float));
		

	// allocate memory on the device
	cudaMalloc((void**)&dev_u1, N * sizeof(float));
	cudaMalloc((void**)&dev_u2, N * sizeof(float));
	
	cudaMalloc((void**)&dev_n1, N * sizeof(float));
	cudaMalloc((void**)&dev_n2, N * sizeof(float));
	
	
	// create a mersenne twister
	curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_MTGP32);
	curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_MTGP32);

	// set seed
	curandGenerateUniform(gen1, dev_u1, N);
	curandGenerateUniform(gen2, dev_u2, N);
	
	int numBlocks = ceil(float(N) / BLOCK_SIZE);
	
	// box muller transform
	norm_transform<<<numBlocks, BLOCK_SIZE>>>(dev_u1, dev_u2, dev_n1, dev_n2, N);
	

	// copy device memory to host
	cudaMemcpy(host_n1, dev_n1, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_n2, dev_n2, N * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	printf("Time elapsed to generate 2 x %d normal variables: %f seconds.\n",  N, elapsedTime/1000.0);

	printf("Random normal draws: \n");
	for (i = 0; i < 10; i++)
	{
		printf(" %1.4f  %1.4f\n", host_n1[i], host_n2[i]);
	}
	printf("\n");
	curandDestroyGenerator(gen1);
	curandDestroyGenerator(gen2);
	cudaDeviceReset();
	free(host_n1);
	free(host_n2);
	
	return 0;
}
