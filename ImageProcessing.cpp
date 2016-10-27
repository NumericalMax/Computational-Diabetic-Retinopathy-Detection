/**********************************************************************************************

Author:			Maximilian Kapsecker

Name:			ImageProcessing

Created on:		02.07.2016

Inputs / Arguments:	None

Description:		This class holds various efficient image processing techniques, which are applicable for gray scale images.
			Thereby the CUDA interface is used to do image processing in parallel on the GPU.
				
Remark:			Minimal CUDA Architecture: >= SM 2.0
			Preferred CUDA Architecture: >= SM 3.5

**********************************************************************************************/

#include "ImageProcessing.h"

using namespace std;

/*----------------------------------------------------------------------------------------
 GENERAL DEFINITIONS
----------------------------------------------------------------------------------------*/

__constant__ float convolutionKernel[2048];
__constant__ int kernelStore[2048];

#define BIN_COUNT 256
#define BLOCK_N 64
#define HISTOGRAM_SIZE (BIN_COUNT * sizeof(unsigned int))
#define CO_OCCURENCE_SIZE (BIN_COUNT * BIN_COUNT * sizeof(unsigned int))

// Machine warp size and Emulator warp size
// Actually not needed, since we are keen on running the software on a machine with CUDA based architecture
// Never the less, NVIDIA documentation recommends this approach.
#ifndef __DEVICE_EMULATION__
	#define WARP_LOG_SIZE 5
#else
	#define WARP_LOG_SIZE 0
#endif

// WARPS executed in one block
#define WARP_N 6
// In our case: 6 << 5 = 192
// 110 + 5 times 0 = 11000000 = 192
// i.e. 192 Threads per Block
#define THREAD_N (WARP_N << WARP_LOG_SIZE)
#define BLOCK_MEMORY (WARP_N * BIN_COUNT)
#define BLOCK_MEMORY_CO (WARP_N * BIN_COUNT * BIN_COUNT)

// TO BE CHECKED: I READ THAT THIS IS A FASTER MULTIPLICATION APPROACH
// UPDATE: Indeed this instrinsic function seems to be a faster approach
// This yields the NVIDIA programming guide: https://www.cs.unc.edu/~prins/Classes/633/Readings/CUDA_C_Programming_Guide_4.2.pdf
// TODO: By now this approach is just used in Histogram kernel. Also use in the other kernels.
#define IMUL(a, b) __mul24(a, b)

/*----------------------------------------------------------------------------------------
INLINE ERROR HANDLER
----------------------------------------------------------------------------------------*/

#define gpuError(arg){gpuAssert((arg), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t err, const char *file, int line, bool fail = true){
	if (err != cudaSuccess)	{
	//	cout << endl;
	//	fprintf(stderr, "GPU Error: %s %s %d\n", cudaGetErrorString(err), file, line);
		if (fail){
			//ERROR HANDLING; GO TO NEXT IMAGE?
			cudaThreadExit();
		};
	}
}

/*----------------------------------------------------------------------------------------
GRID DETERMINATION
----------------------------------------------------------------------------------------*/
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

/*----------------------------------------------------------------------------------------
CONSTRUCTOR AND DECONSTRUCTOR
 ----------------------------------------------------------------------------------------*/

ImageProcessing::ImageProcessing(){

}

ImageProcessing::~ImageProcessing(){

}

/*----------------------------------------------------------------------------------------
HISTOGRAM
----------------------------------------------------------------------------------------*/

// Histogram implementation is mainly inspired by the NVIDIA Corporation.
// Copyright reference:
/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */
__global__ void mergeHistogram256(unsigned int *histogram){

	__shared__ unsigned int data[64];

	// Refering to Visual Profiler: uncoalesced reads!
	data[threadIdx.x] = histogram[IMUL(threadIdx.x, 256) + blockIdx.x];

	// 32 >>= 1 = 16
	// 16 >>= 1 = 8
	// ...
	// stride = 64 / 2
	for (int stride = 32; stride > 0; stride >>= 1){
		__syncthreads();
		if (threadIdx.x < stride){
			data[threadIdx.x] += data[threadIdx.x + stride];
		}
	}

	if (threadIdx.x == 0){
		histogram[blockIdx.x] = data[0];
	}
}

// IMPORTANT: Identifier volatile, otherwise bank conflicts would arise.
// volatile affects that the compiler does not optimize execution, which is in this case important,
// since we like to write back to SM, and don't want any race conditions or blocked memory.
__device__ void addData256(volatile unsigned int *s_WarpHist, unsigned int data, unsigned int threadTag){

	unsigned int count;
	do{
		// EXPLANATION (example): 	1100 & 1010 = 1000
		//				1100 | 1010 = 1110

		count = s_WarpHist[data] & 0x07FFFFFFU;
		count = threadTag | (count + 1);
		s_WarpHist[data] = count;
	} while (s_WarpHist[data] != count);
}


__global__ void histogram256(uint8_t *inputImage, unsigned int *histogram, int width, int height){
	
	// ROUND UP... If width * height % 4 equals not zero it leads to a marginal wrong result
	int dataN = width * height / 4;

	const int globalTid = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	const int numThreads = IMUL(blockDim.x, gridDim.x);

	#ifndef __DEVICE_EMULATION__

		// WARP_LOG_SIZE higher bits of counter values are tagged 
		// by lower WARP_LOG_SIZE threadID bits
		const unsigned int threadTag = threadIdx.x << (32 - WARP_LOG_SIZE);
	#else
		// Explicitly set to zero to avoid potential troubles
		const unsigned int threadTag = 0;
	#endif
	
	// Shared memory cache for each warp in current thread block
	// Declare as volatile to prevent incorrect compiler optimizations in addData()
	volatile __shared__ unsigned int s_Hist[BLOCK_MEMORY];

	// Current warp shared memory frame
	const int warpBase = IMUL(threadIdx.x >> WARP_LOG_SIZE, BIN_COUNT);

	// Clear shared memory buffer for current thread block before processing
	for (int pos = threadIdx.x; pos < BLOCK_MEMORY; pos += blockDim.x){
		s_Hist[pos] = 0;
	}

	__syncthreads();

	// Cycle through the entire data set, update subhistograms for each warp
	// Since threads in warps always execute the same instruction,
	// we are safe with the addData trick
	for (int pos = globalTid; pos < dataN; pos += numThreads){

		unsigned int data4 = inputImage[pos];

		// Avoiding intra-warp shared memory collisions.
		addData256(s_Hist + warpBase, (data4 >> 0) & 0xFFU, threadTag);
		addData256(s_Hist + warpBase, (data4 >> 8) & 0xFFU, threadTag);
		addData256(s_Hist + warpBase, (data4 >> 16) & 0xFFU, threadTag);
		addData256(s_Hist + warpBase, (data4 >> 24) & 0xFFU, threadTag);
	}

	__syncthreads();

	// Writing block sub-histogram into global memory. 
	// Merge per-warp histograms into per-block and write to global memory
	for (int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x){
		unsigned int sum = 0;

		for (int base = 0; base < BLOCK_MEMORY; base += BIN_COUNT){
			sum += s_Hist[base + pos] & 0x07FFFFFFU;
		}

		//REVERSE #ifdef ATOMICS
			atomicAdd(histogram + pos, sum);
		//REVERSE #else
		//	histogram[IMUL(BIN_COUNT, blockIdx.x) + pos] = sum;
		//REVERSE #endif
	}
}


__global__ void histogramEqualization(uint8_t *in, uint8_t *out, unsigned int *histTrafo, int width, int height){
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = index_y * width + index_x;

	out[index] = histTrafo[in[index]];
}

/*----------------------------------------------------------------------------------------
MORPHOLOGICAL OPERATIONS & KERNEL METHODS
 
are each splitted in a row and column operation.
For Instance convolution with the matrix [-1, -2, -1; 0, 0, 0; 1, 2, 1] is the same as applying [1, 2, 1]^T and afterwards [-1, 0, 1]
The second approach offers more flexibility in implementation and further reducing arithmetic complexity.

ATTENTION: A filter (matrix) is only seperable iff its a product of two outter vectors, which is the same as rank(Matrix) = 1
FOLLOWING (TODO): the implementation of the following dilation and erosion is only applicable for e.g. rectangle SE and not for instance SE = disk.
--> DO Morphological Operation with out commented code, even though it is not optimized.
----------------------------------------------------------------------------------------*/

__global__ void dilationRowGPU(uint8_t *in, uint8_t *out, int width, int height, int radius){

	extern __shared__ uint8_t data[];

	// Determining the different positions required for computation
	// For reference: See tile image in paper
	// TODO: It may be not a good idea to hardcode e.g. blocksize of 128
	const int begin = IMUL(blockIdx.x, 128);
	const int end = begin + 128 - 1;
	const int start = begin - radius;
	const int finish = end + radius;
	const int clampEnd = min(end, width - 1);
	const int clampStart = max(start, 0);
	const int clampedEnd = min(finish, width - 1);
	const int rowStart = IMUL(blockIdx.y, width);
	const int alignStart = begin - 16;
	const int loadPos = alignStart + threadIdx.x;

	// Write required data from global to shared memory
	if (loadPos >= start){
		const int smPos = loadPos - start;
		data[smPos] = ((loadPos >= clampStart) && (loadPos <= clampedEnd)) ? in[rowStart + loadPos] : 0;
	}

	__syncthreads();

	// Search for maximum in x-Direction
	const int writePos = begin + threadIdx.x;
	if (writePos <= clampEnd){
		const int smPos = writePos - start;
		float max = 0;
		for (int k = -radius; k <= radius; k++){
			max = fmaxf((float)data[smPos + k] * kernelStore[radius - k], max);
		}
		out[rowStart + writePos] = (uint8_t)max;
	}
}

__global__ void dilationColumnGPU(uint8_t *in, uint8_t *out, int width, int height, int radius, int smemStride, int gmemStride){

	extern __shared__ uint8_t data[];

	// Determining the different positions required for computation
	// For reference: See tile image in paper
	// TODO: It may be not a good idea to hardcode e.g. blocksize of 48
	const int begin = IMUL(blockIdx.y, 48);
	const int end = begin + 48 - 1;
	const int start = begin - radius;
	const int finish = end + radius;
	const int clampEnd = min(end, height - 1);
	const int clampStart = max(start, 0);
	const int clampedEnd = min(finish, height - 1);
	const int colStart = IMUL(blockIdx.x, 16) + threadIdx.x;

	int smPos = IMUL(threadIdx.y, 16) + threadIdx.x;
	int gmPos = IMUL(start + threadIdx.y, width) + colStart;

	for (int y = start + threadIdx.y; y <= finish; y += blockDim.y){
		data[smPos] = ((y >= clampStart) && (y <= clampedEnd)) ? in[gmPos] : 0;
		smPos += smemStride;
		gmPos += gmemStride;
	}

	__syncthreads();

	smPos = IMUL(threadIdx.y + radius, 16) + threadIdx.x;
	gmPos = IMUL(begin + threadIdx.y, width) + colStart;

	// Search for maximum in y-Direction
	for (int y = begin + threadIdx.y; y <= clampEnd; y += blockDim.y){
		float max = 0;
		for (int k = -radius; k <= radius; k++){
			max = fmaxf((float)data[smPos + IMUL(k, 16)] * kernelStore[radius - k], max);
		}
		out[gmPos] = (uint8_t)max;
		smPos += smemStride;
		gmPos += gmemStride;
	}
}

__global__ void erosionRowGPU(uint8_t *in, uint8_t *out, int width, int height, int radius){

	extern __shared__ uint8_t data[];

	const int begin = IMUL(blockIdx.x, 128);
	const int end = begin + 128 - 1;
	const int start = begin - radius;
	const int finish = end + radius;
	const int clampEnd = min(end, width - 1);
	const int clampStart = max(start, 0);
	const int clampedEnd = min(finish, width - 1);
	const int rowStart = IMUL(blockIdx.y, width);
	const int alignStart = begin - 16;
	const int loadPos = alignStart + threadIdx.x;

	if (loadPos >= start){
		const int smPos = loadPos - start;
		data[smPos] = ((loadPos >= clampStart) && (loadPos <= clampedEnd)) ? in[rowStart + loadPos] : 0;
	}

	__syncthreads();

	const int writePos = begin + threadIdx.x;
	if (writePos <= clampEnd){
		const int smPos = writePos - start;
		float min = 256;
		for (int k = -radius; k <= radius; k++){
			if (kernelStore[radius - k] == 1){
				min = fminf((float)data[smPos + k] * kernelStore[radius - k], min);
			}
		}
		out[rowStart + writePos] = (uint8_t)min;
	}
}

__global__ void erosionColumnGPU(uint8_t *in, uint8_t *out, int width, int height, int radius, int smemStride, int gmemStride){

	extern __shared__ uint8_t data[];

	const int begin = IMUL(blockIdx.y, 48);
	const int end = begin + 48 - 1;
	const int start = begin - radius;
	const int finish = end + radius;
	const int clampEnd = min(end, height - 1);
	const int clampStart = max(start, 0);
	const int clampedEnd = min(finish, height - 1);
	const int colStart = IMUL(blockIdx.x, 16) + threadIdx.x;

	int smPos = IMUL(threadIdx.y, 16) + threadIdx.x;
	int gmPos = IMUL(start + threadIdx.y, width) + colStart;

	for (int y = start + threadIdx.y; y <= finish; y += blockDim.y){
		data[smPos] = ((y >= clampStart) && (y <= clampedEnd)) ? in[gmPos] : 0;
		smPos += smemStride;
		gmPos += gmemStride;
	}

	__syncthreads();

	smPos = IMUL(threadIdx.y + radius, 16) + threadIdx.x;
	gmPos = IMUL(begin + threadIdx.y, width) + colStart;

	for (int y = begin + threadIdx.y; y <= clampEnd; y += blockDim.y){
		float min = 256;
		for (int k = -radius; k <= radius; k++){
			if (kernelStore[radius - k] == 1){
				min = fminf((float)data[smPos + IMUL(k, 16)] * kernelStore[radius - k], min);
			}
		}
		out[gmPos] = (uint8_t)min;
		smPos += smemStride;
		gmPos += gmemStride;
	}
}

/*----------------------------------------------------------------------------------------
KERNEL FILTER
----------------------------------------------------------------------------------------*/

__global__ void convolutionRowGPU(uint8_t *out, uint8_t *in, int width, int height, int radius){

	extern __shared__ uint8_t data[];

	// Clamp to tile borders and image borders. For reference: see Paper
	const int begin = IMUL(blockIdx.x, 128);
	const int end = begin + 128 - 1;
	const int start = begin - radius;
	const int finish = end + radius;
	const int clampEnd = min(end, width - 1);
	const int clmapStart = max(start, 0);
	const int clampedEnd = min(finish, width - 1);
	const int rowStart = IMUL(blockIdx.y, width);
	const int alginStart = begin - 16;
	const int loadPos = alginStart + threadIdx.x;

	if (loadPos >= start){
		// loading required tile (with respect to radius) to SM
		const int smPos = loadPos - start;
		data[smPos] = ((loadPos >= clmapStart) && (loadPos <= clampedEnd)) ? in[rowStart + loadPos] : 0;
	}

	__syncthreads();

	const int writePos = begin + threadIdx.x;
	if (writePos <= clampEnd){
		const int smemPos = writePos - start;
		float sum = 0;
		for (int k = -radius; k <= radius; k++){
			sum += (float)data[smemPos + k] * convolutionKernel[radius - k];
		}
		out[rowStart + writePos] = (uint8_t)sum;
	}
}

__global__ void convolutionColumnGPU(uint8_t *out, uint8_t *in, int width, int height, int radius, int smemStride, int gmemStride){
	
	extern __shared__ uint8_t data[];

	const int begin = IMUL(blockIdx.y, 48);
	const int end = begin + 48 - 1;
	const int start = begin - radius;
	const int finish = end + radius;
	const int clampEnd = min(end, height - 1);
	const int clampStart = max(start, 0);
	const int clampedEnd = min(finish, height - 1);
	const int colStart = IMUL(blockIdx.x, 16) + threadIdx.x;

	int smPos = IMUL(threadIdx.y, 16) + threadIdx.x;
	int gmPos = IMUL(start + threadIdx.y, width) + colStart;

	// fill shared memory with required values
	for (int y = start + threadIdx.y; y <= finish; y += blockDim.y){
		data[smPos] = ((y >= clampStart) && (y <= clampedEnd)) ? in[gmPos] : 0;
		smPos += smemStride;
		gmPos += gmemStride;
	}

	__syncthreads();

	smPos = IMUL(threadIdx.y + radius, 16) + threadIdx.x;
	gmPos = IMUL(begin + threadIdx.y, width) + colStart;
	for (int y = begin + threadIdx.y; y <= clampEnd; y += blockDim.y){
		float sum = 0;
		for (int k = -radius; k <= radius; k++){
			sum += (float)data[smPos + IMUL(k, 16)] * convolutionKernel[radius - k];
		}	
		out[gmPos] = (uint8_t)sum;
		smPos += smemStride;
		gmPos += gmemStride;
	}
}


/*----------------------------------------------------------------------------------------
REDUCTION METHODS
----------------------------------------------------------------------------------------*/

//With template: If-blocks inside kernel are evaluated at compile time!!!
template <unsigned int blockSize>
__device__ void warpMax(volatile uint8_t *sdata, volatile int *slocation, unsigned int tid){

	if (blockSize >= 64){
		sdata[tid] = fmaxf(sdata[tid + 32], sdata[tid]);
		slocation[tid] = (sdata[tid] > sdata[tid + 32]) ? slocation[tid] : slocation[tid + 32];
	}
	if (blockSize >= 32){
		sdata[tid] = fmaxf(sdata[tid + 16], sdata[tid]);
		slocation[tid] = (sdata[tid] > sdata[tid + 16]) ? slocation[tid] : slocation[tid + 16];
	}
	if (blockSize >= 16){
		sdata[tid] = fmaxf(sdata[tid + 8], sdata[tid]);
		slocation[tid] = (sdata[tid] > sdata[tid + 8]) ? slocation[tid] : slocation[tid + 8];
	}
	if (blockSize >= 8){
		sdata[tid] = fmaxf(sdata[tid + 4], sdata[tid]);
		slocation[tid] = (sdata[tid] > sdata[tid + 4]) ? slocation[tid] : slocation[tid + 4];
	}
	if (blockSize >= 4){
		sdata[tid] = fmaxf(sdata[tid + 2], sdata[tid]);
		slocation[tid] = (sdata[tid] > sdata[tid + 2]) ? slocation[tid] : slocation[tid + 2];
	}
	if (blockSize >= 2){
		sdata[tid] = fmaxf(sdata[tid + 1], sdata[tid]);
		slocation[tid] = (sdata[tid] > sdata[tid + 1]) ? slocation[tid] : slocation[tid + 1];
	}
}

template <unsigned int blockSize>
__global__ void maxElement(uint8_t *currentImage, int *value, int *location, int width, int height){

	extern __shared__ uint8_t sdata[];
	__shared__ int slocation[blockSize];
	int n = width * height;
	unsigned int tid = threadIdx.x;

	// Subsequent we load two values to SM, therefore we multiply by 2
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	sdata[tid] = 0;
	slocation[tid] = 0;

	// Fill Shared Memory Array
	while (i < n){
		// Instead of loading one value to SM per thread, we load two values to SM.
		// This affects Latency hiding and reduction of blocks needed
		sdata[tid] = fmaxf(currentImage[i], currentImage[i + blockSize]);
		slocation[tid] = (currentImage[i] > currentImage[i + blockSize]) ? 1000000 : (i + blockSize);
		
		i += gridSize;
	}

	__syncthreads();

	// Actual reduction
	// sequential addressing
	if (blockSize >= 512){
		if (tid < 256){
			sdata[tid] = fmaxf(sdata[tid + 256], sdata[tid]);
			slocation[tid] = (sdata[tid] > sdata[tid + 256]) ? (slocation[tid]) : (slocation[tid + 256]);
		}
		__syncthreads();
	}
	if (blockSize >= 256)
	{
		if (tid < 128){
			sdata[tid] = fmaxf(sdata[tid + 128], sdata[tid]);
			slocation[tid] = (sdata[tid] > sdata[tid + 128]) ? (slocation[tid]) : (slocation[tid + 128]);
		}
		__syncthreads();
	}
	if (blockSize >= 128){
		if (tid < 64){
			sdata[tid] = fmaxf(sdata[tid + 64], sdata[tid]);
			slocation[tid] = (sdata[tid] > sdata[tid + 64]) ? (slocation[tid]) : (slocation[tid + 64]);
		}
		__syncthreads();
	}

	// The last warp (i.e. the last 32 threads) are handled seperate
	// e.g. no syncthreads() is needed
	if (tid < 32) {
		warpMax<blockSize>(sdata, slocation, tid);
	}

	// Writing the final result
	if (tid == 0){
		value[blockIdx.x] = sdata[0];
		location[blockIdx.x] = slocation[0];
	}
}




//With template: If-blocks inside kernel are evaluated at compile time!!!
template <unsigned int blockSize>
__device__ void warpMean(volatile double *sdata2, unsigned int tid){

        if (blockSize >= 64){
                sdata2[tid] = (sdata2[tid + 32] + sdata2[tid]) / 2.0;
        }
        if (blockSize >= 32){
                sdata2[tid] = (sdata2[tid + 16] + sdata2[tid]) / 2.0;
        }
        if (blockSize >= 16){
                sdata2[tid] = (sdata2[tid + 8] + sdata2[tid]) / 2.0;
        }
        if (blockSize >= 8){
                sdata2[tid] = (sdata2[tid + 4] + sdata2[tid]) / 2.0;
        }
        if (blockSize >= 4){
                sdata2[tid] = (sdata2[tid + 2] + sdata2[tid]) / 2.0;
        }
        if (blockSize >= 2){
                sdata2[tid] = (sdata2[tid + 1] + sdata2[tid]) / 2.0;
        }
}

template <unsigned int blockSize>
__global__ void meanValue(uint8_t *currentImage, double *value, int width, int height){

        extern __shared__ double sdata3[];
        int n = width * height;
        unsigned int tid = threadIdx.x;

        // Subsequent we load two values to SM, therefore we multiply by 2
        unsigned int i = blockIdx.x*(blockSize * 2) + tid;
        unsigned int gridSize = blockSize * 2 * gridDim.x;

        sdata3[tid] = 0.0;

        // Fill Shared Memory Array
        while (i < n){
                // Instead of loading one value to SM per thread, we load two values to SM.
                // This affects Latency hiding and reduction of blocks needed
                sdata3[tid] = (currentImage[i] + currentImage[i + blockSize]) / 2.0;

                i += gridSize;
        }

        __syncthreads();

        // Actual reduction
        // sequential addressing
        if (blockSize >= 512){
                if (tid < 256){
                        sdata3[tid] = (sdata3[tid + 256] + sdata3[tid]) / 2.0;
                }
                __syncthreads();
        }
        if (blockSize >= 256)
        {
                if (tid < 128){
                        sdata3[tid] = (sdata3[tid + 128] + sdata3[tid]) / 2.0;
                }
                __syncthreads();
        }
        if (blockSize >= 128){
                if (tid < 64){
                        sdata3[tid] = (sdata3[tid + 64] + sdata3[tid]) / 2.0;
                }
                __syncthreads();
        }

        // The last warp (i.e. the last 32 threads) are handled seperate
        // e.g. no syncthreads() is needed
        if (tid < 32) {
                warpMean<blockSize>(sdata3, tid);
        }

        // Writing the final result
        if (tid == 0){
                value[blockIdx.x] = sdata3[0];
        }
}


template <unsigned int blockSize>
__device__ void warpArea(volatile int *sdata, unsigned int tid){

	if (blockSize >= 64){ sdata[tid] += sdata[tid + 32]; }
	if (blockSize >= 32){ sdata[tid] += sdata[tid + 16]; }
	if (blockSize >= 16){ sdata[tid] += sdata[tid + 8]; }
	if (blockSize >= 8){ sdata[tid] += sdata[tid + 4]; }
	if (blockSize >= 4){ sdata[tid] += sdata[tid + 2]; }
	if (blockSize >= 2){ sdata[tid] += sdata[tid + 1]; }
}


template <unsigned int blockSize>
__global__ void getArea(uint8_t *currentImage, int *value, int width, int height){

	int n = width * height;

	extern __shared__ int sdata1[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize) + tid;
	unsigned int gridSize = blockSize * gridDim.x;

	int sdata = 0;
	
	while (i < n) {
		if (currentImage[i] == 255){
			sdata += 1;
		}
		i += gridSize;
	}
	sdata1[tid] = sdata;

	__syncthreads();

	if (blockSize >= 512) {
		if (tid < 256){
			sdata1[tid] += sdata1[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256){
		if (tid < 128){
			sdata1[tid] += sdata1[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128){
		if (tid < 64){
			sdata1[tid] += sdata1[tid + 64];
		}
		__syncthreads();
	}

	if (tid < 32) {
		warpArea<blockSize>(sdata1, tid);
	}
	if (tid == 0){
		value[blockIdx.x] = sdata1[0];
	}
}

/*----------------------------------------------------------------------------------------
 NAIV IMAGE PROCESSING TECHNIQUES
----------------------------------------------------------------------------------------*/

// PROBABLY WRONG APPROACH, COULD BE INCLUDED IN REDUCTION METHOD
__global__ void maxElementPosition(uint8_t *currentImage, int *coordinate, int value, int width, int height){

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = index_y * width + index_x;

	const int clampWidth = min(index_x, width - 1);
	const int clampHeight = min(index_y, height - 1);

	if (index_x < width && index_y < height){
		//may cause unnecessary overwritings, Never the less not so important here
		if (currentImage[index] == value){
			coordinate[0] = index_x;
			coordinate[1] = index_y;
		}
	}
}


__global__ void  imadjust_d(uint8_t *currentImage, uint8_t *outputImage, int width, int height, int lowerLimit, int upperLimit, double lowerScale, double upperScale){

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = index_y * width + index_x;

	if (index_x < width && index_y < height){
		if (currentImage[index] >= upperLimit){
			if (currentImage[index] * upperScale > 255){
				outputImage[index] = 255;
			}
			else{
				outputImage[index] = (uint8_t)(currentImage[index] * upperScale);
			}			
		}
		else if(currentImage[index] <= lowerLimit){
			outputImage[index] = (uint8_t) (currentImage[index] * lowerScale);
		}
		else{
			outputImage[index] = (uint8_t)(currentImage[index]);
		}
	}
}


__global__ void invert_d(uint8_t *in, uint8_t *out, int width, int height){
	
	
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = index_y * width + index_x;

	if (index_x < width && index_y < height){
		out[index] = 255 - in[index];
	}
}


__global__ void threshold(uint8_t *in, uint8_t *out, int width, int height, int thresholdValue){

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (index_x < width && index_y < height){

		int index = index_y * width + index_x;

		if (in[index] >= thresholdValue){
			out[index] = 255;
		}
		else{
			out[index] = 0;
		}
	}
}

__global__ void subtractImages_d(uint8_t *in1, uint8_t *in2, uint8_t *out, int width, int height){

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = index_y * width + index_x;
	if (index_x < width && index_y < height){

		int diff = (int)(in1[index] - in2[index]);

		if (diff < 0){
			out[index] = 0;
		}
		else{
			out[index] = in1[index] - in2[index];
		}
	}
}

__global__ void removeArea(uint8_t *currentImage, uint8_t *outputImage, int width, int height, int x, int y, int radius){
	
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (index_x < width && index_y < height){
		int index = index_y * width + index_x;

		if (abs(x - index_x) < radius && abs(y - index_y) < radius){
			outputImage[index] = 0;
		}
	}
}

__global__ void linearTrafo(uint8_t *currentImage, uint8_t *outputImage, int width, int height, float alpha, int beta){

        int index_x = blockIdx.x * blockDim.x + threadIdx.x;
        int index_y = blockIdx.y * blockDim.y + threadIdx.y;

        if (index_x < width && index_y < height){

                int index = index_y * width + index_x;

                if (alpha * currentImage[index] + beta > 255){
			outputImage[index] = 255;
                }
		else if(alpha * currentImage[index] + beta < 0){
			outputImage[index] = 0;
		}
		else{
			outputImage[index] = __float2int_rd(alpha * currentImage[index] + beta);
		}
        }
}


/*----------------------------------------------------------------------------------------
CO OCCURENCE MATRIX
----------------------------------------------------------------------------------------*/

template <unsigned int blockSize>
__device__ void warpTexture(volatile double *sdata3, unsigned int tid){
    
    if (blockSize >= 64){ sdata3[tid] += sdata3[tid + 32]; }
    if (blockSize >= 32){ sdata3[tid] += sdata3[tid + 16]; }
    if (blockSize >= 16){ sdata3[tid] += sdata3[tid + 8]; }
    if (blockSize >= 8){ sdata3[tid] += sdata3[tid + 4]; }
    if (blockSize >= 4){ sdata3[tid] += sdata3[tid + 2]; }
    if (blockSize >= 2){ sdata3[tid] += sdata3[tid + 1]; }
}


template <unsigned int blockSize>
__global__ void textureDetermination(uint8_t *currentImage, double *value, int width, int height){
    
    int n = width * height;
    double p = 1.0 / n;
    
    extern __shared__ double sdata2[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize) + tid;
    unsigned int gridSize = blockSize * gridDim.x;
    
    //sdata1[tid] = 0;
    
	double temp = 0.0;

    while (i < n) {
        
        temp += (currentImage[i] - currentImage[i + 1]) * (currentImage[i] - currentImage[i + 1]) * p;

        i += gridSize;
    }
    
	sdata2[tid] = temp;

    __syncthreads();
    
    if (blockSize >= 512) {
        if (tid < 256){
            sdata2[tid] += sdata2[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256){
        if (tid < 128){
            sdata2[tid] += sdata2[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128){
        if (tid < 64){
            sdata2[tid] += sdata2[tid + 64];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        warpTexture<blockSize>(sdata2, tid);
    }
    if (tid == 0){
        value[blockIdx.x] = sdata2[0];
    }
}


/*----------------------------------------------------------------------------------------
// WRAPPER FUNCTIONS / CPU INTERFACE
------------------------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------------------
HISTOGRAM
 
Following function computed the histogram for 256 discrete gray levels

@param input image
@param image pointer onto result array
@param width The width of the underlying image
@param height The height of the underlying image
-----------------------------------------------------------------------------------------*/

void ImageProcessing::histogramGPU(uint8_t *currentImage, unsigned int *h_Result, int width, int height, bool timing){

	// init histogram
	unsigned int *d_Result256;
    for (int i = 0; i < 256; i++){
		h_Result[i] = 0;
	}

	// ATOMICS is full available since compute capability 1.2
	// Never the less, atomics might be not the best approach
    // (memory locations become blocked and may result in performance decrease)
	//#ifdef ATOMICS
		gpuError(cudaMalloc((void **)&d_Result256, HISTOGRAM_SIZE));
	//#else
	//	gpuError(cudaMalloc((void **)&d_Result256, BLOCK_N * HISTOGRAM_SIZE));
	//#endif

	cudaThreadSynchronize();

	// compute histogram
	//REVERSE #ifdef ATOMICS
		gpuError(cudaMemset(d_Result256, 0, HISTOGRAM_SIZE));

        cudaEvent_t start,stop;
        if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
        }

		histogram256 <<<BLOCK_N, THREAD_N>>>(currentImage, d_Result256, width, height);


        if(timing){
                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
		cout << "Computed Histogram in Image of Size " << width << " x " << height << " in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);




	//REVERSE #else
	//REVERSE	gpuError(cudaMemset(d_Result256, 0, HISTOGRAM_SIZE));
	//REVERSE	histogram256 <<<BLOCK_N, THREAD_N>>>(currentImage, d_Result256, width, height);
		// WITH THE ABSENCE OF ATOMIC OPERATIONS FOLLOWING KERNEL BECOMES NECESSARY
	//REVERSE	mergeHistogram256 <<<BLOCK_N, THREAD_N>>>(d_Result256);
	//REVERSE #endif
	
	// HERE ARISES FREQUENTLY AN UNKOWN ERROR
	gpuError(cudaMemcpy(h_Result, d_Result256, HISTOGRAM_SIZE, cudaMemcpyDeviceToHost));

	cudaThreadSynchronize();

	// close histogram
	gpuError(cudaFree(d_Result256));

}

/*----------------------------------------------------------------------------------------
HISTOGRAM EQUALIZATION
 
Following function performs histogram equlization

@param currentImage     input image
@param outputImage      output image
@param width            The width of the underlying image
@param height           The height of the underlying image
@param histogram        pointer onto histogram array
----------------------------------------------------------------------------------------*/
void ImageProcessing::histogramEqualizationGPU(uint8_t *currentImage, uint8_t *outputImage, unsigned int *histogram, int width, int height, bool timing){

	float *probDist = (float*)malloc(256 * (sizeof(float)));
	float *histT = (float*)malloc(256 * (sizeof(float)));
	int *histTr = (int*)malloc(256 * (sizeof(int)));
	int size = width * height;
	
	for (int i = 0; i < 256; i++){
		probDist[i] = 0.f;
		histT[i] = 0;
	}

	for (int i = 0; i < 256; i++){
		probDist[i] = ((float)histogram[i]) / ((float)size);
	}


	for (int i = 0; i < 256; i++){
		for (int j = 0; j <= i; j++){
			histT[i] += probDist[j];
		}
		histT[i] = histT[i] * float(i);
	}

	for (int i = 0; i < 256; i++){
		histTr[i] = floor(histT[i]);
	}

	dim3 blockSize;
	blockSize.x = 16;
	blockSize.y = 16;

	dim3 gridSize;
	gridSize.x = ceil(width / blockSize.x);
	gridSize.y = ceil(height / blockSize.y);

	unsigned int *histTrafo;
	gpuError(cudaMalloc((void **)&histTrafo, 256 * sizeof(int)));

	gpuError(cudaMemcpy(histTrafo, histTr, 256 * sizeof(int), cudaMemcpyHostToDevice));

        cudaEvent_t start,stop;
        if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
        }

	histogramEqualization <<<gridSize, blockSize>>>(currentImage, outputImage, histTrafo, width, height);
	
	if(timing){
                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
                cout << "Performed Histogram Equalization in Image of Size " << width << " x " << height << " in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

	free(probDist);
	free(histT);
	free(histTr);
}


/*----------------------------------------------------------------------------------------
 MORPHOLOGICAL OPERATIONS
 ----------------------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------------------
DILATION

Following function performs dilation

@param currentImage     input image
@param outputImage      output image
@param width            The width of the underlying image
@param height           The height of the underlying image
@param structuringElement (By now: disk1, disk4, disk5, disk6, disk15)
----------------------------------------------------------------------------------------*/
void ImageProcessing::dilateGPU(uint8_t *currentImage, uint8_t *outputImage, int structure, int width, int height, bool timing){
	
        cudaEvent_t start,stop;
        if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
        }
	
	int radius = structure - 1;
	int sharedMemAllocRow = (radius + 128 + radius) * sizeof(uint8_t);
	int sharedMemAllocCol = 16 * (radius + 48 + radius) * sizeof(uint8_t);

	dim3 blockGridRows(iDivUp(width, 128), height);
	dim3 blockGridColumns(iDivUp(width, 16), iDivUp(height, 48));
	dim3 threadBlockRows(16 + 128 + radius);
	dim3 threadBlockColumns(16, 8);
	
	int disk[radius * radius];

	fill_n(disk, radius*radius, 1);

 	gpuError(cudaMemcpyToSymbol(kernelStore, disk, sizeof(disk)));

	gpuError(cudaThreadSynchronize());
 	dilationRowGPU <<<blockGridRows, threadBlockRows, sharedMemAllocRow>>>(currentImage, currentImage, width, height, radius);
 	dilationColumnGPU <<<blockGridColumns, threadBlockColumns, sharedMemAllocCol>>>(currentImage, outputImage, width, height, radius, 16 * threadBlockColumns.y, width * threadBlockColumns.y);
 	gpuError(cudaThreadSynchronize());
	

        if(timing){
                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
                cout << "Dilated Image of Size " << width << " x " << height << " with Structuring Element Square" << structure << " in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

}

/*----------------------------------------------------------------------------------------
EROSION
 
Following function performs erosion

@param currentImage     input image
@param outputImage      output image
@param width            The width of the underlying image
@param height           The height of the underlying image
@param structuringElement (By now: disk1, disk4, disk5, disk6, disk15)
----------------------------------------------------------------------------------------*/
void ImageProcessing::erodeGPU(uint8_t *currentImage, uint8_t *outputImage, int structure, int width, int height, bool timing){

	// SINCE SHARED MEMORY SIZE IS VARIOUS, SHARED MEMORY ALLOCATION HAS TO BE DONE IN KERNEL CALL (THIRD ARGUMENT INSIDE BREACKETS)
	// SE IS LOADED TO CONSTANT MEMORY, S.T. FAST MEMORY ACCESS IS GUARENTEED

        cudaEvent_t start,stop;
        if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
        }

	int radius = structure - 1;
        int sharedMemAllocRow = (radius + 128 + radius) * sizeof(uint8_t);
        int sharedMemAllocCol = 16 * (radius + 48 + radius) * sizeof(uint8_t);

        dim3 blockGridRows(iDivUp(width, 128), height);
        dim3 blockGridColumns(iDivUp(width, 16), iDivUp(height, 48));
        dim3 threadBlockRows(16 + 128 + radius);
        dim3 threadBlockColumns(16, 8);

        int disk[radius * radius];

        fill_n(disk, radius*radius, 1);

        gpuError(cudaMemcpyToSymbol(kernelStore, disk, sizeof(disk)));

        gpuError(cudaThreadSynchronize());
        erosionRowGPU <<<blockGridRows, threadBlockRows, sharedMemAllocRow>>>(currentImage, currentImage, width, height, radius);
        erosionColumnGPU <<<blockGridColumns, threadBlockColumns, sharedMemAllocCol>>>(currentImage, outputImage, width, height, radius, 16 * threadBlockColumns.y, width * threadBlockColumns.y);
        gpuError(cudaThreadSynchronize());


        if(timing){

                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
                cout << "Dilated Image of Size " << width << " x " << height << " with Structuring Element Square" << structure << " in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

}




/*----------------------------------------------------------------------------------------
KERNEL FILTER
 ----------------------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------------------
GAUSSIAN BLUR
 
Following function performs gaussian Filter

@param currentImage     input image
@param outputImage      output image
@param width            The width of the underlying image
@param height           The height of the underlying image
@param radius           The radius of the gaussian Kernel
----------------------------------------------------------------------------------------*/
void ImageProcessing::gaussianBlurGPU(uint8_t *currentImage, uint8_t *outputImage, int width, int height, int radius, bool timing){
	
	// SINCE SHARED MEMORY SIZE IS VARIOUS, SHARED MEMORY ALLOCATION HAS TO BE DONE IN KERNEL CALL (THIRD ARGUMENT INSIDE BREACKETS)

	float *convKernel;
	float kernelSum = 0;

	int kernelSize = radius * 2 + 1;
	int sharedMemAllocRow = radius + 128 + radius * sizeof(uint8_t);
	int sharedMemAllocCol = 16 * (radius + 48 + radius) * sizeof(uint8_t);

	convKernel = (float *)malloc(kernelSize * sizeof(float));

	for (int i = 0; i < kernelSize; i++){
		float dist = (float)(i - radius) / (float)radius;
		convKernel[i] = expf(-dist * dist / 2);
		kernelSum += convKernel[i];
	}
	for (int i = 0; i < kernelSize; i++){
		convKernel[i] /= kernelSum;
	}
	
	// KERNEL IS LOADED TO CONSTANT MEMORY
	gpuError(cudaMemcpyToSymbol(convolutionKernel, convKernel, kernelSize * sizeof(float)));

	dim3 blockGridRows(iDivUp(width, 128), height);
	dim3 blockGridColumns(iDivUp(width, 16), iDivUp(height, 48));
	dim3 threadBlockRows(16 + 128 + radius);
	dim3 threadBlockColumns(16, 8);

	gpuError(cudaThreadSynchronize());

        cudaEvent_t start,stop;
        if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
        }


	convolutionRowGPU <<<blockGridRows, threadBlockRows, sharedMemAllocRow>>>(currentImage, outputImage, width, height, radius);
	convolutionColumnGPU <<<blockGridColumns, threadBlockColumns, sharedMemAllocCol>>>(outputImage, outputImage, width, height, radius, 16 * threadBlockColumns.y, width * threadBlockColumns.y);
	
        if(timing){
                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
                cout << "Gaussian Blur applied on Image of Size " << width << " x " << height << " in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

//gpuError(cudaThreadSynchronize());
	//gpuError(cudaFree(convolutionKernel));
}


/*----------------------------------------------------------------------------------------
REDUCTION OPERATIONS
 ----------------------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------------------
MAXIMAL ELEMENT
 
Following function computed the maximal value and position inside an image

@param input image
@param pointer onto maximal value
@param pointer onto corresponding x-Coordinate --- TO BE IMPLEMENTED
@param pointer onto corresponding y-Coordinate --- TO BE IMPLEMENTED
@param width The width of the underlying image
@param height The height of the underlying image
----------------------------------------------------------------------------------------*/
void ImageProcessing::maximalElementGPU(uint8_t *currentImage, int &val, int &x, int &y, int width, int height, bool timing){
	
	dim3 blockSize;
	blockSize.x = 256;

	dim3 gridSize;
	gridSize.x = ceil(width * height / blockSize.x);
	
	int *h_coord =new int[gridSize.x];
	int *h_value = new int[gridSize.x];
	
	int *value;
	int *location;

	int smemSize = blockSize.x * sizeof(int);

	gpuError(cudaMalloc((void **)&value, gridSize.x * sizeof(int)));
	gpuError(cudaMalloc((void **)&location, gridSize.x * sizeof(int)));
	
	int threads = 256;

        cudaEvent_t start,stop;
        if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
        }

	switch (threads){
	case 512:
		maxElement<512> <<<gridSize, blockSize, smemSize>>>(currentImage, value, location, width, height);
	break;
	case 256:
		maxElement<256> <<<gridSize, blockSize, smemSize>>>(currentImage, value, location, width, height);
	break;
	case 128:
		maxElement<128> <<<gridSize, blockSize, smemSize>>>(currentImage, value, location, width, height);
	break;
	case 64:
		maxElement<64> <<<gridSize, blockSize, smemSize>>>(currentImage, value, location, width, height);
	break;
	case 32:
		maxElement<32> <<<gridSize, blockSize, smemSize>>>(currentImage, value, location, width, height);
	break;
	case 16:
		maxElement<16> <<<gridSize, blockSize, smemSize>>>(currentImage, value, location, width, height);
	break;
	case 8:
		maxElement<8> <<<gridSize, blockSize, smemSize>>>(currentImage, value, location, width, height);
	break;
	case 4:
		maxElement<4> <<<gridSize, blockSize, smemSize>>>(currentImage, value, location, width, height);
	break;
	case 2:
		maxElement<2> <<<gridSize, blockSize, smemSize>>>(currentImage, value, location, width, height);
	break;
	case 1:
		maxElement<1> <<<gridSize, blockSize, smemSize>>>(currentImage, value, location, width, height);
	break;
	}

	if(timing){
                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
                cout << "Determine maximal Element in Image of Size " << width << " x " << height << " in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

	gpuError(cudaMemcpy(h_value, value, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));
	gpuError(cudaMemcpy(h_coord, location, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));
	
	cout << "Direct from GPU: " << h_coord[0] << endl;

	val = 0;
	int temp = 0;	
	int h_location;
	for(int i = 0; i < gridSize.x; i++){

		val = max(h_value[i], val);
		if(temp != val){
			h_location = h_coord[i];
		}
	}
	

	cout << "Location ist at " << h_location << endl;

	x = h_location % width;
	y = (int)floor((float)h_location / (float)width);
}




void ImageProcessing::meanVaueGPU(uint8_t *currentImage, int &val, int width, int height, bool timing){

        dim3 blockSize;
        blockSize.x = 256;

        dim3 gridSize;
        gridSize.x = ceil(width * height / blockSize.x);

        double *h_value = new double[gridSize.x];
        double *value;

        int smemSize = blockSize.x * sizeof(double);

        gpuError(cudaMalloc((void **)&value, gridSize.x * sizeof(int)));

        int threads = 256;

        cudaEvent_t start,stop;
        if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
        }

        switch (threads){
        // Following 10 cases can occure (Maximal thread size of 512 and it has to be power of 2)
        case 512:
                meanValue<512> <<<gridSize, blockSize, smemSize>>>(currentImage, value, width,  height);
        break;
        case 256:
                meanValue<256> <<<gridSize, blockSize, smemSize>>>(currentImage, value, width, height);
        break;
        case 128:
                meanValue<128> <<<gridSize, blockSize, smemSize>>>(currentImage, value, width, height);
        break;
        case 64:
                meanValue<64> <<<gridSize, blockSize, smemSize>>>(currentImage, value,  width, height);
        break;
        case 32:
                meanValue<32> <<<gridSize, blockSize, smemSize>>>(currentImage, value,  width, height);
        break;
        case 16:
                meanValue<16> <<<gridSize, blockSize, smemSize>>>(currentImage, value, width, height);
        break;
        case 8:
                meanValue<8> <<<gridSize, blockSize, smemSize>>>(currentImage, value, width, height);
        break;
        case 4:
                meanValue<4> <<<gridSize, blockSize, smemSize>>>(currentImage, value, width, height);
        break;
        case 2:
                meanValue<2> <<<gridSize, blockSize, smemSize>>>(currentImage, value, width, height);
        break;
        case 1:
                meanValue<1> <<<gridSize, blockSize, smemSize>>>(currentImage, value, width, height);
	break;
        }

	if(timing){
                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
                cout << "Determine mean in Image of Size " << width << " x " << height << " in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        gpuError(cudaMemcpy(h_value, value, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));

        val = 0;
	double temp;
        for(int i = 0; i < gridSize.x; i++){

                temp += h_value[i] / gridSize.x;

        }

	val = ceil(temp);

}




/*----------------------------------------------------------------------------------------
GET AREA
 
Following function sums the number of white pixels withing an image

@param currentImage     input image
@param area             pointer onto the number of white pixels
@param width            The width of the underlying image
@param height           The height of the underlying image
----------------------------------------------------------------------------------------*/
void ImageProcessing::getAreaGPU(uint8_t *currentImage, double &area, int width, int height, bool timing){

	dim3 blockSize;
	blockSize.x = 256;

	dim3 gridSize;
	gridSize.x = ceil((width*height) / blockSize.x);

	int smemSize = blockSize.x * sizeof(int);

	int *area1 = new int[gridSize.x];
	int *whitePixels;
	gpuError(cudaMalloc((void **)&whitePixels, gridSize.x * sizeof(int)));

	int threads = 256;

        cudaEvent_t start,stop;
        if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
        }

	switch (threads){
	case 512:
		getArea<512><<<gridSize, blockSize, smemSize>>>(currentImage, whitePixels, width, height);
	break;
	case 256:
		getArea<256><<<gridSize, blockSize, smemSize>>>(currentImage, whitePixels, width, height);
	break;
	case 128:
		getArea<128><<<gridSize, blockSize, smemSize>>>(currentImage, whitePixels, width, height);
	break;
	case 64:
		getArea<64><<<gridSize, blockSize, smemSize>>>(currentImage, whitePixels, width, height);
	break;
	case 32:
		getArea<32><<<gridSize, blockSize, smemSize>>>(currentImage, whitePixels, width, height);
	break;
	case 16:
		getArea<16><<<gridSize, blockSize, smemSize>>>(currentImage, whitePixels, width, height);
	break;
	case 8:
		getArea<8><<<gridSize, blockSize, smemSize>>>(currentImage, whitePixels, width, height);
	break;
	case 4:
		getArea<4><<<gridSize, blockSize, smemSize>>>(currentImage, whitePixels, width, height);
	break;
	case 2:
		getArea<2><<<gridSize, blockSize, smemSize>>>(currentImage, whitePixels, width, height);
	break;
	case 1:
		getArea<1><<<gridSize, blockSize, smemSize>>>(currentImage, whitePixels, width, height);
	break;
	}

	if(timing){
                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
                cout << "Get white Area in Image of Size " << width << " x " << height << " in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

	gpuError(cudaThreadSynchronize());

	cudaMemcpy(area1, whitePixels, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost);

	gpuError(cudaFree(whitePixels));
	
	area = 0;

	for(int i = 0; i < gridSize.x; i++){
		
		area += (double) area1[i];

	}

	delete area1;

}


/*----------------------------------------------------------------------------------------
NAIV IMAGE PROCESSING TECHNIQUES
----------------------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------------------
INVERT
 
Following function inverts an image

@param currentImage     input image
@param outputImage      output image
@param width            The width of the underlying image
@param height           The height of the underlying image
----------------------------------------------------------------------------------------*/
void ImageProcessing::invertGPU(uint8_t *currentImage, uint8_t *outputImage, int width, int height, bool timing){

	dim3 blockSize;
	blockSize.x = 16;
	blockSize.y = 16;

	dim3 gridSize;
	gridSize.x = ceil(width / blockSize.x);
	gridSize.y = ceil(height / blockSize.y);
	
	cudaEvent_t start,stop;
	if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);		
		cudaEventRecord(start,0);
	}

	invert_d<<<gridSize, blockSize>>>(currentImage, outputImage, width, height);
	
	if(timing){
		float memsettime;
		cudaEventRecord(stop,0);
		cudaThreadSynchronize();
		cudaEventElapsedTime(&memsettime, start, stop);
		cout << "Invert Image of Size " << width << " x " << height << " in " << memsettime << " milliseconds" << endl;
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}

/*----------------------------------------------------------------------------------------
BLACK AND WHITE
 
Following function makes an image black white

@param currentImage     input image
@param outputImage      output image
@param width            The width of the underlying image
@param height           The height of the underlying image
@param thresholdValue   Determines which color the processed pixel becomes
----------------------------------------------------------------------------------------*/
void ImageProcessing::blackwhiteGPU(uint8_t *currentImage, uint8_t *outputImage, int thresholdValue, int width, int height, bool timing){

	dim3 blockSize;
	blockSize.x = 16;
	blockSize.y = 16;

	dim3 gridSize;
	gridSize.x = ceil(width / blockSize.x);
	gridSize.y = ceil(height / blockSize.y);

        cudaEvent_t start,stop;
        if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
        }

	threshold <<<gridSize, blockSize>>>(currentImage, outputImage, width, height, thresholdValue);
	
	if(timing){
                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
                cout << "Turn Image of Size " << width << " x " << height << " to blackwhite in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

}

/*----------------------------------------------------------------------------------------
SUBTRACT
 
Following function subtracts input image 2 from input image 1

@param inImage1         first input image
@param inImage2         second input image
@param outputImage      output image
@param width            The width of the underlying image
@param height           The height of the underlying image
----------------------------------------------------------------------------------------*/
void ImageProcessing::subtrachtImagesGPU(uint8_t *inImage1, uint8_t *inImage2, uint8_t *outputImage, int width, int height, bool timing){

	dim3 blockSize;
	blockSize.x = 16;
	blockSize.y = 16;

	dim3 gridSize;
	gridSize.x = ceil(width / blockSize.x);
	gridSize.y = ceil(height / blockSize.y);

        cudaEvent_t start,stop;
        if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
        }

	subtractImages_d <<<gridSize, blockSize>>>(inImage1, inImage2, outputImage, width, height);
	
	if(timing){
                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
                cout << "Subtracted Images of Size " << width << " x " << height << " in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

}

/*----------------------------------------------------------------------------------------
IMADJUST
 
Following function intensifies bright values while softens dark values

@param currentImage     input image
@param outputImage      output image
@param width            The width of the underlying image
@param height           The height of the underlying image
@param limit            Determines the threshold, when a pixel becomes brighter resp. darker
----------------------------------------------------------------------------------------*/
void ImageProcessing::imadjustGPU(uint8_t *currentImage, uint8_t *outputImage, int width, int height, int lowerLimit, int upperLimit, double lowerScale, double upperScale, bool timing){

	dim3 blockSize;
	blockSize.x = 16;
	blockSize.y = 16;

	dim3 gridSize;
	gridSize.x = ceil(width / blockSize.x);
	gridSize.y = ceil(height / blockSize.y);

        cudaEvent_t start,stop;
        if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
        }

	imadjust_d <<<gridSize, blockSize>>>(currentImage, outputImage, width, height, lowerLimit, upperLimit, lowerScale, upperScale);

	if(timing){
                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
                cout << "Adjusted Image of Size " << width << " x " << height << " in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

}

/*----------------------------------------------------------------------------------------
REMOVE AREA
 
Following function blackens the area around given center coordinate within a certain radius specified by the user

@param currentImage     input image
@param outputImage      output image
@param width            The width of the underlying image
@param height           The height of the underlying image
@param x                x-Coordinate of the center pixel
@param y                y-Coordinate of the center pixel
@param radius           radius
----------------------------------------------------------------------------------------*/
void ImageProcessing::removeAreaGPU(uint8_t *currentImage, uint8_t *outputImage, int width, int height, int x, int y, int radius, bool timing){

	dim3 blockSize;
	blockSize.x = 16;
	blockSize.y = 16;

	dim3 gridSize;
	gridSize.x = ceil(width / blockSize.x);
	gridSize.y = ceil(height / blockSize.y);
        
	cudaEvent_t start,stop;
        if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
        }

	removeArea <<<gridSize, blockSize>>>(currentImage, outputImage, width, height, x, y, radius);

        if(timing){
                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
                cout << "Removed Area (Radius " << radius <<  ") in Image of Size " << width << " x " << height << " in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

}




/*----------------------------------------------------------------------------------------
LINEAR TRAFO
 
Following function applies pixelwise linear transformation according to ax+b

@param currentImage     input image
@param outputImage      output image
@param width            The width of the underlying image
@param height           The height of the underlying image
@param alpha            multiplicator
@param beta             bias
----------------------------------------------------------------------------------------*/
void ImageProcessing::linearTransformationGPU(uint8_t *currentImage, uint8_t *outputImage, int width, int height, float alpha, int beta, bool timing){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;

        dim3 gridSize;
        gridSize.x = ceil(width / blockSize.x);
        gridSize.y = ceil(height / blockSize.y);

        cudaEvent_t start,stop;
        if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
        }

        linearTrafo <<<gridSize, blockSize>>>(currentImage, outputImage, width, height, alpha, beta);

        if(timing){
                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
                cout << "Linear Transformation accomplished in Image of Size " << width << " x " << height << " in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

}


/*----------------------------------------------------------------------------------------
 CO-OCCURENCE
 ----------------------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------------------
TEXTURE
 
Following function determines co-occurence matrix and extracts the feature texture
The Procedure is similar to Histogram Computation
 
@param currentImage    input image
@param texture         output feature
@param width           The width of the underlying image
@param height          The height of the underlying image
 ----------------------------------------------------------------------------------------*/
void ImageProcessing::textureGPU(uint8_t *currentImage, double &texture, int width, int height, bool timing){
    
    double *d_texture;
    
    dim3 blockSize;
    blockSize.x = 256;
    
    dim3 gridSize;
    gridSize.x = ceil((width*height) / blockSize.x);
    
	double *h_texture = new double[gridSize.x];

    gpuError(cudaMalloc((void **)&d_texture, gridSize.x * sizeof(double)));
    
    gpuError(cudaThreadSynchronize());

    gpuError(cudaMemset(d_texture, 0, gridSize.x * sizeof(double)));
   
    int smemSize = blockSize.x * sizeof(double);
    
    int threads = 256;

    cudaEvent_t start,stop;
    if(timing){
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start,0);
    }

    switch (threads){
        case 512:
            textureDetermination<512><<<gridSize, blockSize, smemSize>>>(currentImage, d_texture, width, height);
        	break;
	case 256:
            textureDetermination<256><<<gridSize, blockSize, smemSize>>>(currentImage, d_texture, width, height);
        break;
	case 128:
            textureDetermination<128><<<gridSize, blockSize, smemSize>>>(currentImage, d_texture, width, height);
        break;
	case 64:
            textureDetermination<64><<<gridSize, blockSize, smemSize>>>(currentImage, d_texture, width, height);
        break;
	case 32:
            textureDetermination<32><<<gridSize, blockSize, smemSize>>>(currentImage, d_texture, width, height);
        break;
	case 16:
            textureDetermination<16><<<gridSize, blockSize, smemSize>>>(currentImage, d_texture, width, height);
        break;
	case 8:
            textureDetermination<8><<<gridSize, blockSize, smemSize>>>(currentImage, d_texture, width, height);
        break;
	case 4:
            textureDetermination<4><<<gridSize, blockSize, smemSize>>>(currentImage, d_texture, width, height);
        break;
	case 2:
            textureDetermination<2><<<gridSize, blockSize, smemSize>>>(currentImage, d_texture, width, height);
            break;
	case 1:
            textureDetermination<1><<<gridSize, blockSize, smemSize>>>(currentImage, d_texture, width, height);
	    break;    
	}
    
    
    	if(timing){
                float memsettime;
                cudaEventRecord(stop,0);
                cudaThreadSynchronize();
                cudaEventElapsedTime(&memsettime, start, stop);
                cout << "Compute Contrast in Image of Size " << width << " x " << height << " in " << memsettime << " milliseconds" << endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

	cudaThreadSynchronize();
    
    	gpuError(cudaMemcpy(h_texture, d_texture, gridSize.x * sizeof(double), cudaMemcpyDeviceToHost));
    
    	gpuError(cudaThreadSynchronize());
    	
	texture = 0.0;

	for(int i = 0; i < gridSize.x; i++){

		texture += h_texture[i];

	}
    
    	gpuError(cudaFree(d_texture));
	delete h_texture;
}
