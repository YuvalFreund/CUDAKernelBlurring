#include "conv_gpu.h"
#define BLOCK_SIZE 32
#include <stdio.h>
//constant kenrel for cmem_conv
__constant__ float constKernel[127];

//declare of kernel calls

//kernel calls for global memory
__global__ void onePixelConvVerticalGlobal(image_gpu dst, const image_gpu src, const filterkernel_gpu kernel);
__global__ void onePixelConvHorizintalGlobal(image_gpu dst, const image_gpu src, const filterkernel_gpu kernel);
//kernel calls for shared memory
__global__ void onePixelConvVerticalShared(image_gpu dst, const image_gpu src, const filterkernel_gpu kernel);
__global__ void onePixelConvHorizintalShared(image_gpu dst, const image_gpu src, const filterkernel_gpu kernel);
//kernel calls for constant memory
__global__ void onePixelConvVerticalConstant(image_gpu dst, const image_gpu src, const filterkernel_gpu kernel);
__global__ void onePixelConvHorizintalConstant(image_gpu dst, const image_gpu src, const filterkernel_gpu kernel);
//kernel calls for the Texture memory
__global__ void onePixelConvVerticalTexture(unsigned int* __restrict__ dstData, const unsigned int* __restrict__ srcData, const float* __restrict__ kernelData,int width,int height,int ks);
__global__ void onePixelConvHorizintalTexture(unsigned int* __restrict__ dstData, const unsigned int* __restrict__ srcData, const float* __restrict__ kernelData,int width,int height,int ks);
//kernel calls for the all memory use
__global__ void onePixelConvVerticalAll(unsigned int* __restrict__ dstData, const unsigned int* __restrict__ srcData, int width,int height,int ks);
__global__ void onePixelConvHorizontalAll(unsigned int* __restrict__ dstData, const unsigned int* __restrict__ srcData, int width,int height,int ks);
void conv_h_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel)
{
    int grid_width = div_up(src.width,BLOCK_SIZE);
	int grid_height = div_up(src.height,BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(grid_width,grid_height);
    cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
 	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);
    onePixelConvHorizintalGlobal <<<dimGrid, dimBlock>>>( dst ,src, kernel);
	cudaEventRecord(evStop, 0);
 	cudaEventSynchronize(evStop);
 	float elapsedTime_ms;
 	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
 	printf("CUDA horizontal processing from global memory took: %f ms\n", elapsedTime_ms);
 	cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    CUDA_CHECK_ERROR;
}
void conv_v_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel)
{
    int grid_width = div_up(src.width,BLOCK_SIZE);
	int grid_height = div_up(src.height,BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(grid_width,grid_height);
    CUDA_CHECK_ERROR;
    cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
 	cudaEventCreate(&evStop);
    cudaEventRecord(evStart, 0);
    onePixelConvVerticalGlobal<<<dimGrid, dimBlock>>>( dst ,src, kernel);
	cudaEventRecord(evStop, 0);
 	cudaEventSynchronize(evStop);
 	float elapsedTime_ms;
 	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
 	printf("CUDA vertical processing from global memory: %f ms\n", elapsedTime_ms);
 	cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    CUDA_CHECK_ERROR;
}

void conv_h_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel){
    int grid_width = div_up(src.width,BLOCK_SIZE);
    int grid_height = div_up(src.height,BLOCK_SIZE);
    int sharedMemorySize = ((kernel.ks-1) + BLOCK_SIZE)*((kernel.ks-1) + BLOCK_SIZE)*sizeof(float);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(grid_width,grid_height);
    cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
 	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);
    onePixelConvHorizintalShared<<<dimGrid, dimBlock,sharedMemorySize>>>( dst ,src, kernel);
    CUDA_CHECK_ERROR;
	cudaEventRecord(evStop, 0);
 	cudaEventSynchronize(evStop);
 	float elapsedTime_ms;
 	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
 	printf("CUDA horizontal processing from shared memory took: %f ms\n", elapsedTime_ms);
 	cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
}
void conv_v_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel){
    int grid_width = div_up(src.width,BLOCK_SIZE);
    int grid_height = div_up(src.height,BLOCK_SIZE);
    int sharedMemorySize = (kernel.ks-1 + BLOCK_SIZE)*(kernel.ks-1 + BLOCK_SIZE)*sizeof(float);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(grid_width,grid_height);
    cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
 	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);
    onePixelConvVerticalShared <<<dimGrid, dimBlock,sharedMemorySize>>>( dst ,src, kernel);
	cudaEventRecord(evStop, 0);
 	cudaEventSynchronize(evStop);
 	float elapsedTime_ms;
 	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
 	printf("CUDA vertical processing from shared memory took: %f ms\n", elapsedTime_ms);
 	cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    CUDA_CHECK_ERROR;
}

void conv_h_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel){
    int grid_width = div_up(src.width,BLOCK_SIZE);
	int grid_height = div_up(src.height,BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(grid_width,grid_height);
    float* kernelValues = new float[kernel.ks];
    cudaMemcpy(kernelValues, kernel.data, kernel.ks*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(constKernel, kernelValues, kernel.ks*sizeof(float),0,cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR;
    cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
 	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);
    onePixelConvHorizintalConstant<<<dimGrid, dimBlock>>>( dst ,src, kernel);
	cudaEventRecord(evStop, 0);
 	cudaEventSynchronize(evStop);
 	float elapsedTime_ms;
 	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
 	printf("CUDA horizontal processing from constant kernel and global memory took: %f ms\n", elapsedTime_ms);
 	cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    CUDA_CHECK_ERROR;
}
void conv_v_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel){
    int grid_width = div_up(src.width,BLOCK_SIZE);
	int grid_height = div_up(src.height,BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(grid_width,grid_height);
    float* kernelValues = new float[kernel.ks];
    cudaMemcpy(kernelValues, kernel.data, kernel.ks*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(constKernel, kernelValues, kernel.ks*sizeof(float));
    cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
 	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);
    onePixelConvVerticalConstant <<<dimGrid, dimBlock>>>( dst ,src, kernel);
	cudaEventRecord(evStop, 0);
 	cudaEventSynchronize(evStop);
 	float elapsedTime_ms;
 	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
 	printf("CUDA vertical processing from constant kernel and global memory took: %f ms\n", elapsedTime_ms);
 	cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    CUDA_CHECK_ERROR;
}

void conv_h_gpu_tmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel){
    int grid_width = div_up(src.width,BLOCK_SIZE);
	int grid_height = div_up(src.height,BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(grid_width,grid_height);
    cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
 	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);
    onePixelConvHorizintalTexture<<<dimGrid, dimBlock>>>(dst.data ,src.data, kernel.data,src.width,src.height,kernel.ks);
	cudaEventRecord(evStop, 0);
 	cudaEventSynchronize(evStop);
 	float elapsedTime_ms;
 	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
 	printf("CUDA horizontal processing from texture memory took: %f ms\n", elapsedTime_ms);
 	cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    CUDA_CHECK_ERROR;
}
void conv_v_gpu_tmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel){
    int grid_width = div_up(src.width,BLOCK_SIZE);
	int grid_height = div_up(src.height,BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(grid_width,grid_height);
    cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
 	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);
    onePixelConvVerticalTexture <<<dimGrid, dimBlock>>>(dst.data ,src.data, kernel.data,src.width,src.height,kernel.ks);
	cudaEventRecord(evStop, 0);
 	cudaEventSynchronize(evStop);
 	float elapsedTime_ms;
 	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
 	printf("CUDA vertical processing from texture memory took: %f ms\n", elapsedTime_ms);
 	cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    CUDA_CHECK_ERROR;
}

void conv_h_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel){
    int grid_width = div_up(src.width,BLOCK_SIZE);
    int grid_height = div_up(src.height,BLOCK_SIZE);
    int sharedMemorySize = ((kernel.ks-1) + BLOCK_SIZE)*((kernel.ks-1) + BLOCK_SIZE)*sizeof(float);
    float* kernelValues = new float[kernel.ks];
    cudaMemcpy(kernelValues, kernel.data, kernel.ks*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(constKernel, kernelValues, kernel.ks*sizeof(float),0,cudaMemcpyHostToDevice);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(grid_width,grid_height);
    cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
 	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);
    onePixelConvHorizontalAll<<<dimGrid, dimBlock,sharedMemorySize>>>(dst.data ,src.data,src.width,src.height,kernel.ks);
    CUDA_CHECK_ERROR;
	cudaEventRecord(evStop, 0);
 	cudaEventSynchronize(evStop);
 	float elapsedTime_ms;
 	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
 	printf("CUDA horizontal processing from All memory accesses took: %f ms\n", elapsedTime_ms);
 	cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
}
void conv_v_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel){
    int grid_width = div_up(src.width,BLOCK_SIZE);
    int grid_height = div_up(src.height,BLOCK_SIZE);
    int sharedMemorySize = ((kernel.ks-1) + BLOCK_SIZE)*((kernel.ks-1) + BLOCK_SIZE)*sizeof(float);
    float* kernelValues = new float[kernel.ks];
    cudaMemcpy(kernelValues, kernel.data, kernel.ks*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(constKernel, kernelValues, kernel.ks*sizeof(float),0,cudaMemcpyHostToDevice);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(grid_width,grid_height);
    cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
 	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);
    onePixelConvVerticalAll<<<dimGrid, dimBlock,sharedMemorySize>>>(dst.data ,src.data,src.width,src.height,kernel.ks);
    CUDA_CHECK_ERROR;
	cudaEventRecord(evStop, 0);
 	cudaEventSynchronize(evStop);
 	float elapsedTime_ms;
 	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
 	printf("CUDA vertical processing from All memory accesses took: %f ms\n", elapsedTime_ms);
 	cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
}

__global__ void onePixelConvHorizintalGlobal(image_gpu dst, const image_gpu src, const filterkernel_gpu kernel){
	int threadx = threadIdx.x;
	int thready = threadIdx.y;
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	int tx = blockx*BLOCK_SIZE +threadx;
	int ty = blocky*BLOCK_SIZE +thready;
	if(tx>=dst.width || ty>=dst.height) return;
    int w = src.width;
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    for (int i = 0; i < kernel.ks; i++) {
    	int xx = tx + (i - kernel.ks / 2);
        // Clamp to [0, w-1]
        if(xx >= w) xx = w-1;
        if(xx < 0) xx = 0;
    	unsigned int pixel = src.data[ty * w + xx];
    	unsigned char r = pixel & 0xff;
    	unsigned char g = (pixel >> 8) & 0xff;
    	unsigned char b = (pixel >> 16) & 0xff;
    	rr += r * kernel.data[i];
    	gg += g * kernel.data[i];
    	bb += b * kernel.data[i];
    }
    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;
    dst.data[ty * w + tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}
__global__ void onePixelConvVerticalGlobal(image_gpu dst, const image_gpu src, const filterkernel_gpu kernel){
	int threadx = threadIdx.x;
	int thready = threadIdx.y;
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	int tx = blockx*BLOCK_SIZE +threadx;
	int ty = blocky*BLOCK_SIZE +thready;
	if(tx>=dst.width || ty>=dst.height) return;
    int w = src.width;
    int h = src.height;
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    for (int i = 0; i < kernel.ks; i++) {
    	int yy = ty + (i - kernel.ks / 2);
        // Clamp to [0, w-1]
        if(yy>=h) yy = h - 1;
        if(yy<0) yy = 0;
    	unsigned int pixel = src.data[yy * w + tx];
    	unsigned char r = pixel & 0xff;
    	unsigned char g = (pixel >> 8) & 0xff;
    	unsigned char b = (pixel >> 16) & 0xff;
    	rr += r * kernel.data[i];
    	gg += g * kernel.data[i];
    	bb += b * kernel.data[i];
    }
    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;
    dst.data[ty * w + tx] = rr_c | (gg_c << 8) | (bb_c << 16);

}
__global__ void onePixelConvHorizintalShared(image_gpu dst, const image_gpu src, const filterkernel_gpu kernel){
    int apron = kernel.ks/2;
    int w = src.width;
    int h = src.height;
	int threadx = threadIdx.x;
    int thready = threadIdx.y;
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	int tx = blockx*BLOCK_SIZE +threadx;
    int ty = blocky*BLOCK_SIZE +thready;
    int apronx = threadx+apron;
    int aprony = thready+apron;
    extern __shared__ unsigned int sharedSrc[];
    if(tx>=dst.width || ty>=dst.height) return;
    int sharedLineSize = (2*(apron)+BLOCK_SIZE);
    //each thread copies his ownpixel
    sharedSrc[(aprony)*sharedLineSize + (apronx)] = src.data[ty * w + tx];
    if(threadx==0){ //left line
        for(int i=1;i<=apron;i++){
            int xx = tx-i;
            if (xx<0) xx=0; 
            sharedSrc[(aprony)*sharedLineSize + (apronx-i)] = src.data[ty * w + xx];
        }          
    }
    if(threadx == BLOCK_SIZE-1){ //right line
        for(int i=1;i<=apron;i++){
            int xx = tx+i;
            if (xx>w-1) xx = w-1; 
            sharedSrc[(aprony)*sharedLineSize + (apronx+i)] = src.data[ty * w + xx];
        }   
    }
    __syncthreads();
      
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    for (int i = 0; i < kernel.ks; i++) {
    	int xx = apronx + (i - kernel.ks / 2);
        unsigned int pixel = sharedSrc[aprony * sharedLineSize + xx];
    	unsigned char r = pixel & 0xff;
    	unsigned char g = (pixel >> 8) & 0xff;
    	unsigned char b = (pixel >> 16) & 0xff;
    	rr += r *  kernel.data[i];
    	gg += g *  kernel.data[i];
    	bb += b *  kernel.data[i];
    }
    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;
    dst.data[ty * w + tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}
__global__ void onePixelConvVerticalShared(image_gpu dst, const image_gpu src, const filterkernel_gpu kernel){
    extern __shared__  unsigned int sharedSrc[];
    int apron = kernel.ks/2;
    int w = src.width;
    int h = src.height;
	int threadx = threadIdx.x;
    int thready = threadIdx.y;
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	int tx = blockx*BLOCK_SIZE +threadx;
    int ty = blocky*BLOCK_SIZE +thready;
    int apronx = threadx+apron;
    int aprony = thready+apron;
    if(tx>=dst.width || ty>=dst.height) return;
    int sharedLineSize = (2*(apron)+BLOCK_SIZE);

    sharedSrc[(aprony)*sharedLineSize + (apronx)] = src.data[ty * w + tx];
    if(thready==0){//top line
        for(int i=1;i<=apron;i++){
            int yy = ty-i;
            if(yy<0) yy = 0;     
            sharedSrc[(aprony-i)*sharedLineSize + (apronx)] = src.data[yy * w + tx];
        }
    }
    if(thready==BLOCK_SIZE-1){
        for(int i=1;i<=apron;i++){
            int yy = ty+i;
            if(yy>h-1) yy =h-1;     
            sharedSrc[(aprony+i)*sharedLineSize + (apronx)] = src.data[yy * w + tx];
        }
    }
    __syncthreads();
  
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    for (int i = 0; i < kernel.ks; i++) {
        int yy = aprony + (i - kernel.ks / 2);
        unsigned int pixel = sharedSrc[yy * sharedLineSize + apronx];
    	unsigned char r = pixel & 0xff;
    	unsigned char g = (pixel >> 8) & 0xff;
    	unsigned char b = (pixel >> 16) & 0xff;
    	rr += r * kernel.data[i];
    	gg += g * kernel.data[i];
    	bb += b * kernel.data[i];
    }
    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;
    dst.data[ty * w + tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}
__global__ void onePixelConvHorizintalConstant(image_gpu dst, const image_gpu src, const filterkernel_gpu kernel){
    int threadx = threadIdx.x;
	int thready = threadIdx.y;
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	int tx = blockx*BLOCK_SIZE +threadx;
	int ty = blocky*BLOCK_SIZE +thready;
	if(tx>=dst.width || ty>=dst.height) return;
    int w = src.width;
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    for (int i = 0; i < kernel.ks; i++) {
    	int xx = tx + (i - kernel.ks / 2);
        // Clamp to [0, w-1]
        if(xx >= w) xx = w-1;
        if(xx < 0) xx = 0;
    	unsigned int pixel = src.data[ty * w + xx];
    	unsigned char r = pixel & 0xff;
    	unsigned char g = (pixel >> 8) & 0xff;
    	unsigned char b = (pixel >> 16) & 0xff;
    	rr += r * constKernel[i];
    	gg += g * constKernel[i];
    	bb += b * constKernel[i];
    }
    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;
    dst.data[ty * w + tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}
__global__ void onePixelConvVerticalConstant(image_gpu dst, const image_gpu src, const filterkernel_gpu kernel){

	int threadx = threadIdx.x;
	int thready = threadIdx.y;
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	int tx = blockx*BLOCK_SIZE +threadx;
	int ty = blocky*BLOCK_SIZE +thready;
	if(tx>=dst.width || ty>=dst.height) return;
    int w = src.width;
    int h = src.height;
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    for (int i = 0; i < kernel.ks; i++) {
    	int yy = ty + (i - kernel.ks / 2);
        // Clamp to [0, w-1]
        if(yy>=h) yy = h - 1;
        if(yy<0) yy = 0;
    	unsigned int pixel = src.data[yy * w + tx];
    	unsigned char r = pixel & 0xff;
    	unsigned char g = (pixel >> 8) & 0xff;
    	unsigned char b = (pixel >> 16) & 0xff;
    	rr += r * constKernel[i];
    	gg += g * constKernel[i];
    	bb += b * constKernel[i];
    }
    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;
    dst.data[ty * w + tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}
__global__ void onePixelConvHorizintalTexture(unsigned int* __restrict__ dstData, const unsigned int* __restrict__ srcData,const float* __restrict__ kernelData,int width,int height,int ks){
    int threadx = threadIdx.x;
	int thready = threadIdx.y;
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	int tx = blockx*BLOCK_SIZE +threadx;
	int ty = blocky*BLOCK_SIZE +thready;
	if(tx>=width || ty>=height) return;
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    for (int i = 0; i < ks; i++) {
    	int xx = tx + (i - ks / 2);
        // Clamp to [0, w-1]
        if(xx >= width) xx = width-1;
        if(xx < 0) xx = 0;
    	unsigned int pixel = srcData[ty * width + xx];
    	unsigned char r = pixel & 0xff;
    	unsigned char g = (pixel >> 8) & 0xff;
    	unsigned char b = (pixel >> 16) & 0xff;
    	rr += r * kernelData[i];
    	gg += g * kernelData[i];
    	bb += b * kernelData[i];
    }
    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;
    dstData[ty * width + tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}
__global__ void onePixelConvVerticalTexture(unsigned int* __restrict__ dstData, const unsigned int* __restrict__ srcData, const float* __restrict__ kernelData,int width,int height,int ks){
	int threadx = threadIdx.x;
	int thready = threadIdx.y;
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	int tx = blockx*BLOCK_SIZE +threadx;
	int ty = blocky*BLOCK_SIZE +thready;
    if(tx>=width || ty>=height) return;
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    for (int i = 0; i < ks; i++) {
    	int yy = ty + (i - ks / 2);
        // Clamp to [0, w-1]
        if(yy>=height) yy = height - 1;
        if(yy<0) yy = 0;
    	unsigned int pixel = srcData[yy * width + tx];
    	unsigned char r = pixel & 0xff;
    	unsigned char g = (pixel >> 8) & 0xff;
    	unsigned char b = (pixel >> 16) & 0xff;
    	rr += r * kernelData[i];
    	gg += g * kernelData[i];
    	bb += b * kernelData[i];
    }
    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;
    dstData[ty * width + tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}

__global__ void onePixelConvHorizontalAll(unsigned int* __restrict__ dstData, const unsigned int* __restrict__ srcData, int width,int height,int ks){
    int apron = ks/2;
    int w = width;
    int h = height;
	int threadx = threadIdx.x;
    int thready = threadIdx.y;
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	int tx = blockx*BLOCK_SIZE +threadx;
    int ty = blocky*BLOCK_SIZE +thready;
    int apronx = threadx+apron;
    int aprony = thready+apron;
    extern __shared__ unsigned int sharedSrc[];
    if(tx>=width || ty>=height) return;
    int sharedLineSize = (2*(apron)+BLOCK_SIZE);
    //each thread copies his ownpixel
    sharedSrc[(aprony)*sharedLineSize + (apronx)] = srcData[ty * w + tx];
    if(threadx==0){ //left line
        for(int i=1;i<=apron;i++){
            int xx = tx-i;
            if (xx<0) xx=0; 
            sharedSrc[(aprony)*sharedLineSize + (apronx-i)] = srcData[ty * w + xx];
        }          
    }
    if(threadx == BLOCK_SIZE-1){ //right line
        for(int i=1;i<=apron;i++){
            int xx = tx+i;
            if (xx>w-1) xx = w-1; 
            sharedSrc[(aprony)*sharedLineSize + (apronx+i)] = srcData[ty * w + xx];
        }   
    }
    __syncthreads();
      
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    for (int i = 0; i < ks; i++) {
    	int xx = apronx + (i - ks / 2);
        unsigned int pixel = sharedSrc[aprony * sharedLineSize + xx];
    	unsigned char r = pixel & 0xff;
    	unsigned char g = (pixel >> 8) & 0xff;
    	unsigned char b = (pixel >> 16) & 0xff;
    	rr += r *  constKernel[i];
    	gg += g *  constKernel[i];
    	bb += b *  constKernel[i];
    }
    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;
    dstData[ty * w + tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}
__global__ void onePixelConvVerticalAll(unsigned int* __restrict__ dstData, const unsigned int* __restrict__ srcData,int width,int height,int ks){
    extern __shared__  unsigned int sharedSrc[];
    int apron = ks/2;
    int w = width;
    int h = height;
	int threadx = threadIdx.x;
    int thready = threadIdx.y;
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	int tx = blockx*BLOCK_SIZE +threadx;
    int ty = blocky*BLOCK_SIZE +thready;
    int apronx = threadx+apron;
    int aprony = thready+apron;
    if (tx>=width || ty>= height) return;
    int sharedLineSize = (2*(apron)+BLOCK_SIZE);
    sharedSrc[(aprony)*sharedLineSize + (apronx)] = srcData[ty * w + tx];
    if(thready==0){//top line
        for(int i=1;i<=apron;i++){
            int yy = ty-i;
            if(yy<0) yy = 0;     
            sharedSrc[(aprony-i)*sharedLineSize + (apronx)] = srcData[yy * w + tx];
        }
    }
    if(thready==BLOCK_SIZE-1){
        for(int i=1;i<=apron;i++){
            int yy = ty+i;
            if(yy>h-1) yy =h-1;     
            sharedSrc[(aprony+i)*sharedLineSize + (apronx)] = srcData[yy * w + tx];
        }
    }
    __syncthreads();
  
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    for (int i = 0; i < ks; i++) {
        int yy = aprony + (i - ks / 2);
        unsigned int pixel = sharedSrc[yy * sharedLineSize + apronx];
    	unsigned char r = pixel & 0xff;
    	unsigned char g = (pixel >> 8) & 0xff;
    	unsigned char b = (pixel >> 16) & 0xff;
    	rr += r * constKernel[i];
    	gg += g * constKernel[i];
    	bb += b * constKernel[i];
    }
    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;
    dstData[ty * w + tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}