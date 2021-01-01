#include "filtering.h"
#include "image.h"
#include "common.h"
#include "conv_cpu.h"
#include "conv_gpu.h"
#include <chrono>

void filtering(const char *imgfile, int ks)
{
	// === Task 1 ===
	image_cpu* CPUsrc = new image_cpu(imgfile);
	image_cpu* CPUdst = new image_cpu(CPUsrc->width, CPUsrc->height);
	filterkernel_cpu* CPUkernel = new filterkernel_cpu(ks); 
	std::chrono::high_resolution_clock::time_point beginh = std::chrono::high_resolution_clock::now();
	conv_h_cpu(*CPUdst,*CPUsrc,*CPUkernel);
	std::chrono::high_resolution_clock::time_point endh = std::chrono::high_resolution_clock::now();
	printf("CPU horizon processing took: %f ms\n",std::chrono::duration<double, std::milli>(beginh - endh).count());
	std::chrono::high_resolution_clock::time_point beginv = std::chrono::high_resolution_clock::now();
	conv_v_cpu(*CPUdst,*CPUdst,*CPUkernel);
	std::chrono::high_resolution_clock::time_point endv = std::chrono::high_resolution_clock::now();
	printf("CPU vertical processing took: %f ms\n",std::chrono::duration<double, std::milli>(beginv - endv).count());
	CPUdst->save("out_cpu.ppm");
	
	// === Task 2 ===
	image_gpu* GPUsrcGlobal = new image_gpu(CPUsrc->width,CPUsrc->height);
	image_gpu* GPUdstGlobal = new image_gpu(CPUsrc->width,CPUsrc->height);
	filterkernel_gpu *GPUkernelGlobal = new filterkernel_gpu(ks);
	CPUkernel->upload(*GPUkernelGlobal);
	CPUsrc->upload(*GPUsrcGlobal);
	conv_h_gpu_gmem(*GPUdstGlobal,*GPUsrcGlobal,*GPUkernelGlobal);
	conv_v_gpu_gmem(*GPUdstGlobal,*GPUdstGlobal,*GPUkernelGlobal);
	CPUdst->download(*GPUdstGlobal);
	CPUdst->save("out_gpu_gmem.ppm");
	
	// === Task 3 ===
	// TODO: Blur image on GPU (Shared memory)
	image_gpu* GPUsrcShared = new image_gpu(CPUsrc->width,CPUsrc->height);
	image_gpu* GPUdstShared = new image_gpu(CPUsrc->width,CPUsrc->height);
	filterkernel_gpu *GPUkernelShared = new filterkernel_gpu(ks);
	CPUsrc->upload(*GPUsrcShared);
	CPUkernel->upload(*GPUkernelShared);
	conv_h_gpu_smem(*GPUdstShared,*GPUsrcShared,*GPUkernelShared);
	conv_v_gpu_smem(*GPUdstShared,*GPUdstShared,*GPUkernelShared);
	CPUdst->download(*GPUdstShared);
	CPUdst->save("out_gpu_smem.ppm");

	// === Task 4 ===
	// TODO: Blur image on GPU (Constant memory)
	image_gpu* GPUsrcConstant = new image_gpu(CPUsrc->width,CPUsrc->height);
	image_gpu* GPUdstConstant = new image_gpu(CPUsrc->width,CPUsrc->height);
	filterkernel_gpu *GPUkernelConstant = new filterkernel_gpu(ks);
	CPUsrc->upload(*GPUsrcConstant);
	CPUkernel->upload(*GPUkernelConstant);
	conv_h_gpu_cmem(*GPUdstConstant,*GPUsrcConstant,*GPUkernelConstant); //will turn to constant in the call
	conv_v_gpu_cmem(*GPUdstConstant,*GPUdstConstant,*GPUkernelConstant); //will turn to constant in the call
	CPUdst->download(*GPUdstConstant);
	CPUdst->save("out_gpu_cmem.ppm");

	// === Task 5 ===

	// TODO: Blur image on GPU (L1/texture cache)
	image_gpu* GPUsrcTexture = new image_gpu(CPUsrc->width,CPUsrc->height);
	image_gpu* GPUdstTexture = new image_gpu(CPUsrc->width,CPUsrc->height);
	filterkernel_gpu *GPUkernelTexture = new filterkernel_gpu(ks);
	CPUsrc->upload(*GPUsrcTexture);
	CPUkernel->upload(*GPUkernelTexture);
	conv_h_gpu_tmem(*GPUdstTexture,*GPUsrcTexture,*GPUkernelTexture);
	conv_v_gpu_tmem(*GPUdstTexture,*GPUdstTexture,*GPUkernelTexture);
	CPUdst->download(*GPUdstTexture);
	CPUdst->save("out_gpu_tmem.ppm");

	// === Task 6 ===
	// TODO: Blur image on GPU (all memory types)
	image_gpu* GPUsrcAll = new image_gpu(CPUsrc->width,CPUsrc->height);
	image_gpu* GPUdstAll = new image_gpu(CPUsrc->width,CPUsrc->height);
	filterkernel_gpu *GPUkernelAll = new filterkernel_gpu(ks);
	CPUsrc->upload(*GPUsrcAll);
	CPUkernel->upload(*GPUkernelAll);
	conv_h_gpu_all(*GPUdstAll,*GPUsrcAll,*GPUkernelAll);
	conv_v_gpu_all(*GPUdstAll,*GPUdstAll,*GPUkernelAll);
	CPUdst->download(*GPUdstAll);
	CPUdst->save("out_gpu_all.ppm");

}


/************************************************************
 * 
 * TODO: Write your text answers here!
 * 
 * (Task 7) nvprof output
 * 
 * 
 * 
 * ==23161== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
671.16ms  1.4400us                    -               -         -         -         -      140B  92.718MB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
671.39ms  3.0102ms                    -               -         -         -         -  16.000MB  5.1906GB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
674.41ms  5.8337ms            (64 64 1)       (32 32 1)        26        0B        0B         -           -           -           -  GeForce GTX 980         1         7  onePixelConvHorizintalGlobal(image_gpu, image_gpu, filterkernel_gpu) [868]
680.31ms  5.5431ms            (64 64 1)       (32 32 1)        26        0B        0B         -           -           -           -  GeForce GTX 980         1         7  onePixelConvVerticalGlobal(image_gpu, image_gpu, filterkernel_gpu) [879]
685.89ms  2.5115ms                    -               -         -         -         -  16.000MB  6.2214GB/s      Device    Pageable  GeForce GTX 980         1         7  [CUDA memcpy DtoH]
801.43ms  4.3814ms                    -               -         -         -         -  16.000MB  3.5662GB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
805.83ms  1.3440us                    -               -         -         -         -      140B  99.341MB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
805.91ms  5.1748ms            (64 64 1)       (32 32 1)        32        0B  17.016KB         -           -           -           -  GeForce GTX 980         1         7  onePixelConvHorizintalShared(image_gpu, image_gpu, filterkernel_gpu) [895]
811.17ms  4.7475ms            (64 64 1)       (32 32 1)        32        0B  17.016KB         -           -           -           -  GeForce GTX 980         1         7  onePixelConvVerticalShared(image_gpu, image_gpu, filterkernel_gpu) [905]
815.98ms  3.4885ms                    -               -         -         -         -  16.000MB  4.4790GB/s      Device    Pageable  GeForce GTX 980         1         7  [CUDA memcpy DtoH]
955.44ms  4.2572ms                    -               -         -         -         -  16.000MB  3.6702GB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
959.71ms  1.3120us                    -               -         -         -         -      140B  101.76MB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
959.73ms  1.6320us                    -               -         -         -         -      140B  81.810MB/s      Device    Pageable  GeForce GTX 980         1         7  [CUDA memcpy DtoH]
959.77ms  1.3120us                    -               -         -         -         -      140B  101.76MB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
959.84ms  1.6147ms            (64 64 1)       (32 32 1)        27        0B        0B         -           -           -           -  GeForce GTX 980         1         7  onePixelConvHorizintalConstant(image_gpu, image_gpu, filterkernel_gpu) [924]
961.53ms  1.6320us                    -               -         -         -         -      140B  81.810MB/s      Device    Pageable  GeForce GTX 980         1         7  [CUDA memcpy DtoH]
961.55ms  1.0240us                    -               -         -         -         -      140B  130.39MB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
961.59ms  1.4819ms            (64 64 1)       (32 32 1)        27        0B        0B         -           -           -           -  GeForce GTX 980         1         7  onePixelConvVerticalConstant(image_gpu, image_gpu, filterkernel_gpu) [936]
963.12ms  3.9029ms                    -               -         -         -         -  16.000MB  4.0034GB/s      Device    Pageable  GeForce GTX 980         1         7  [CUDA memcpy DtoH]
1.08806s  4.2854ms                    -               -         -         -         -  16.000MB  3.6461GB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
1.09238s  1.3120us                    -               -         -         -         -      140B  101.76MB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
1.09246s  1.4139ms            (64 64 1)       (32 32 1)        26        0B        0B         -           -           -           -  GeForce GTX 980         1         7  onePixelConvHorizintalTexture(unsigned int*, unsigned int const *, float const *, int, int, int) [952]
1.09397s  1.5195ms            (64 64 1)       (32 32 1)        27        0B        0B         -           -           -           -  GeForce GTX 980         1         7  onePixelConvVerticalTexture(unsigned int*, unsigned int const *, float const *, int, int, int) [962]
1.09554s  3.9205ms                    -               -         -         -         -  16.000MB  3.9855GB/s      Device    Pageable  GeForce GTX 980         1         7  [CUDA memcpy DtoH]
1.22235s  4.3061ms                    -               -         -         -         -  16.000MB  3.6286GB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
1.22669s  1.3120us                    -               -         -         -         -      140B  101.76MB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
1.22671s  1.6640us                    -               -         -         -         -      140B  80.237MB/s      Device    Pageable  GeForce GTX 980         1         7  [CUDA memcpy DtoH]
1.22674s  1.3440us                    -               -         -         -         -      140B  99.341MB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
1.22682s  1.6969ms            (64 64 1)       (32 32 1)        32        0B  17.016KB         -           -           -           -  GeForce GTX 980         1         7  onePixelConvHorizontalAll(unsigned int*, unsigned int const *, int, int, int) [980]
1.22859s  1.6640us                    -               -         -         -         -      140B  80.237MB/s      Device    Pageable  GeForce GTX 980         1         7  [CUDA memcpy DtoH]
1.22862s  1.0240us                    -               -         -         -         -      140B  130.39MB/s    Pageable      Device  GeForce GTX 980         1         7  [CUDA memcpy HtoD]
1.22866s  1.1514ms            (64 64 1)       (32 32 1)        32        0B  17.016KB         -           -           -           -  GeForce GTX 980         1         7  onePixelConvVerticalAll(unsigned int*, unsigned int const *, int, int, int) [992]
1.22986s  3.8684ms                    -               -         -         -         -  16.000MB  4.0391GB/s      Device    Pageable  GeForce GTX 980         1         7  [CUDA memcpy DtoH]

 ************************************************************/
