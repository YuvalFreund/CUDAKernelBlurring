#pragma once

#include "image.h"
#include "common.h"

void conv_h_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);

void conv_h_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);

void conv_h_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);

void conv_h_gpu_tmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_tmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);

void conv_h_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
