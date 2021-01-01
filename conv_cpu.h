#pragma once

#include "image.h"
#include "common.h"

void conv_h_cpu(image_cpu &dst, const image_cpu &src, const filterkernel_cpu &kernel);
void conv_v_cpu(image_cpu &dst, const image_cpu &src, const filterkernel_cpu &kernel);
