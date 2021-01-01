#include "conv_cpu.h"

void conv_h_cpu(image_cpu &dst, const image_cpu &src, const filterkernel_cpu &kernel) {
	int w = src.width, h = src.height;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			float rr = 0.0f, gg = 0.0f, bb = 0.0f;

			for (int i = 0; i < kernel.ks; i++) {
				int xx = x + (i - kernel.ks / 2);

				// Clamp to [0, w-1]
				xx = std::max(std::min(xx, w - 1), 0);

				unsigned int pixel = src.data[y * w + xx];

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

			dst.data[y * w + x] = rr_c | (gg_c << 8) | (bb_c << 16);
		}
	}
}

void conv_v_cpu(image_cpu &dst, const image_cpu &src, const filterkernel_cpu &kernel) {
	int w = src.width, h = src.height;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			float rr = 0.0f, gg = 0.0f, bb = 0.0f;

			for (int i = 0; i < kernel.ks; i++) {
				int yy = y + (i - kernel.ks / 2);

				// Clamp to [0, w-1]
				yy = std::max(std::min(yy, h - 1), 0);

				unsigned int pixel = src.data[yy * w + x];

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

			dst.data[y * w + x] = rr_c | (gg_c << 8) | (bb_c << 16);
		}
	}
}
