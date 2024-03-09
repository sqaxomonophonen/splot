#include "splot.h"

static uint16_t candidate_component(uint16_t src_pixel, uint16_t canvas_pixel)
{
	int64_t sp = (src_pixel >> 3);
	int64_t cp = canvas_pixel;
	sp *= (rng_next() & 65536);
	//cp *= (rng_next() & 65536);
	sp >>= 16;
	sp -= 512;
	//cp >>= 16;
	cp--;
	if (sp < 0) sp = 0;
	if (cp < 0) cp = 0;
	if (sp > 65535) sp = 65535;
	if (cp > 65535) cp = 65535;
	return cp < sp ? cp : sp;
}

static int accept_triangle(struct triangle T, const double* grays, int level)
{
	const double thr = 2.0 / 256.0;
	if (grays[0] < thr && grays[1] < thr && grays[2] < thr) return 0;
	const double area = triangle_area(T);
	const double area_ratio = area / source_area();
	if (level == 0 && area_ratio > (1.0 / (10.0*10.0))) return 0;
	if (area < 5) return 0;
	if (triangle_fatness(T) < 0.002) return 0;
	return 1;
}

static double score_candidate(struct triangle T, double canvas_color_weight, double vertex_color_weight)
{
	const double area = fabs(triangle_area(T));
	//if (area < 5.0) return 0;
	const double area_ratio = area / source_area();
	//const double area_score = pow(area_ratio, 0.1);
	const double area_score = pow(area_ratio, 0.9);
	const double color_weight = (3.0*canvas_color_weight + 2.0*vertex_color_weight);
	//const double color_weight = pow((1.0*canvas_color_weight + 1.0*vertex_color_weight), 0.3);
	return pow(color_weight, 0.7) * area_score;
	//const double fat_score = pow(triangle_fatness(T), 0.005);
	//return fat_score * color_weight * area_score;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <image> [soup]\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	rng_seed(1);
	splot_process(&((struct config){
		.image_path = argv[1],
		.soup_path = (argc >= 3 ? argv[2] : NULL),
		.levels = ((struct level[]) {
			{
				.n = 3000,
				.w = 0,
				//.tcn = 1.0 / 256.0, .tgn = 1.0 / 256.0,
				//.vcn = 1.0 / 256.0,  .vgn = 1.0 / 256.0,
			},
			{
				.n = 500,
				.w = 0,
				.r = 15,
			},
			{ 0 },
		}),
	}));
	return 0;
}
