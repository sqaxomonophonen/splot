#include "splot.h"

static uint16_t candidate_component(uint16_t src_pixel, uint16_t canvas_pixel)
{
	int p = (int)src_pixel >> 3;
	//int cp = (int)canvas_pixel >> 2;
	int cp = (int)canvas_pixel >> 2;
	//cp--;
	if (cp < 0) cp = 0;
	assert(p >= 0);
	//assert(cp >= 0);
	return p > cp ? cp : p;
}

static int accept_triangle(struct triangle T, const double* grays)
{
	const double thr = 3.0 / 256.0;
	if (grays[0] < thr && grays[1] < thr && grays[2] < thr) return 0;
	const double area = triangle_area(T);
	const double area_ratio = area / source_area();
	if (area_ratio > (1.0 / (6.0*6.0))) return 0;
	if (area < 5) return 0;
	if (triangle_fatness(T) < 0.004) return 0;
	return 1;
}

static double score_candidate(struct triangle T, double canvas_color_weight, double vertex_color_weight)
{
	const double area = fabs(triangle_area(T));
	if (area < 5.0) return 0;
	const double area_ratio = area / source_area();
	const double area_score = pow(area_ratio, 0.03);
	const double color_weight = (1.0*canvas_color_weight + 4.0*vertex_color_weight);
	const double fat_score = powf(triangle_fatness(T), 0.03);
	return fat_score * color_weight * area_score;
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
			{ .n = 3000 , .w = 200 ,           .cn = 14.0 / 256.0 , .gn = 5.0 / 256.0 , },
			{ .n = 500  , .w = 400 , .r = 50 , .cn =  8.0 / 256.0 , .gn = 4.0 / 256.0 , },
			{ .n = 300  , .w = 500 , .r = 30 , .cn =  1.0 / 256.0 , .gn = 8.0 / 256.0 , } ,
			{ .n = 30   , .w = 0   , .r = 3  , .cn =  1.0 / 256.0 , .gn = 1.0 / 256.0 , },
			{ 0 },
		}),
	}));
	return 0;
}
