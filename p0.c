#include "splot.h"

static uint16_t candidate_component(uint16_t src_pixel, uint16_t canvas_pixel)
{
	int p = (int)src_pixel >> 3;
	int cp = (int)canvas_pixel - 1;
	if (cp < 0) cp = 0;
	if (p > cp) p = cp;
	return p;
}

static int accept_triangle(struct triangle T, const double* grays)
{
	const double thr = 1.0 / 512.0;
	if (grays[0] < thr && grays[1] < thr && grays[2] < thr) return 0;
	const double area_ratio = triangle_area(T) / source_area();
	if (area_ratio > 1.0 / (6.0*6.0)) return 0;
	if (triangle_fatness(T) < 0.005) return 0;
	return 1;
}

static double score_candidate(struct triangle T, double canvas_color_weight, double vertex_color_weight)
{
	const double area = fabs(triangle_area(T));
	if (area < 10.0) return 0;
	const double area_ratio = area / source_area();
	const double area_score = pow(area_ratio, 0.01);
	const double color_weight = g.level_index > 0 ? 1.0 : (1.0*canvas_color_weight + 4.0*vertex_color_weight);
	const double fat_score = g.level_index > 0 ? 1.0 : powf(triangle_fatness(T), 0.5);
	return fat_score * color_weight * area_score;
}

int main(int argc, char** argv)
{
	rng_seed(1);
	splot_process(argv[1], &((struct config){
		.levels = ((struct level[]) {
			{ .n = 2000 , .w = 200 },
			{ .n = 200  , .w = 500 , .r = 30 , .gn = 16.0 / 256.0, .cn = 4.0 / 256.0 } ,
			{ .n = 20  , .w = 0    , .r = 2  , .gn = 1.0 / 256.0,  .cn = 1.0 / 256.0 },
			{ 0 },
		}),
	}));
	return 0;
}
