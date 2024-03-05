#include "splot.h"

static uint16_t candidate_component(uint16_t src_pixel, uint16_t canvas_pixel)
{
	int p = (int)src_pixel >> 4;
	int cp = (int)canvas_pixel - 1;
	if (cp < 0) cp = 0;
	if (p > cp) p = cp;
	return p;
}

static int accept_triangle(struct triangle T)
{
	const double area_ratio = triangle_area(T) / source_area();
	if (area_ratio > 1.0 / (6.0*6.0)) return 0;
	if (triangle_fatness(T) < 0.005) return 0;
	return 1;
}

static double score_candidate(struct triangle T, double canvas_color_weight, double vertex_color_weight)
{
	const double area = fabs(triangle_area(T));
	if (area < 10.0) return 0;
	const double fat = triangle_fatness(T);
	const double area_ratio = area / source_area();
	const double area_score = pow(area_ratio, 0.01);
	const double color_weight = 1.0*canvas_color_weight + 4.0*vertex_color_weight;
	const double fat_score = powf(fat, 0.5);
	return fat_score * color_weight * area_score;
}

int main(int argc, char** argv)
{
	rng_seed(0);
	splot_process(argv[1], &((struct config){
		.levels = ((struct level[]) {
			{ .n = 1000 , .w = 160 },
			{ .n = 500  , .w = 400 , .r = 10 , .cn = 10.0 / 256.0 } ,
			{ .n = 80  , .w = 0    , .r = 2  , .cn =  2.0 / 256.0 },
			{ 0 },
		}),
	}));
	return 0;
}
