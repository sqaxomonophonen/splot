#define NO_PROCESS
#include "splot.h"

static void ppad(int npad, int nprint)
{
	const int nspace = npad - nprint;
	for (int i = 0; i < nspace; i++) putchar(' ');
}

int main(int argc, char** argv)
{
	load_soup(argv[1], 0);
	const int n_tris = get_n_triangles();
	printf("%d triangles\n", n_tris);
	uint16_t* p = g.chosen_vs;
	for (int i = 0; i < n_tris; i++) {
		printf(" #%.5d ", i);
		float co[6];
		float* pco = co;
		for (int point = 0; point < 3; point++) {
			uint16_t x = *(p++);
			uint16_t y = *(p++);
			*(pco++) = x;
			*(pco++) = y;
			int npr = 0;
			npr += printf(" (%d,%d", (int)x, (int)y);
			for (int c = 0; c < g.source_n_channels; c++) {
				uint16_t c = *(p++);
				npr += printf("/%.4x", (int)c);
			}
			npr += printf(")");
			ppad(28, npr);
		}

		struct triangle T = mk_triangle(
			co[0], co[1],
			co[2], co[3],
			co[4], co[5]);

		double area = triangle_area(T);
		double fat = triangle_fatness(T);
		ppad(10, printf(" A=%.1f", area));
		ppad(15, printf(" sqrt(A)=%.1f", sqrt(area)));
		printf(" F=%.4f", fat);

		printf("\n");
	}
	return 0;
}
