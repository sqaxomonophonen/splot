#ifndef SPLOT_H

struct level {
	int n; // number of candidates
	int w; // resolution; 0=source resolution
	// these must be 0 in first level:
	double r; // search radius
	double cn; // color noise
	double gn; // gray noise
};

struct config {
	struct level* levels;
};

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdarg.h>
#include <time.h>

typedef struct {
	struct timespec ts;
} TIME;

static inline TIME NOW(void)
{
	TIME t;
	assert(0 == clock_gettime(CLOCK_MONOTONIC, &t.ts));
	return t;
}

static inline double DIFF(TIME ta, TIME tb)
{
	const double dt =
		(double)(ta.ts.tv_sec - tb.ts.tv_sec) +
		1e-9 * (double)(ta.ts.tv_nsec - tb.ts.tv_nsec);
	return dt;
}

#include "gl3w.h"
#include "stb_image.h"
#include "stb_ds.h"
#include "xoshiro256plusplus.h"

#include <SDL.h>


struct vec2 { double x,y; };
struct triangle { struct vec2 A,B,C; };

static inline struct vec2 mk_vec2(double x, double y)
{
	return (struct vec2) {.x=x, .y=y};
}

static inline double vec2_dot(struct vec2 a, struct vec2 b)
{
	return a.x*b.x + a.y*b.y;
}

static inline double vec2_length(struct vec2 v)
{
	return sqrt(vec2_dot(v,v));
}

static inline struct vec2 vec2_sub(struct vec2 a, struct vec2 b)
{
	return mk_vec2(a.x-b.x, a.y-b.y);
}

static inline double vec2_distance(struct vec2 a, struct vec2 b)
{
	return vec2_length(vec2_sub(b, a));
}

static inline struct vec2 vec2_scale(struct vec2 v, double scalar)
{
	return mk_vec2(v.x*scalar, v.y*scalar);
}

static inline double triangle_area(struct triangle t)
{
	return 0.5 * (t.A.x*(t.B.y-t.C.y) + t.B.x*(t.C.y-t.A.y) + t.C.x*(t.A.y-t.B.y));
}

static inline struct triangle mk_triangle(double x1, double y1, double x2, double y2, double x3, double y3)
{
	struct triangle t = {
		.A = mk_vec2(x1,y1),
		.B = mk_vec2(x2,y2),
		.C = mk_vec2(x3,y3),
	};
	//printf("%f,%f - %f,%f - %f,%f   area=%f\n", x1,y1, x2,y2, x3,y3, triangle_area(t));
	return t;
}

static inline void triangle_sides(struct triangle t, double* a, double* b, double* c){
	*a = vec2_distance(t.A, t.B);
	*b = vec2_distance(t.B, t.C);
	*c = vec2_distance(t.C, t.A);
}

static inline double triangle_semiperimeter(struct triangle t)
{
	double a,b,c;
	triangle_sides(t, &a, &b, &c);
	return (a+b+c)/2.0;
}

static inline double triangle_inradius(struct triangle t)
{
	return triangle_area(t) / triangle_semiperimeter(t);
}

static inline double triangle_circumradius(struct triangle t)
{
	double a,b,c;
	triangle_sides(t, &a, &b, &c);
	return (a*b*c) / (4.0 * triangle_area(t));
}

static inline double triangle_fatness(struct triangle t)
{
	return 2.0 * (triangle_inradius(t) / triangle_circumradius(t));
}

enum mode {
	MODE_POSITIVE = 0,
	MODE_NEGATIVE,
	MODE_ORIGINAL,
	MODE_DUMMY,
};

struct stat {
	int _init;
	TIME t0;
	int n_lvls;
	int n_lvl0s;
	int n_generated_triangles;
	int n_accepted_triangles;
	int n_trial_runs;
	int n_trials;
	int n_underflows;
	int n_rejected;
	int n_accepted;
};


static struct {
	SDL_Window* window;
	SDL_GLContext glctx;
	int window_width;
	int window_height;

	enum mode mode;
	int save_image;
	int save_minsamps;
	int stretch;
	unsigned frame_counter;
	int max_batch_size_log2;

	GLuint framebuffer;
	GLuint vao;
	GLuint canvas_tex;
	GLuint* minsamp_texs;
	GLuint original_tex;
	GLenum source_format;
	GLuint dummy_fb_tex;
	GLuint signal_buf;
	GLuint paint_vbo;
	GLuint trial_vbo;

	GLuint trial_prg;
	GLint trial_uloc_pos_scale;
	GLint trial_uloc_uv_scale;
	GLint trial_uloc_canvas_tex;
	GLint trial_aloc_pos;
	GLint trial_aloc_signal;
	GLint trial_aloc_color;

	GLuint paint_prg;
	GLint paint_uloc_scale;
	GLint paint_aloc_pos;
	GLint paint_aloc_color;

	GLuint present_prg;
	GLint present_uloc_scale;
	GLint present_uloc_frame;
	GLint present_aloc_pos;
	GLint present_aloc_color;

	GLuint blit_prg;
	GLint blit_uloc_offset;
	GLint blit_uloc_scale;
	GLint blit_uloc_tex;

	GLuint minsamp_prg;
	GLint minsamp_uloc_srcdim;
	GLint minsamp_uloc_dstdim;
	GLint minsamp_uloc_tex;

	struct xoshiro256 rng;

	uint16_t* canvas_image;
	uint16_t* source_image;

	struct config* config;
	int level_index;
	int n_trials_remaining;
	uint16_t winner_triangle[3*6];
	int64_t canvas_sum;
	int canvas_n_weights;
	size_t atomic_buffer_sz;
	uint16_t* vtxbuf;

	double best_score;
	struct triangle best_triangle;
	double best_color_weight;
	uint16_t best_triangle_elems[3*6];

	uint16_t* chosen_vs;

	int source_width;
	int source_height;
	int source_n_channels;

	uint64_t* canvas_cum_weight;
	int* canvas_cum_idx;
	struct stat stat;
} g;


static inline uint64_t rng_next(void)
{
	return xoshiro256_next(&g.rng);
}

static double scalar64 = 0.0;
static inline double rng_nextf(void)
{
	if (scalar64 == 0.0) scalar64 = pow(2.0, -64.0);
	return ((double)rng_next()) * scalar64;
}

static inline void rng_seed(uint64_t seed)
{
	xoshiro256_seed(&g.rng, seed);
}

static inline struct vec2 rng_vec2_on_unit_circle(void)
{
	const double r = sqrt(rng_nextf());
	const double theta = rng_nextf() * 6.283185307179586;
	const double x = r * cos(theta);
	const double y = r * sin(theta);
	return mk_vec2(x,y);
}

#define RED10K   (2126)
#define GREEN10K (7152)
#define BLUE10K  (722)
#define ALPHA10K (10000)

static inline const char* gl_err_string(GLenum err)
{
	switch (err) {
	#define X(NAME) case NAME: return #NAME;
	X(GL_NO_ERROR)
	X(GL_INVALID_ENUM)
	X(GL_INVALID_VALUE)
	X(GL_INVALID_OPERATION)
	X(GL_STACK_OVERFLOW)
	X(GL_STACK_UNDERFLOW)
	X(GL_OUT_OF_MEMORY)
	#undef X
	default: return "???";
	}
}

static inline void _chkgl(const char* file, int line)
{
	GLenum xx_GLERR = glGetError();
	if (xx_GLERR != GL_NO_ERROR) {
		fprintf(stderr, "OPENGL ERROR 0x%.4x (%s) at %s:%d\n", xx_GLERR, gl_err_string(xx_GLERR), file, line);
		abort();
	}
}

#define CHKGL _chkgl(__FILE__, __LINE__)

static const size_t paint_vbo_sz = 1<<20;


static inline int get_n_trial_elems(void) {
	assert(g.source_n_channels > 0);
	return 3 + g.source_n_channels;
}
static inline int get_n_paint_elems(void) {
	assert(g.source_n_channels > 0);
	return 2 + g.source_n_channels;
}
static inline size_t get_trial_stride(void) { return sizeof(uint16_t) * get_n_trial_elems(); }
static inline size_t get_paint_stride(void) { return sizeof(uint16_t) * get_n_paint_elems(); }

static inline int get_n_levels(void)
{
	int n_levels = 0;
	for (struct level* l = g.config->levels; l->n; l++, n_levels++) {};
	return n_levels;
}

static inline int get_n_minsamps(void)
{
	return get_n_levels() - 1;
}

static inline struct level* get_current_level(void)
{
	const int i = g.level_index;
	assert(0 <= i && i < get_n_levels());
	return &g.config->levels[i];
}

static inline int get_tri_num(void)
{
	return 1+(arrlen(g.chosen_vs)/(3*get_n_paint_elems()));
}

static inline double source_area(void) { return g.source_width * g.source_height; }

static void check_shader(GLuint shader, GLenum type, int n_sources, const char** sources)
{
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (status == GL_TRUE) return;

	GLint msglen;
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &msglen);
	GLchar* msg = (GLchar*) malloc(msglen + 1);
	assert(msg != NULL);
	glGetShaderInfoLog(shader, msglen, NULL, msg);
	const char* stype =
		type == GL_COMPUTE_SHADER ? "COMPUTE" :
		type == GL_VERTEX_SHADER ? "VERTEX" :
		type == GL_FRAGMENT_SHADER ? "FRAGMENT" :
		"???";

	// attempt to parse "0:<linenumber>" in error message
	int error_in_line_number = 0;
	if (strlen(msg) >= 3 && msg[0] == '0' && msg[1] == ':' && '0' <= msg[2] && msg[2] <= '9') {
		const char* p0 = msg+2;
		const char* p1 = p0+1;
		while ('0' <= *p1 && *p1 <= '9') p1++;
		char buf[32];
		const int n = p1-p0;
		if (n < sizeof buf) {
			memcpy(buf, p0, n);
			buf[n] = 0;
			error_in_line_number = atoi(buf);
		}
	}

	char errbuf[1<<14];

	char* pe = errbuf;
	char* pe1 = errbuf + sizeof errbuf;

	pe += snprintf(pe, pe1-pe, "%s GLSL COMPILE ERROR: %s in:\n", stype, msg);
	if (error_in_line_number > 0) {
		char line_buffer[1<<14];
		int line_number = 1;
		for (int pi = 0; pi < n_sources; pi++) {
			const char* p = sources[pi];
			int is_end_of_string = 0;
			while (!is_end_of_string)  {
				const char* p0 = p;
				for (;;) {
					char ch = *p;
					if (ch == 0) {
						is_end_of_string = 1;
						break;
					} else if (ch == '\n') {
						p++;
						break;
					} else {
						p++;
					}
				}
				if (p > p0) {
					size_t n = (p-1) - p0;
					if (n >= sizeof(line_buffer)) n = sizeof(line_buffer)-1;
					memcpy(line_buffer, p0, n);
					line_buffer[n] = 0;
					pe += snprintf(pe, pe1-pe, "(%.4d)  %s\n", line_number, line_buffer);
				}
				if (line_number == error_in_line_number) {
					pe += snprintf(pe, pe1-pe, "~^~^~^~ ERROR ~^~^~^~\n");
				}
				line_number++;
			}
			line_number--;
		}
	} else {
		for (int i = 0; i < n_sources; i++) {
			pe += snprintf(pe, pe1-pe, "src[%d]: %s\n", i, sources[i]);
		}
	}
	pe += snprintf(pe, pe1-pe, "shader compilation failed\n");

	fprintf(stderr, "%s\n", errbuf);
	exit(EXIT_FAILURE);
}

static void check_program(GLint program)
{
	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_TRUE) return;
	GLint msglen;
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &msglen);
	GLchar* msg = (GLchar*) malloc(msglen + 1);
	glGetProgramInfoLog(program, msglen, NULL, msg);
	char errbuf[1<<14];
	snprintf(errbuf, sizeof errbuf, "shader link error: %s", msg);
	fprintf(stderr, "%s\n", errbuf);
	exit(EXIT_FAILURE);
}


static GLuint mk_shader(GLenum type, int n_sources, const char** sources)
{
	GLuint shader = glCreateShader(type); CHKGL;
	glShaderSource(shader, n_sources, sources, NULL); CHKGL;
	glCompileShader(shader); CHKGL;
	check_shader(shader, type, n_sources, sources);
	return shader;
}

#if 0
static GLuint mk_compute_program(int n_sources, const char** sources)
{
	GLuint shader = mk_shader(GL_COMPUTE_SHADER, n_sources, sources);
	GLuint program = glCreateProgram(); CHKGL;
	glAttachShader(program, shader); CHKGL;
	glLinkProgram(program); CHKGL;
	check_program(program);

	// when we have a program the shader is no longer needed
	glDeleteShader(shader); CHKGL;

	return program;
}
#endif

static GLuint mk_render_program(int n_vertex_sources, int n_fragment_sources, const char** sources)
{
	const char** vertex_sources = sources;
	const char** fragment_sources = sources +  n_vertex_sources;
	GLuint vertex_shader = mk_shader(GL_VERTEX_SHADER, n_vertex_sources, vertex_sources);
	GLuint fragment_shader = mk_shader(GL_FRAGMENT_SHADER, n_fragment_sources, fragment_sources);
	GLuint program = glCreateProgram(); CHKGL;
	glAttachShader(program, vertex_shader); CHKGL;
	glAttachShader(program, fragment_shader); CHKGL;
	glLinkProgram(program); CHKGL;
	check_program(program);

	// when we have a program the shaders are no longer needed
	glDeleteShader(vertex_shader); CHKGL;
	glDeleteShader(fragment_shader); CHKGL;

	return program;
}

static void do_save_texture_ex(const char* prefix, int id2)
{
	char filename[1<<10];
	if (id2 >= 0) {
		snprintf(filename, sizeof filename, "%s_%d_%d.ppm", prefix, get_tri_num(), id2);
	} else {
		snprintf(filename, sizeof filename, "%s_%d.ppm", prefix, get_tri_num());
	}

	int width = -1;
	int height = -1;
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width); CHKGL;
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height); CHKGL;
	printf("write texture %dx%d\n", width, height);
	assert(width > 0);
	assert(height > 0);
	assert(((width%4) == 0) && "TODO handle non-divisible-by-4 width");
	const int n_pixels = width*height;
	const int n_channels = 3;
	uint8_t* data = malloc(n_channels*n_pixels);
	GLenum format;
	switch (n_channels) {
	case 3: format = GL_RGB; break;
	default: assert(!"unhandled value");
	}
	glGetTexImage(GL_TEXTURE_2D, 0, format, GL_UNSIGNED_BYTE, data); CHKGL;

	FILE* f = fopen(filename, "w");
	fprintf(f, "P3\n");
	fprintf(f, "%d %d\n", width, height);
	fprintf(f, "255\n");
	uint8_t* p = data;
	for (int i0 = 0; i0 < n_pixels; i0++) {
		for (int i1 = 0; i1 < n_channels; i1++) {
			uint8_t px = *(p++);
			fprintf(f, "%s%d", i1 > 0 ? " " : "", (int)px);
		}
		fprintf(f, "\n");
	}
	fclose(f);
	free(data);
}

static void do_save_texture(const char* prefix)
{
	do_save_texture_ex(prefix, -1);
}

static void init(void)
{
	assert(SDL_Init(SDL_INIT_TIMER | SDL_INIT_VIDEO) == 0);
	atexit(SDL_Quit);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	SDL_GL_SetAttribute(SDL_GL_RED_SIZE,     8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE,   8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,    8);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,   24);

	g.window = SDL_CreateWindow(
		"splot!",
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
		1920, 1080,
		SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
	assert(g.window != NULL);
	g.glctx = SDL_GL_CreateContext(g.window);

	SDL_GL_SetSwapInterval(1);

	assert(gl3wInit() == 0);

	GLint gl_major_version, gl_minor_version;
	glGetIntegerv(GL_MAJOR_VERSION, &gl_major_version); CHKGL;
	glGetIntegerv(GL_MINOR_VERSION, &gl_minor_version); CHKGL;
	printf("OpenGL%d.%d / GLSL%s\n", gl_major_version, gl_minor_version, glGetString(GL_SHADING_LANGUAGE_VERSION));

	{
		GLint max_fragment_atomic_counters = -1;
		glGetIntegerv(GL_MAX_FRAGMENT_ATOMIC_COUNTERS, &max_fragment_atomic_counters); CHKGL;
		g.max_batch_size_log2 = 0;
		while ( (1 << (g.max_batch_size_log2+1)) <= (max_fragment_atomic_counters/2) ) {
				g.max_batch_size_log2++;
		}

	}
}

static int frame(void)
{
	if (g.frame_counter > 0) SDL_GL_SwapWindow(g.window);

	SDL_Event ev;
	while (SDL_PollEvent(&ev)) {
		if ((ev.type == SDL_QUIT) || (ev.type == SDL_WINDOWEVENT && ev.window.event == SDL_WINDOWEVENT_CLOSE)) {
			return 0;
		} else {
			switch (ev.type) {
			case SDL_KEYDOWN: {
				int sym = ev.key.keysym.sym;
				//int mod = ev.key.keysym.mod;
				if (sym == SDLK_ESCAPE) return 0;
				if (sym == '1') g.mode = MODE_POSITIVE;
				if (sym == '2') g.mode = MODE_NEGATIVE;
				if (sym == '3') g.mode = MODE_ORIGINAL;
				if (sym == '4') g.mode = MODE_DUMMY;
				if (sym == 's') g.save_image = 1;
				if (sym == 'm') g.save_minsamps = 1;
				if (sym == SDLK_SPACE) g.stretch = !g.stretch;
				} break;
			case SDL_MOUSEMOTION:
				break;
			case SDL_MOUSEWHEEL:
				break;
			case SDL_MOUSEBUTTONDOWN:
			case SDL_MOUSEBUTTONUP:
				break;
			}
		}
	}

	SDL_GL_GetDrawableSize(g.window, &g.window_width, &g.window_height);

	glViewport(0, 0, g.window_width, g.window_height);
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	return 1;
}

static uint16_t candidate_component(uint16_t src_pixel, uint16_t canvas_pixel);
static int accept_triangle(struct triangle, const double* grays);
static double score_candidate(struct triangle, double canvas_color_weight, double vertex_color_weight);

static void stat_tick(void)
{
	TIME now = NOW();
	if (!g.stat._init) {
		g.stat.t0 = now;
		g.stat._init = 1;
	}
	const double dt = DIFF(now, g.stat.t0);
	if (dt < 1.0) return;

	printf("STATISTICS frame=%d tri %d\n", g.frame_counter, get_tri_num());
	#define ST(N) \
	{ \
		printf("  " #N "/s = %.2f\n", (double)g.stat.N / dt); \
		g.stat.N = 0; \
	}
	ST(n_lvls)
	ST(n_lvl0s)
	ST(n_generated_triangles)
	ST(n_accepted_triangles)
	ST(n_trial_runs)
	ST(n_trials)
	ST(n_underflows)
	ST(n_rejected)
	ST(n_accepted)
	#undef ST
	g.stat.t0 = now;
}

static inline int get_minsamp_dim(int index, int* pw, int* ph)
{
	const int w = g.config->levels[index].w;
	assert(w > 0);
	const int h = (g.source_height * w + g.source_width-1) / g.source_width;
	if (pw) *pw = w;
	if (ph) *ph = h;
}

static void process_search(void)
{
	stat_tick();

	if (g.save_minsamps) {
		const int n_minsamp = get_n_minsamps();
		for (int i = 0; i < n_minsamp; i++) {
			glBindTexture(GL_TEXTURE_2D, g.minsamp_texs[i]); CHKGL;
			do_save_texture_ex("minsamp", i);
		}
		glBindTexture(GL_TEXTURE_2D, g.canvas_tex); CHKGL;
		do_save_texture_ex("minsamp", n_minsamp);
	}
	g.save_minsamps = 0;

	if (g.n_trials_remaining == 0) {
		const int n_levels = get_n_levels();
		g.level_index = (g.level_index + 1) % n_levels;
		g.n_trials_remaining = get_current_level()->n;
		g.stat.n_lvls++;

		g.best_score = 0.0;

		if (g.level_index == 0) {
			g.stat.n_lvl0s++;

			const int n_minsamp = get_n_minsamps();
			if (n_minsamp > 0) {
				assert(g.minsamp_texs != NULL);
				glUseProgram(g.minsamp_prg); CHKGL;
				glBindFramebuffer(GL_FRAMEBUFFER, g.framebuffer); CHKGL;
				glBindVertexArray(g.vao); CHKGL;
				for (int i = 0; i < n_minsamp; i++) {
					glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g.minsamp_texs[i], /*level=*/0); CHKGL;
					assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
					int w,h;
					get_minsamp_dim(i, &w, &h);
					glViewport(0, 0, w, h); CHKGL;
					glUniform2f(g.minsamp_uloc_dstdim, w, h); CHKGL;
					glUniform2f(g.minsamp_uloc_srcdim, g.source_width, g.source_height); CHKGL;
					glUniform1i(g.minsamp_uloc_tex, 0); CHKGL;
					glBindTexture(GL_TEXTURE_2D, g.canvas_tex); CHKGL;
					glDisable(GL_BLEND);
					glDrawArrays(GL_TRIANGLES, 0, 6); CHKGL;
					glEnable(GL_BLEND);
				}
				glUseProgram(0); CHKGL;
			}

			// download canvas, and use it to construct new binary searchable
			// table for weighted pixel picking
			glBindTexture(GL_TEXTURE_2D, g.canvas_tex); CHKGL;

			{
				int tw, th;
				glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &tw);
				glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &th);
				assert(tw == g.source_width);
				assert(th == g.source_height);
			}
			glGetTexImage(GL_TEXTURE_2D, 0, g.source_format, GL_UNSIGNED_SHORT, g.canvas_image);
			glBindTexture(GL_TEXTURE_2D, 0); CHKGL;

			uint16_t* p = g.canvas_image;
			int64_t canvas_accum = 0;
			uint64_t* cwp = g.canvas_cum_weight;
			int* cip = g.canvas_cum_idx;
			int pixel_index = 0;
			for (int y = 0; y < g.source_height; y++) {
				for (int x = 0; x < g.source_width; x++) {
					int s = 0;
					switch (g.source_n_channels) {
					case 1:
						s += *(p++);
						break;
					case 3:
						s += *(p++) * RED10K;
						s += *(p++) * GREEN10K;
						s += *(p++) * BLUE10K;
						break;
					case 4:
						s += *(p++) * RED10K;
						s += *(p++) * GREEN10K;
						s += *(p++) * BLUE10K;
						s += *(p++) * ALPHA10K;
						break;
					default: assert(!"unhandled source_n_channels");
					}

					if (s > 0) {
						canvas_accum += s;
						*(cwp++) = canvas_accum;
						*(cip++) = pixel_index;
					}
					pixel_index++;
				}
			}
			g.canvas_sum = canvas_accum;
			g.canvas_n_weights = cwp - g.canvas_cum_weight;
			assert(g.canvas_n_weights == (cip - g.canvas_cum_idx));
		}
	}

	arrsetlen(g.vtxbuf, 0);

	struct level* lvl = get_current_level();
	const int virtual_width = (lvl->w == 0 || lvl->w > g.source_width) ? g.source_width : lvl->w;
	const int virtual_height = (g.source_height * virtual_width) / g.source_width;

	int batch_index = 0;
	const int max_batch_size = 1 << g.max_batch_size_log2;
	for (batch_index = 0; batch_index < max_batch_size && g.n_trials_remaining > 0; batch_index++, g.n_trials_remaining--) {
		uint16_t* v0 = arraddnptr(g.vtxbuf, 3*get_n_trial_elems());
		const int max_attempts = 100; // CFG?
		for (int attempt = 0; attempt < max_attempts; attempt++) {
			uint16_t* vp = v0;
			if (g.level_index == 0) {
				for (int point = 0; point < 3; point++) {
					assert(g.canvas_sum > 0);
					int idx, px, py;
					if (attempt < max_attempts/2) { // CFG?
						uint64_t find = rng_next() % g.canvas_sum;

						int left = 0;
						int right = g.canvas_n_weights;
						int n_iterations = 0;
						while (left < right) {
							n_iterations++;
							int mid = (left + right) >> 1;
							uint64_t wk = g.canvas_cum_weight[mid];
							if (wk > find) {
								right = mid;
							} else {
								left = mid + 1;
							}
						}

						const int cidx = (right-1) < 0 ? 0 : (right-1);
						assert(cidx >= 0);
						idx = g.canvas_cum_idx[cidx];
						px = idx % g.source_width;
						py = idx / g.source_width;
					} else {
						uint64_t r0 = rng_next() % g.canvas_sum;
						px = r0 % g.source_width;
						py = (r0/g.source_width) % g.source_height;
						idx = px + py * g.source_width;
					}
					uint16_t* canvas_pixel = &g.canvas_image[idx*g.source_n_channels];
					uint16_t* src_pixel = &g.source_image[idx*g.source_n_channels];

					*(vp++) = px;
					*(vp++) = py;
					*(vp++) = batch_index;
					for (int i = 0; i < g.source_n_channels; i++) {
						*(vp++) = candidate_component(src_pixel[i], canvas_pixel[i]);
					}
				}
			} else {
				assert(g.level_index > 0);
				for (int point = 0; point < 3; point++) {
					struct vec2 d = rng_vec2_on_unit_circle();
					struct level* lvl = get_current_level();
					const double r = lvl->r * ((double)g.source_width / (double)virtual_width);
					assert(r > 0);
					d = vec2_scale(d, r);

					const int of = point * get_n_paint_elems();
					int px = g.winner_triangle[of+0] + d.x;
					int py = g.winner_triangle[of+1] + d.y;
					//printf("d%f,%f -> %d,%d\n", d.x, d.y, px, py);
					if (px < 0) px = 0;
					if (py < 0) py = 0;
					if (px >= g.source_width) px = g.source_width-1;
					if (py >= g.source_height) py = g.source_height-1;
					*(vp++) = px;
					*(vp++) = py;
					*(vp++) = batch_index;
					for (int i = 0; i < g.source_n_channels; i++) {
						*(vp++) = g.winner_triangle[of+2+i];
					}
				}
			}

			g.stat.n_generated_triangles++;

			const int nt = get_n_trial_elems();

			double grays[] = {0,0,0};
			const double gs = (65535.0 * lvl->gn);
			const double cs = (65535.0 * lvl->cn);
			const double gn = gs == 0.0 ? 0.0 : rng_nextf() * gs - gs*0.5;
			for (int point = 0; point < 3; point++) {
				uint16_t* vp = &v0[point*nt+3];
				for (int i = 0; i < g.source_n_channels; i++) {
					int c = vp[i];
					c += gn;
					if (cs != 0.0) c += rng_nextf() * cs - cs*0.5;
					if (c < 0) c = 0;
					if (c > 65535) c = 65535;
					vp[i] = c;
				}

				double gray;
				switch (g.source_n_channels) {
				case 1:
					gray = (double)vp[0] * (1.0 / 65536.0);
					break;
				case 3:
					gray = (
						  (double)vp[0] * (double)RED10K
						+ (double)vp[1] * (double)GREEN10K
						+ (double)vp[2] * (double)BLUE10K
						) * (1.0 / (65536.0 * 10000.0));
					break;
				case 4:
					gray = (
						  (double)vp[0] * (double)RED10K
						+ (double)vp[1] * (double)GREEN10K
						+ (double)vp[2] * (double)BLUE10K
						+ (double)vp[3] * (double)ALPHA10K
						) * (1.0 / (65536.0 * 20000.0));
					break;
				default: assert(!"unhandled value");
				}
				grays[point] = gray;
			}

			assert((vp-v0) == 3*nt);
			struct triangle T = mk_triangle(
				v0[0*nt+0], v0[0*nt+1],
				v0[1*nt+0], v0[1*nt+1],
				v0[2*nt+0], v0[2*nt+1]);
			double signed_area = triangle_area(T);
			if (signed_area < 0.0) {
				uint16_t tmp[7];
				const size_t sz = sizeof(tmp[0]) * nt;
				memcpy(tmp, &v0[1*nt], sz);
				memcpy(&v0[1*nt], &v0[2*nt], sz);
				memcpy(&v0[2*nt], tmp, sz);
				T = mk_triangle(
					v0[0*nt+0], v0[0*nt+1],
					v0[1*nt+0], v0[1*nt+1],
					v0[2*nt+0], v0[2*nt+1]);
				signed_area = triangle_area(T);
			}
			assert(signed_area >= 0.0);
			if (accept_triangle(T, grays)) {
				g.stat.n_accepted_triangles++;
				break;
			}
		}
	}
	const int batch_size = batch_index;

	assert(batch_size > 0);

	glBindFramebuffer(GL_FRAMEBUFFER, g.framebuffer); CHKGL;
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g.dummy_fb_tex, /*level=*/0); CHKGL;
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	glViewport(0, 0, g.source_width, g.source_height);
	if (g.mode == MODE_DUMMY) {
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	glUseProgram(g.trial_prg); CHKGL;

	// clear atomic signals
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, g.signal_buf); CHKGL;
	{
		GLuint* a = glMapBufferRange(
			GL_ATOMIC_COUNTER_BUFFER,
			0, g.atomic_buffer_sz,
			GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT); CHKGL;
		memset(a, 0, g.atomic_buffer_sz);
		glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER); CHKGL;
	}
	glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, g.signal_buf, 0, g.atomic_buffer_sz); CHKGL;

	glBindBuffer(GL_ARRAY_BUFFER, g.trial_vbo); CHKGL;
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(g.vtxbuf[0]) * arrlen(g.vtxbuf), g.vtxbuf); CHKGL;

	glBindVertexArray(g.vao); CHKGL;
	glEnableVertexAttribArray(g.trial_aloc_pos); CHKGL;
	glEnableVertexAttribArray(g.trial_aloc_signal); CHKGL;
	glEnableVertexAttribArray(g.trial_aloc_color); CHKGL;

	glVertexAttribPointer(g.trial_aloc_pos, 2, GL_UNSIGNED_SHORT, GL_TRUE, get_trial_stride(), (void*)0); CHKGL;
	glVertexAttribPointer(g.trial_aloc_signal, 1, GL_UNSIGNED_SHORT, GL_FALSE, get_trial_stride(), (void*)4); CHKGL;
	glVertexAttribPointer(g.trial_aloc_color,  g.source_n_channels, GL_UNSIGNED_SHORT, GL_TRUE,  get_trial_stride(), (void*)6); CHKGL;

	GLuint trial_tex;
	if (g.level_index < (get_n_levels() - 1)) {
		assert(g.level_index >= 0);
		trial_tex = g.minsamp_texs[g.level_index];
	} else {
		assert(g.level_index == (get_n_levels() - 1));
		trial_tex = g.canvas_tex;
	}
	glBindTexture(GL_TEXTURE_2D, trial_tex); CHKGL;
	const double sx = 65535.0f / (float)g.source_width;
	const double sy = 65535.0f / (float)g.source_height;
	glUniform2f(g.trial_uloc_pos_scale,
		sx * ((float)virtual_width / (float)g.source_width),
		sy * ((float)virtual_height / (float)g.source_height)
	); CHKGL;
	glUniform2f(g.trial_uloc_uv_scale,
		sx,
		sy
	); CHKGL;
	glUniform1i(g.trial_uloc_canvas_tex, 0); CHKGL;

	g.stat.n_trial_runs++;
	glDrawArrays(GL_TRIANGLES, 0, 3*batch_size); CHKGL;

	int n_trials = 0;
	int n_underflows = 0;
	{
		GLuint* atomics = glMapBufferRange(
			GL_ATOMIC_COUNTER_BUFFER,
			0, g.atomic_buffer_sz,
			GL_MAP_READ_BIT);

		int i1 = 0;
		for (int i0 = 0; i0 < ((batch_size+31)>>5); i0++) {
			for (; i1 < ((i0<<5)+32) && i1 < batch_size; i1++) {
				const int underflow = atomics[i0] & (1 << (i1&31));
				n_trials++;
				g.stat.n_trials++;
				if (underflow) {
					n_underflows++;
					g.stat.n_underflows++;
					continue;
				}
				uint16_t* v0 = &g.vtxbuf[3*get_n_trial_elems()*i1];
				uint16_t* v1 = v0;
				int64_t canvas_color_accum = 0;
				int64_t vertex_color_accum = 0;
				double accum_scale;
				for (int i2 = 0; i2 < 3; i2++, v1 += get_n_trial_elems()) {
					assert(v1[2] == i1);
					uint16_t* vp0 = &g.canvas_image[(v1[0] + v1[1] * g.source_width) * g.source_n_channels];
					uint16_t* vp1 = &v1[3];

					switch (g.source_n_channels) {
					case 1:
						canvas_color_accum += *(vp0++);
						vertex_color_accum += *(vp1++);
						accum_scale = 1.0 / (2.0 * 65536.0);
						break;
					case 3:
						canvas_color_accum += *(vp0++) * (int64_t)RED10K;
						canvas_color_accum += *(vp0++) * (int64_t)GREEN10K;
						canvas_color_accum += *(vp0++) * (int64_t)BLUE10K;
						vertex_color_accum += *(vp1++) * (int64_t)RED10K;
						vertex_color_accum += *(vp1++) * (int64_t)GREEN10K;
						vertex_color_accum += *(vp1++) * (int64_t)BLUE10K;
						accum_scale = 1.0 / (20000.0 * 65536.0);
						break;
					case 4:
						canvas_color_accum += *(vp0++) * (int64_t)RED10K;
						canvas_color_accum += *(vp0++) * (int64_t)GREEN10K;
						canvas_color_accum += *(vp0++) * (int64_t)BLUE10K;
						canvas_color_accum += *(vp0++) * (int64_t)ALPHA10K;
						vertex_color_accum += *(vp1++) * (int64_t)RED10K;
						vertex_color_accum += *(vp1++) * (int64_t)GREEN10K;
						vertex_color_accum += *(vp1++) * (int64_t)BLUE10K;
						vertex_color_accum += *(vp1++) * (int64_t)ALPHA10K;
						accum_scale = 1.0 / (40000.0 * 65536.0);
						break;
					default: assert(!"unhandled source_n_channels");
					}
				}
				const double canvas_color_weight = (double)canvas_color_accum * accum_scale;
				const double vertex_color_weight = (double)vertex_color_accum * accum_scale;

				const int nt = get_n_trial_elems();
				const struct triangle T = mk_triangle(
					v0[0*nt+0], v0[0*nt+1],
					v0[1*nt+0], v0[1*nt+1],
					v0[2*nt+0], v0[2*nt+1]);

				const double score = score_candidate(T, canvas_color_weight, vertex_color_weight);
				if (score <= 0.0) {
					g.stat.n_rejected++;
					continue;
				}

				g.stat.n_accepted++;

				if (score > g.best_score) {
					g.best_score = score;
					g.best_triangle = T;
					g.best_color_weight = (double)vertex_color_weight;
					uint16_t* src = &g.vtxbuf[3*nt*i1];
					uint16_t* dst = g.best_triangle_elems;
					for (int i = 0; i < 3; i++) {
						*(dst++) = *(src++);
						*(dst++) = *(src++);
						src++;
						for (int j = 0; j < g.source_n_channels; j++) {
							*(dst++) = *(src++);
						}
					}
				}
			}
		}

		glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER); CHKGL;
	}

	glBindTexture(GL_TEXTURE_2D, 0); CHKGL;

	glDisableVertexAttribArray(g.trial_aloc_color); CHKGL;
	glDisableVertexAttribArray(g.trial_aloc_signal); CHKGL;
	glDisableVertexAttribArray(g.trial_aloc_pos); CHKGL;
	glBindVertexArray(0); CHKGL;

	glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0); CHKGL;
	glUseProgram(0); CHKGL;

	assert(g.n_trials_remaining >= 0);
	if (g.n_trials_remaining == 0) {
		assert(g.best_score >= 0.0);
		if (g.best_score == 0.0) {
			//g.n_trials_remaining = get_current_level()->n;
			//printf("at level %d; failed to find a valid candidate after %d trials; going for another pass\n", g.level_index+1, g.n_trials_remaining);
			printf("at level %d; failure after %d trials; starting over\n", g.level_index+1, get_current_level()->n);
			g.level_index = -1;
		} else {
			const int n_levels = get_n_levels();
			const int is_final_level = (g.level_index+1) == n_levels;
			#if 0
			printf("tria [%f,%f]-[%f,%f]-[%f,%f]\n",
					g.best_triangle.A.x, g.best_triangle.A.y, 
					g.best_triangle.B.x, g.best_triangle.B.y, 
					g.best_triangle.C.x, g.best_triangle.C.y);
			#endif
			printf("tri %d:%d/%d :: area=%.0f; fat=%.3f; cw=%.6f; underflow=%d/%d\n",
				get_tri_num(),
				1+g.level_index,
				n_levels,
				fabs(triangle_area(g.best_triangle)),
				triangle_fatness(g.best_triangle),
				g.best_color_weight,
				n_underflows,
				n_trials
			);
			const int ncp = 3*get_n_paint_elems();
			const size_t psz = sizeof(g.best_triangle_elems[0])*ncp;
			if (!is_final_level) {
				memcpy(g.winner_triangle, g.best_triangle_elems, psz);
			} else {
				// store triangle, update canvas
				uint16_t* vout = arraddnptr(g.chosen_vs, ncp);
				memcpy(vout, g.best_triangle_elems, psz);

				glBindBuffer(GL_ARRAY_BUFFER, g.paint_vbo); CHKGL;
				const size_t blitsz = sizeof(g.chosen_vs[0]) * arrlen(g.chosen_vs);
				assert(blitsz <= paint_vbo_sz);
				glBufferSubData(GL_ARRAY_BUFFER, 0, blitsz, g.chosen_vs); CHKGL;
				glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;

				glBindFramebuffer(GL_FRAMEBUFFER, g.framebuffer); CHKGL;
				glViewport(0, 0, g.source_width, g.source_height);
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g.canvas_tex, /*level=*/0); CHKGL;
				assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
				glUseProgram(g.paint_prg); CHKGL;
				glBindVertexArray(g.vao); CHKGL;
				glBindBuffer(GL_ARRAY_BUFFER, g.paint_vbo); CHKGL;
				glEnableVertexAttribArray(g.paint_aloc_pos); CHKGL;
				glEnableVertexAttribArray(g.paint_aloc_color); CHKGL;

				glVertexAttribPointer(g.paint_aloc_pos, 2, GL_UNSIGNED_SHORT, GL_TRUE, get_paint_stride(), (void*)0); CHKGL;
				glVertexAttribPointer(g.paint_aloc_color,  g.source_n_channels, GL_UNSIGNED_SHORT, GL_TRUE,  get_paint_stride(), (void*)4); CHKGL;
				glUniform2f(g.paint_uloc_scale, 65536.0f / (float)g.source_width, 65536.0f / (float)g.source_height); CHKGL;

				glBlendFunc(GL_ONE, GL_ONE); CHKGL;
				glBlendEquation(GL_FUNC_REVERSE_SUBTRACT); CHKGL;

				glDrawArrays(GL_TRIANGLES, arrlen(g.chosen_vs)/get_n_paint_elems()-3, 3); CHKGL;
				glBlendEquation(GL_FUNC_ADD); CHKGL;

				glDisableVertexAttribArray(g.paint_aloc_color); CHKGL;
				glDisableVertexAttribArray(g.paint_aloc_pos); CHKGL;
				glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;
				glBindVertexArray(0); CHKGL;
				glUseProgram(0); CHKGL;
			}
		}
	}
}

static void process_draw(void)
{
	if (g.mode == MODE_POSITIVE) {
		glBindFramebuffer(GL_FRAMEBUFFER, g.framebuffer); CHKGL;
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g.dummy_fb_tex, /*level=*/0); CHKGL;
		assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
		glViewport(0, 0, g.source_width, g.source_height); CHKGL;
		glClearColor(0,0,0,0); CHKGL;
		glClear(GL_COLOR_BUFFER_BIT); CHKGL;

		if (arrlen(g.chosen_vs) > 0) {
			glUseProgram(g.present_prg); CHKGL;
			glBindVertexArray(g.vao); CHKGL;
			glBindBuffer(GL_ARRAY_BUFFER, g.paint_vbo); CHKGL;
			glEnableVertexAttribArray(g.present_aloc_pos); CHKGL;
			glEnableVertexAttribArray(g.present_aloc_color); CHKGL;

			glVertexAttribPointer(g.present_aloc_pos, 2, GL_UNSIGNED_SHORT, GL_TRUE, get_paint_stride(), (void*)0); CHKGL;
			glVertexAttribPointer(g.present_aloc_color,  g.source_n_channels, GL_UNSIGNED_SHORT, GL_TRUE,  get_paint_stride(), (void*)4); CHKGL;
			glUniform2f(g.present_uloc_scale, 65536.0f / (float)g.source_width, 65536.0f / (float)g.source_height); CHKGL;
			glUniform1ui(g.present_uloc_frame, g.frame_counter); CHKGL;

			glBlendFunc(GL_ONE, GL_ONE); CHKGL;

			glDrawArrays(GL_TRIANGLES, 0, arrlen(g.chosen_vs)/get_n_paint_elems()); CHKGL;

			glDisableVertexAttribArray(g.present_aloc_color); CHKGL;
			glDisableVertexAttribArray(g.present_aloc_pos); CHKGL;
			glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;
			glBindVertexArray(0); CHKGL;
			glUseProgram(0); CHKGL;
		}
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0); CHKGL;
	glViewport(0, 0, g.window_width, g.window_height);

	glUseProgram(g.blit_prg); CHKGL;
	glBindVertexArray(g.vao); CHKGL;

	const float sw = g.stretch ? 1.0f : (float)g.source_width / (float)g.window_width;
	const float sh = g.stretch ? 1.0f : (float)g.source_height / (float)g.window_height;

	switch (g.mode) {
	case MODE_POSITIVE:
		glBindTexture(GL_TEXTURE_2D, g.dummy_fb_tex); CHKGL;
		if (g.save_image) do_save_texture("positive");
		glUniform2f(g.blit_uloc_offset, 0.0f, 0.0f); CHKGL;
		glUniform2f(g.blit_uloc_scale, sw, sh); CHKGL;
		break;
	case MODE_NEGATIVE:
		glBindTexture(GL_TEXTURE_2D, g.canvas_tex); CHKGL;
		if (g.save_image) do_save_texture("negative");
		glUniform2f(g.blit_uloc_offset, 0.0f, 0.0f); CHKGL;
		glUniform2f(g.blit_uloc_scale, -sw, -sh); CHKGL;
		break;
	case MODE_ORIGINAL:
		glBindTexture(GL_TEXTURE_2D, g.original_tex); CHKGL;
		if (g.save_image) do_save_texture("original");
		glUniform2f(g.blit_uloc_offset, 0.0f, 0.0f); CHKGL;
		glUniform2f(g.blit_uloc_scale, -sw, -sh); CHKGL;
		break;
	case MODE_DUMMY:
		glBindTexture(GL_TEXTURE_2D, g.dummy_fb_tex); CHKGL;
		if (g.save_image) do_save_texture("dummy");
		glUniform2f(g.blit_uloc_offset, 0.0f, 0.0f); CHKGL;
		glUniform2f(g.blit_uloc_scale, -sw, -sh); CHKGL;
		break;
	default: assert(!"unhandled mode");
	}
	g.save_image = 0;

	glUniform1i(g.blit_uloc_tex, 0);
	glDisable(GL_BLEND);
	glDrawArrays(GL_TRIANGLES, 0, 6); CHKGL;
	glEnable(GL_BLEND);
	glBindTexture(GL_TEXTURE_2D, 0); CHKGL;
	glBindVertexArray(0); CHKGL;
	glUseProgram(0); CHKGL;
}


#define IS_Q0 "(gl_VertexID == 0 || gl_VertexID == 3)"
#define IS_Q1 "(gl_VertexID == 1)"
#define IS_Q2 "(gl_VertexID == 2 || gl_VertexID == 4)"
#define IS_Q3 "(gl_VertexID == 5)"

static void splot_process(const char* image_path, struct config* config)
{
	g.config = config;

	{
		assert(get_n_levels() > 0);
		assert((config->levels[get_n_levels()-1].w == 0) && "last level must be source width");
	}

	init();

	assert(image_path != NULL);
	g.source_image = stbi_load_16(image_path, &g.source_width, &g.source_height, &g.source_n_channels, 0);
	if (g.source_image == NULL) {
		fprintf(stderr, "%s: could not read\n", image_path);
		exit(EXIT_FAILURE);
	}
	printf("%d×%dc%d\n", g.source_width, g.source_height, g.source_n_channels);

	const size_t canvas_image_sz = g.source_width*g.source_height*g.source_n_channels*sizeof(uint16_t);
	g.canvas_image = malloc(canvas_image_sz);
	//memset(g.canvas_image, -1, canvas_image_sz);
	g.canvas_cum_weight = malloc(g.source_width*g.source_height*sizeof(*g.canvas_cum_weight));
	g.canvas_cum_idx = malloc(g.source_width*g.source_height*sizeof(*g.canvas_cum_idx));

	GLint internal_format;
	int pixel_size = -1;
	switch (g.source_n_channels) {
	case 1:
		internal_format = GL_R16;
		g.source_format = GL_RGB;
		pixel_size = 2;
		break;
	case 3:
		internal_format = GL_RGB16;
		g.source_format = GL_RGB;
		pixel_size = 6;
		break;
	case 4:
		internal_format = GL_RGBA16;
		g.source_format = GL_RGBA;
		pixel_size = 8;
		break;
	default:
		fprintf(stderr, "unhandled number of image channels: %d\n", g.source_n_channels);
		exit(EXIT_FAILURE);
	}

	for (int tex = 0; tex < 2; tex++) {
		GLuint* tp = tex == 0 ? &g.canvas_tex : tex == 1 ? &g.original_tex : NULL;
		glGenTextures(1, tp); CHKGL;
		glBindTexture(GL_TEXTURE_2D, *tp); CHKGL;
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); CHKGL;
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CHKGL;
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CHKGL;
		assert((((g.source_width*pixel_size) & 3) == 0) && "FIXME unaligned rows; use glPixelStorei(GL_UNPACK_ALIGNMENT, 1)?");
		const GLvoid* data = g.source_image;
		glTexImage2D(GL_TEXTURE_2D, /*level=*/0, internal_format, g.source_width, g.source_height, /*border=*/0, g.source_format, GL_UNSIGNED_SHORT, data); CHKGL;
		//glGenerateMipmap(GL_TEXTURE_2D); CHKGL;
		glBindTexture(GL_TEXTURE_2D, 0); CHKGL;
	}

	const int n_levels = get_n_levels();
	const int n_minsamp = n_levels-1;
	if (n_minsamp > 0) {
		if (g.minsamp_texs == NULL) {
			g.minsamp_texs = malloc(n_minsamp * sizeof *g.minsamp_texs);
			for (int i = 0; i < n_minsamp; i++) {
				GLuint* tex = &g.minsamp_texs[i];
				glGenTextures(1, tex);
				glBindTexture(GL_TEXTURE_2D, *tex); CHKGL;
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CHKGL;
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CHKGL;
				int w,h;
				get_minsamp_dim(i, &w, &h);
				glTexImage2D(GL_TEXTURE_2D, /*level=*/0, internal_format, w, h, /*border=*/0, g.source_format, GL_UNSIGNED_SHORT, NULL); CHKGL;
			}
		}
	}
	glBindTexture(GL_TEXTURE_2D, 0); CHKGL;

	glGenFramebuffers(1, &g.framebuffer); CHKGL;

	glGenTextures(1, &g.dummy_fb_tex); CHKGL;
	glBindTexture(GL_TEXTURE_2D, g.dummy_fb_tex); CHKGL;
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CHKGL;
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CHKGL;
	glTexImage2D(GL_TEXTURE_2D, /*level=*/0, GL_RGB, g.source_width, g.source_height, /*border=*/0, GL_RGB, GL_UNSIGNED_BYTE, NULL); CHKGL;

	glBindTexture(GL_TEXTURE_2D, 0); CHKGL;

	glGenVertexArrays(1, &g.vao); CHKGL;

	glGenBuffers(1, &g.trial_vbo); CHKGL;
	glBindBuffer(GL_ARRAY_BUFFER, g.trial_vbo); CHKGL;
	glBufferData(GL_ARRAY_BUFFER, (1 << g.max_batch_size_log2) * 3 * get_trial_stride(), NULL, GL_DYNAMIC_DRAW); CHKGL;
	glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;

	glGenBuffers(1, &g.paint_vbo); CHKGL;
	glBindBuffer(GL_ARRAY_BUFFER, g.paint_vbo); CHKGL;
	glBufferData(GL_ARRAY_BUFFER, paint_vbo_sz, NULL, GL_DYNAMIC_DRAW); CHKGL;
	glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;

	const char* define_colortype_glsl = NULL;
	switch (g.source_n_channels) {
	case 1:
		define_colortype_glsl = "#define COLORTYPE float\n";
		break;
	case 3:
		define_colortype_glsl = "#define COLORTYPE vec3\n";
		break;
	case 4:
		define_colortype_glsl = "#define COLORTYPE vec4\n";
		break;
	default: assert(!"unhandled source_n_channels");
	}

	assert(g.max_batch_size_log2 >= 5);
	const int n_signal_u32s = 1 << (g.max_batch_size_log2 - 5);
	char n_signal_u32s_str[1<<5];
	snprintf(n_signal_u32s_str, sizeof n_signal_u32s_str, "%d", n_signal_u32s);

	glGenBuffers(1, &g.signal_buf); CHKGL;
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, g.signal_buf);
	g.atomic_buffer_sz = n_signal_u32s * sizeof(GLuint);
	glBufferData(GL_ATOMIC_COUNTER_BUFFER, g.atomic_buffer_sz, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

	const char* blit_sources[] = {
	// vertex
	"#version 460\n"
	"\n"
	"layout (location = 0) uniform vec2 u_offset;\n"
	"layout (location = 1) uniform vec2 u_scale;\n"
	"\n"
	"out vec2 v_uv;\n"
	"\n"
	"void main()\n"
	"{\n"
	"	vec2 c;\n"
	"	if (" IS_Q0 ") {\n"
	"		c = vec2(-1.0, -1.0);\n"
	"		v_uv = vec2(0.0, 0.0);\n"
	"	} else if (" IS_Q1 ") {\n"
	"		c = vec2( 1.0, -1.0);\n"
	"		v_uv = vec2(1.0, 0.0);\n"
	"	} else if (" IS_Q2 ") {\n"
	"		c = vec2( 1.0,  1.0);\n"
	"		v_uv = vec2(1.0, 1.0);\n"
	"	} else if (" IS_Q3 ") {\n"
	"		c = vec2(-1.0,  1.0);\n"
	"		v_uv = vec2(0.0, 1.0);\n"
	"	}\n"
	"	gl_Position = vec4(u_offset + (c * u_scale), 0.0, 1.0);\n"
	"}\n"
	,
	// fragment
	"#version 460\n"
	"\n"
	"layout (location = 2) uniform sampler2D u_tex;\n"
	"\n"
	"in vec2 v_uv;\n"
	"\n"
	"layout (location = 0) out vec4 frag_color;\n"
	"\n"
	"void main()\n"
	"{\n"
	"	frag_color = texture(u_tex, v_uv);\n"
	"}\n"
	};

	g.blit_prg = mk_render_program(1, 1, blit_sources);

	g.blit_uloc_offset = glGetUniformLocation(g.blit_prg, "u_offset");
	g.blit_uloc_scale = glGetUniformLocation(g.blit_prg, "u_scale");
	g.blit_uloc_tex = glGetUniformLocation(g.blit_prg, "u_tex");


	const char* paint_sources[] = {
	// vertex
	"#version 460\n"
	"\n"
	, define_colortype_glsl ,
	"\n"
	"layout (location = 0) uniform vec2 u_scale;\n"
	"\n"
	"in vec2 a_pos;\n"
	"in COLORTYPE a_color;\n"
	"\n"
	"out COLORTYPE v_color;\n"
	"\n"
	"void main()\n"
	"{\n"
	"	v_color = a_color;\n"
	"	vec2 c = a_pos * u_scale;\n"
	"	vec2 pos = vec2(-1.0, -1.0) + c * vec2(2.0,  2.0);\n"
	"	gl_Position = vec4(pos, 0.0, 1.0);\n"
	"}\n"

	,

	// fragment
	"#version 460\n"
	"\n"
	, define_colortype_glsl ,
	"\n"
	"in COLORTYPE v_color;\n"
	"\n"
	"layout (location = 0) out vec4 frag_color;\n"
	"\n"
	"void main()\n"
	"{\n"
	,
	(
	g.source_n_channels == 1 ?
	"	frag_color = vec4(v_color, v_color, v_color, 1.0);\n"
	: g.source_n_channels == 3 ?
	"	frag_color = vec4(v_color, 1.0);\n"
	: g.source_n_channels == 4 ?
	"	frag_color = v_color;\n"
	: "BANG"
	)
	,
	"}\n"
	};

	g.paint_prg = mk_render_program(3, 5, paint_sources);

	g.paint_uloc_scale = glGetUniformLocation(g.paint_prg, "u_scale");

	g.paint_aloc_pos    = glGetAttribLocation(g.paint_prg, "a_pos");
	g.paint_aloc_color  = glGetAttribLocation(g.paint_prg, "a_color");

	const char* hash_glsl =
	"// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.\n"
	"uint jhash(uint x)\n"
	"{\n"
    "	x += ( x << 10u );\n"
    "	x ^= ( x >>  6u );\n"
    "	x += ( x <<  3u );\n"
    "	x ^= ( x >> 11u );\n"
    "	x += ( x << 15u );\n"
    "	return x;\n"
	"}\n"
	;

	const char* present_sources[] = {
	// vertex
	"#version 460\n"
	"\n"
	,
	define_colortype_glsl
	,
	hash_glsl
	,
	"\n"
	"layout (location = 0) uniform vec2 u_scale;\n"
	"layout (location = 1) uniform uint u_frame;\n"
	"\n"
	"in vec2 a_pos;\n"
	"in COLORTYPE a_color;\n"
	"\n"
	"out COLORTYPE v_color;\n"
	"float bf(uint v)\n"
	"{\n"
	"	return float(v) * (1.0 / 255.0);\n"
	"}\n"
	"\n"
	"float uxf(uint v, int i)\n"
	"{\n"
	"	if (i == 0) {\n"
	"		return bf(v & 255u);\n"
	"	} else if (i == 1) {\n"
	"		return bf((v >> 8u) & 255u);\n"
	"	} else if (i == 2) {\n"
	"		return bf((v >> 16u) & 255u);\n"
	"	} else if (i == 3) {\n"
	"		return bf((v >> 24u) & 255u);\n"
	"	} else {\n"
	"		return 0.0;\n"
	"	}\n"
	"}\n"
	"\n"
	"float cc(float v)\n"
	"{\n"
	"	return (v - 0.5) * (1.0 / 255.0);\n"
	"}\n"
	"\n"
	"void main()\n"
	"{\n"
	"	uint r0 = jhash(gl_VertexID + (u_frame << 16u));\n"
	,
	(
	g.source_n_channels == 1 ?
	"	float nois = cc(uxf(r0,0));\n"
	: g.source_n_channels == 3 ?
	"	vec3 nois = vec3(cc(uxf(r0,0)),cc(uxf(r0,1)),cc(uxf(r0,2)));\n"
	: g.source_n_channels == 4 ?
	"	vec4 nois = vec4(cc(uxf(r0,0)),cc(uxf(r0,1)),cc(uxf(r0,2)),cc(uxf(r0,3)));\n"
	: "BANG"
	)
	,
	"	v_color = a_color + nois;\n"
	"	vec2 c = a_pos * u_scale;\n"
	"	vec2 pos = vec2(1.0, 1.0) + c * vec2(-2.0, -2.0);\n"
	"	gl_Position = vec4(pos, 0.0, 1.0);\n"
	"}\n"

	,

	// fragment
	"#version 460\n"
	"\n"
	, define_colortype_glsl ,
	"\n"
	"in COLORTYPE v_color;\n"
	"\n"
	"layout (location = 0) out vec4 frag_color;\n"
	"\n"
	"void main()\n"
	"{\n"
	,
	(
	g.source_n_channels == 1 ?
	"	frag_color = vec4(v_color, v_color, v_color, 1.0);\n"
	: g.source_n_channels == 3 ?
	"	frag_color = vec4(v_color, 1.0);\n"
	: g.source_n_channels == 4 ?
	"	frag_color = v_color;\n"
	: "BANG"
	)
	,
	"}\n"
	};

	g.present_prg = mk_render_program(6, 5, present_sources);

	g.present_uloc_scale = glGetUniformLocation(g.present_prg, "u_scale");
	g.present_uloc_frame = glGetUniformLocation(g.present_prg, "u_frame");

	g.present_aloc_pos    = glGetAttribLocation(g.present_prg, "a_pos");
	g.present_aloc_color  = glGetAttribLocation(g.present_prg, "a_color");


	const char* trial_sources[] = {
	// vertex
	"#version 460\n"
	"\n"
	, define_colortype_glsl ,
	"\n"
	"layout (location = 0) uniform vec2 u_pos_scale;\n"
	"layout (location = 1) uniform vec2 u_uv_scale;\n"
	"\n"
	"in vec2 a_pos;\n"
	"in float a_signal;\n"
	"in COLORTYPE a_color;\n"
	"\n"
	"     out vec2 v_uv;\n"
	"     out COLORTYPE v_color;\n"
	"flat out uint v_signal_arrindex;\n"
	"flat out uint v_signal_mask;\n"
	"\n"
	"void main()\n"
	"{\n"
	"	vec2 pos = vec2(-1.0, -1.0) + (a_pos * u_pos_scale) * vec2(2.0,  2.0);\n"
	"	v_uv = a_pos * u_uv_scale;\n"
	"	v_color = a_color;\n"
	"	v_signal_arrindex = uint(a_signal) >> 5u;\n"
	"	v_signal_mask = 1u << (uint(a_signal) & 31u);\n"
	"	gl_Position = vec4(pos, 0.0, 1.0);\n"
	"}\n"

	,

	// fragment
	"#version 460\n"
	"\n"
	, define_colortype_glsl ,
	"\n"
	"layout (location = 2) uniform sampler2D u_canvas_tex;\n"
	"layout (binding = 0, offset = 0) uniform atomic_uint u_signal[\n"
		, n_signal_u32s_str ,
	"];\n"
	"\n"
	"     in vec2 v_uv;\n"
	"     in COLORTYPE v_color;\n"
	"flat in uint v_signal_arrindex;\n"
	"flat in uint v_signal_mask;\n"
	"\n"
	"layout (location = 0) out vec4 frag_color;\n"
	"\n"
	"void main()\n"
	"{\n"

	,
	(
	g.source_n_channels == 1 ?
	"	COLORTYPE d = texture(u_canvas_tex, v_uv).x - v_color;\n"
	"	if (d < 0.0) {\n"
	: g.source_n_channels == 3 ?
	"	COLORTYPE d = texture(u_canvas_tex, v_uv).xyz - v_color;\n"
	"	if (d.x < 0.0 || d.y < 0.0 || d.z < 0.0) {\n"
	: g.source_n_channels == 4 ?
	"	COLORTYPE d = texture(u_canvas_tex, v_uv) - v_color;\n"
	"	if (d.x < 0.0 || d.y < 0.0 || d.z < 0.0 || d.w < 0.0) {\n"
	: "BANG"
	)
	,

	"		atomicCounterOr(u_signal[v_signal_arrindex], v_signal_mask);\n"
	"	}\n"
	,
	(
	g.source_n_channels == 1 ?
	"	frag_color = vec4(d,d,d,1.0);\n"
	: g.source_n_channels == 3 ?
	"	frag_color = vec4(d,1.0);\n"
	: g.source_n_channels == 4 ?
	"	frag_color = d;\n"
	: "BANG"
	)
	,
	"}\n"

	};

	g.trial_prg = mk_render_program(3, 9, trial_sources);

	g.trial_uloc_pos_scale = glGetUniformLocation(g.trial_prg, "u_pos_scale");
	g.trial_uloc_uv_scale = glGetUniformLocation(g.trial_prg, "u_uv_scale");
	g.trial_uloc_canvas_tex = glGetUniformLocation(g.trial_prg, "u_canvas_tex");

	g.trial_aloc_pos    = glGetAttribLocation(g.trial_prg, "a_pos");
	g.trial_aloc_signal = glGetAttribLocation(g.trial_prg, "a_signal");
	g.trial_aloc_color  = glGetAttribLocation(g.trial_prg, "a_color");

	const char* minsamp_sources[] = {
	// vertex
	"#version 460\n"
	"\n"
	"out vec2 v_uv;\n"
	"\n"
	"void main()\n"
	"{\n"
	"	vec2 c;\n"
	"	if (" IS_Q0 ") {\n"
	"		c = vec2(-1.0, -1.0);\n"
	"		v_uv = vec2(0.0, 0.0);\n"
	"	} else if (" IS_Q1 ") {\n"
	"		c = vec2( 1.0, -1.0);\n"
	"		v_uv = vec2(1.0, 0.0);\n"
	"	} else if (" IS_Q2 ") {\n"
	"		c = vec2( 1.0,  1.0);\n"
	"		v_uv = vec2(1.0, 1.0);\n"
	"	} else if (" IS_Q3 ") {\n"
	"		c = vec2(-1.0,  1.0);\n"
	"		v_uv = vec2(0.0, 1.0);\n"
	"	}\n"
	"	gl_Position = vec4(c, 0.0, 1.0);\n"
	"}\n"
	,
	// fragment
	"#version 460\n"
	"\n"
	"layout (location = 0) uniform vec2 u_srcdim;\n"
	"layout (location = 1) uniform vec2 u_dstdim;\n"
	"layout (location = 2) uniform sampler2D u_tex;\n"
	"\n"
	"in vec2 v_uv;\n"
	"\n"
	"layout (location = 0) out vec4 frag_color;\n"
	"\n"
	"vec2 snap(vec2 v)\n"
	"{\n"
	"	return floor(v * u_dstdim) / u_dstdim;\n"
	"}\n"
	"\n"
	"void main()\n"
	"{\n"
	"	vec2 p0 = snap(v_uv);\n"
	"	vec2 p1 = p0 + vec2(1.0/u_dstdim.x, 1.0/u_dstdim.y);\n"
	"	vec2 d = vec2(1.0/u_srcdim.x, 1.0/u_srcdim.y);\n"
	"	vec4 v = vec4(1.0, 1.0, 1.0, 1.0);\n"
	"	for (float y = p0.y; y < p1.y; y += d.y) {\n"
	"		for (float x = p0.x; x < p1.x; x += d.x) {\n"
	"			v = min(v, texture(u_tex, vec2(x,y)));\n"
	"		}\n"
	"	}\n"
	"	frag_color = v;\n"
	"}\n"
	};

	g.minsamp_prg = mk_render_program(1, 1, minsamp_sources);
	g.minsamp_uloc_srcdim = glGetUniformLocation(g.minsamp_prg, "u_srcdim");
	g.minsamp_uloc_dstdim = glGetUniformLocation(g.minsamp_prg, "u_dstdim");
	g.minsamp_uloc_tex = glGetUniformLocation(g.minsamp_prg, "u_tex");

	//glDisable(GL_CULL_FACE); CHKGL;
	glEnable(GL_BLEND); CHKGL;

	g.level_index = -1;
	stat_tick();

	while (frame()) {
		const int n_searches_per_draw = 1; // CFG
		for (int i = 0; i < n_searches_per_draw; i++) {
			process_search();
		}
		process_draw();
		g.frame_counter++;
	}
}

#define SPLOT_H
#endif
