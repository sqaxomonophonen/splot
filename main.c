#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdarg.h>

#include "gl3w.h"
#include "stb_image.h"
#include "stb_ds.h"
#include <SDL.h>
#include "xoshiro256plusplus.h"

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
	return (struct vec2) { .x=a.x-b.x, .y=a.y-b.y };
}

static inline double vec2_distance(struct vec2 a, struct vec2 b)
{
	return vec2_length(vec2_sub(b, a));
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
	return triangle_inradius(t) / triangle_circumradius(t);
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

static const int trial_batch_size_log2 = 9; // CFG
static const int n_trials_per_primitive = 1<<9; // CFG
static const size_t paint_vbo_sz = 1<<20;

static struct {
	SDL_Window* window;
	SDL_GLContext glctx;
	int window_width;
	int window_height;
	GLint max_fragment_atomic_counters;

	unsigned frame_counter;

	int next_primitve;
	int64_t canvas_accum;
	int canvas_n_weights;
	int trial_counter;

	GLuint framebuffer;
	GLuint vao;
	GLuint canvas_tex;
	GLuint original_tex;
	GLenum source_format;
	GLuint dummy_fb_tex;
	GLuint trial_prg, present_prg, paint_prg, blit_prg;
	GLuint signal_buf;
	GLuint paint_vbo;
	GLuint trial_vbo;

	GLint trial_uloc_scale;
	GLint trial_uloc_canvas_tex;
	GLint trial_aloc_pos;
	GLint trial_aloc_signal;
	GLint trial_aloc_color;

	GLint paint_uloc_scale;
	GLint paint_aloc_pos;
	GLint paint_aloc_color;

	GLint present_uloc_scale;
	GLint present_uloc_frame;
	GLint present_aloc_pos;
	GLint present_aloc_color;

	GLint blit_uloc_offset;
	GLint blit_uloc_scale;
	GLint blit_uloc_tex;

	struct xoshiro256 rng;

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
	uint16_t* canvas_image;
	uint16_t* source_image;
	uint64_t* canvas_cum_weight;
	int* canvas_cum_idx;
	int n_trial_elems;
	int trial_stride;
	int n_paint_elems;
	int paint_stride;
} g;

const char* prg = NULL;

__attribute__((noreturn))
static void usage(const char* msg)
{
	if (msg != NULL) fprintf(stderr, "%s\n", msg);
	fprintf(stderr, "Usage: %s <cmd> [args...]\n", prg);
	fprintf(stderr, "  %s process </path/to/image>\n", prg);
	fprintf(stderr, "  %s view </path/to/primlist>\n", prg);
	fprintf(stderr, "  %s render </path/to/primlist>\n", prg);
	exit(EXIT_FAILURE);
}


static void usagef(const char* fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char buf[1<<14];
	vsnprintf(buf, sizeof buf, fmt, args);
	usage(buf);
	va_end(args);
}

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


	g.max_fragment_atomic_counters = -1;
	glGetIntegerv(GL_MAX_FRAGMENT_ATOMIC_COUNTERS, &g.max_fragment_atomic_counters); CHKGL;
	printf("max fragment atomic counters: %d\n", g.max_fragment_atomic_counters);
}

enum mode {
	MODE_POSITIVE = 0,
	MODE_NEGATIVE,
	MODE_ORIGINAL,
	MODE_DUMMY,
};

static enum mode mode = MODE_POSITIVE;
static int stretch = 0;
static int frame_number;
static int frame(void)
{
	if (frame_number > 0) SDL_GL_SwapWindow(g.window);

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
				if (sym == '1') mode = MODE_POSITIVE;
				if (sym == '2') mode = MODE_NEGATIVE;
				if (sym == '3') mode = MODE_ORIGINAL;
				if (sym == '4') mode = MODE_DUMMY;
				if (sym == SDLK_SPACE) stretch = !stretch;
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

	frame_number++;

	return 1;
}

static void process_search(void)
{
	if (g.next_primitve) {
		g.trial_counter = 0;
		g.next_primitve = 0;
		g.best_score = 0.0;

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
		g.canvas_accum = 0;
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
					g.canvas_accum += s;
					*(cwp++) = g.canvas_accum;
					*(cip++) = pixel_index;
				}
				pixel_index++;
			}
		}
		g.canvas_n_weights = cwp - g.canvas_cum_weight;
		assert(g.canvas_n_weights == (cip - g.canvas_cum_idx));
	}

	arrsetlen(g.vtxbuf, 0);

	int batch_trial_index = 0;
	const int trial_batch_size = 1 << trial_batch_size_log2;
	for (batch_trial_index = 0; batch_trial_index < trial_batch_size && g.trial_counter < n_trials_per_primitive; batch_trial_index++, g.trial_counter++) {
		uint16_t* v0 = arraddnptr(g.vtxbuf, 3*g.n_trial_elems);
		const int max_attempts = 50; // CFG
		for (int attempt = 0; attempt < max_attempts; attempt++) {
			uint16_t* v = v0;
			for (int point = 0; point < 3; point++) {
				assert(g.canvas_accum > 0);
				int idx, px, py;
				if (attempt < max_attempts/2) { // CFG?
					uint64_t find = xoshiro256_next(&g.rng) % g.canvas_accum;

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
					uint64_t r0 = xoshiro256_next(&g.rng) % g.canvas_accum;
					px = r0 % g.source_width;
					py = (r0>>16) % g.source_height;
					idx = px + py * g.source_width;
				}
				uint16_t* canvas_pixel = &g.canvas_image[idx*g.source_n_channels];
				uint16_t* src_pixel = &g.source_image[idx*g.source_n_channels];

				*(v++) = px;
				*(v++) = py;
				*(v++) = batch_trial_index;
				for (int i = 0; i < g.source_n_channels; i++) {
					int p = src_pixel[i] >> 4; // CFG
					int cp = canvas_pixel[i]-1;
					if (cp < 0) cp = 0;
					if (p > cp) p = cp;
					*(v++) = p;
					//const uint64_t rn0 = xoshiro256_next(&g.rng);
					//*(v++) = ((uint64_t)src_pixel[i] * (rn0 & 0xffff)) >> 19;
				}
			}
			assert((v-v0) == 3*g.n_trial_elems);
			const struct triangle T = mk_triangle(
				v0[0*g.n_trial_elems+0], v0[0*g.n_trial_elems+1],
				v0[1*g.n_trial_elems+0], v0[1*g.n_trial_elems+1],
				v0[2*g.n_trial_elems+0], v0[2*g.n_trial_elems+1]);
			const double area = fabs(triangle_area(T));
			const double area_ratio = area / (double)(g.source_width*g.source_height);
			if (area_ratio > 1.0 / (6.0*6.0)) continue; // CFG
			const double fat = triangle_fatness(T);
			if (fat < 0.1) continue; // CFG
			break;
		}
	}
	const int batch_size = batch_trial_index;

	glBindFramebuffer(GL_FRAMEBUFFER, g.framebuffer); CHKGL;
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g.dummy_fb_tex, /*level=*/0); CHKGL;
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	glViewport(0, 0, g.source_width, g.source_height);
	if (mode == MODE_DUMMY) {
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	glUseProgram(g.trial_prg); CHKGL;

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

	glVertexAttribPointer(g.trial_aloc_pos, 2, GL_UNSIGNED_SHORT, GL_TRUE, g.trial_stride, (void*)0); CHKGL;
	glVertexAttribPointer(g.trial_aloc_signal, 1, GL_UNSIGNED_SHORT, GL_FALSE, g.trial_stride, (void*)4); CHKGL;
	glVertexAttribPointer(g.trial_aloc_color,  g.source_n_channels, GL_UNSIGNED_SHORT, GL_TRUE,  g.trial_stride, (void*)6); CHKGL;


	glBindTexture(GL_TEXTURE_2D, g.canvas_tex); CHKGL;
	glUniform2f(g.trial_uloc_scale, 65536.0f / (float)g.source_width, 65536.0f / (float)g.source_height); CHKGL;
	glUniform1i(g.trial_uloc_canvas_tex, 0); CHKGL;

	glDrawArrays(GL_TRIANGLES, 0, 3*batch_size); CHKGL;

	{
		GLuint* atomics = glMapBufferRange(
			GL_ATOMIC_COUNTER_BUFFER,
			0, g.atomic_buffer_sz,
			GL_MAP_READ_BIT);

		int i1 = 0;
		for (int i0 = 0; i0 < ((batch_size+31)>>5); i0++) {
			for (; i1 < ((i0<<5)+32) && i1 < batch_size; i1++) {
				const int underflow = atomics[i0] & (1 << (i1&31));
				if (underflow) {
					continue;
				}
				uint16_t* v0 = &g.vtxbuf[3*g.n_trial_elems*i1];
				uint16_t* v1 = v0;
				int64_t color_accum = 0;
				double accum_scale;
				for (int i2 = 0; i2 < 3; i2++, v1 += g.n_trial_elems) {
					assert(v1[2] == i1);
					uint16_t* vp0 = &g.canvas_image[(v1[0] + v1[1] * g.source_width) * g.source_n_channels];
					uint16_t* vp1 = &v1[3];
					const int64_t w0 = 5; // CFG
					const int64_t w1 = 1; // CFG

					switch (g.source_n_channels) {
					case 1:
						color_accum += *(vp0++) * w0;
						color_accum += *(vp1++) * w1;
						accum_scale = 1.0 / (2.0 * 65536.0);
						break;
					case 3:
						color_accum += *(vp0++) * RED10K * w0;
						color_accum += *(vp0++) * GREEN10K * w0;
						color_accum += *(vp0++) * BLUE10K * w0;
						color_accum += *(vp1++) * RED10K * w1;
						color_accum += *(vp1++) * GREEN10K * w1;
						color_accum += *(vp1++) * BLUE10K * w1;
						accum_scale = 1.0 / ((double)(w0+w1) * 20000.0 * 65536.0);
						break;
					case 4:
						color_accum += *(vp0++) * RED10K * w0;
						color_accum += *(vp0++) * GREEN10K * w0;
						color_accum += *(vp0++) * BLUE10K * w0;
						color_accum += *(vp0++) * ALPHA10K * w0;
						color_accum += *(vp1++) * RED10K * w1;
						color_accum += *(vp1++) * GREEN10K * w1;
						color_accum += *(vp1++) * BLUE10K * w1;
						color_accum += *(vp1++) * ALPHA10K * w1;
						accum_scale = 1.0 / ((double)(w0+w1) * 40000.0 * 65536.0);
						break;
					default: assert(!"unhandled source_n_channels");
					}
				}
				const double color_weight = (double)color_accum * accum_scale;
				const struct triangle T = mk_triangle(
					v0[0*g.n_trial_elems+0], v0[0*g.n_trial_elems+1],
					v0[1*g.n_trial_elems+0], v0[1*g.n_trial_elems+1],
					v0[2*g.n_trial_elems+0], v0[2*g.n_trial_elems+1]);

				const double area = fabs(triangle_area(T));
				if (area < 10.0) { // CFG? RETHINK?
					continue;
				}
				const double fat = triangle_fatness(T);
				if (fat < 0.01) { // CFG? RETHINK?
					continue;
				}
				const double area_ratio = area / (double)(g.source_width*g.source_height);
				const double area_score = pow(area_ratio, 0.01); // CFG
				const double score = (0.2 + fat) * color_weight * area_score; // CFG

				if (score > g.best_score) {
					g.best_score = score;
					g.best_triangle = T;
					g.best_color_weight = (double)color_weight;
					uint16_t* src = &g.vtxbuf[3*g.n_trial_elems*i1];
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

	if (g.trial_counter == n_trials_per_primitive) {
		g.next_primitve = 1;
		if (g.best_score > 0) {
			const int ncp = 3*g.n_paint_elems;
			uint16_t* vout = arraddnptr(g.chosen_vs, ncp);
			memcpy(vout, g.best_triangle_elems, sizeof(vout[0])*ncp);
			printf("chose triangle %zd: area=%.0f; fat=%.3f; cw=%.6f\n",
				arrlen(g.chosen_vs)/(3*g.n_paint_elems),
				fabs(triangle_area(g.best_triangle)),
				triangle_fatness(g.best_triangle),
				g.best_color_weight
			);

			glBindBuffer(GL_ARRAY_BUFFER, g.paint_vbo); CHKGL;
			const size_t blitsz = sizeof(g.chosen_vs[0]) * arrlen(g.chosen_vs);
			assert(blitsz <= paint_vbo_sz);
			glBufferSubData(GL_ARRAY_BUFFER, 0, blitsz, g.chosen_vs); CHKGL;
			glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;

			// update canvas

			glBindFramebuffer(GL_FRAMEBUFFER, g.framebuffer); CHKGL;
			glViewport(0, 0, g.source_width, g.source_height);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g.canvas_tex, /*level=*/0); CHKGL;
			assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
			glUseProgram(g.paint_prg); CHKGL;
			glBindVertexArray(g.vao); CHKGL;
			glBindBuffer(GL_ARRAY_BUFFER, g.paint_vbo); CHKGL;
			glEnableVertexAttribArray(g.paint_aloc_pos); CHKGL;
			glEnableVertexAttribArray(g.paint_aloc_color); CHKGL;

			glVertexAttribPointer(g.paint_aloc_pos, 2, GL_UNSIGNED_SHORT, GL_TRUE, g.paint_stride, (void*)0); CHKGL;
			glVertexAttribPointer(g.paint_aloc_color,  g.source_n_channels, GL_UNSIGNED_SHORT, GL_TRUE,  g.paint_stride, (void*)4); CHKGL;
			glUniform2f(g.paint_uloc_scale, 65536.0f / (float)g.source_width, 65536.0f / (float)g.source_height); CHKGL;

			glBlendFunc(GL_ONE, GL_ONE); CHKGL;
			glBlendEquation(GL_FUNC_REVERSE_SUBTRACT); CHKGL;

			glDrawArrays(GL_TRIANGLES, arrlen(g.chosen_vs)/g.n_paint_elems-3, 3); CHKGL;
			glBlendEquation(GL_FUNC_ADD); CHKGL;

			glDisableVertexAttribArray(g.paint_aloc_color); CHKGL;
			glDisableVertexAttribArray(g.paint_aloc_pos); CHKGL;
			glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;
			glBindVertexArray(0); CHKGL;
			glUseProgram(0); CHKGL;

		}
	}
}

static void process_draw(void)
{
	if (mode == MODE_POSITIVE) {
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

			glVertexAttribPointer(g.present_aloc_pos, 2, GL_UNSIGNED_SHORT, GL_TRUE, g.paint_stride, (void*)0); CHKGL;
			glVertexAttribPointer(g.present_aloc_color,  g.source_n_channels, GL_UNSIGNED_SHORT, GL_TRUE,  g.paint_stride, (void*)4); CHKGL;
			glUniform2f(g.present_uloc_scale, 65536.0f / (float)g.source_width, 65536.0f / (float)g.source_height); CHKGL;
			glUniform1ui(g.present_uloc_frame, g.frame_counter); CHKGL;

			glBlendFunc(GL_ONE, GL_ONE); CHKGL;

			glDrawArrays(GL_TRIANGLES, 0, arrlen(g.chosen_vs)/g.n_paint_elems); CHKGL;

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

	const float sw = stretch ? 1.0f : (float)g.source_width / (float)g.window_width;
	const float sh = stretch ? 1.0f : (float)g.source_height / (float)g.window_height;

	switch (mode) {
	case MODE_POSITIVE:
		glBindTexture(GL_TEXTURE_2D, g.dummy_fb_tex); CHKGL;
		glUniform2f(g.blit_uloc_offset, 0.0f, 0.0f); CHKGL;
		glUniform2f(g.blit_uloc_scale, sw, sh); CHKGL;
		break;
	case MODE_NEGATIVE:
		glBindTexture(GL_TEXTURE_2D, g.canvas_tex); CHKGL;
		glUniform2f(g.blit_uloc_offset, 0.0f, 0.0f); CHKGL;
		glUniform2f(g.blit_uloc_scale, -sw, -sh); CHKGL;
		break;
	case MODE_ORIGINAL:
		glBindTexture(GL_TEXTURE_2D, g.original_tex); CHKGL;
		glUniform2f(g.blit_uloc_offset, 0.0f, 0.0f); CHKGL;
		glUniform2f(g.blit_uloc_scale, -sw, -sh); CHKGL;
		break;
	case MODE_DUMMY:
		glBindTexture(GL_TEXTURE_2D, g.dummy_fb_tex); CHKGL;
		glUniform2f(g.blit_uloc_offset, 0.0f, 0.0f); CHKGL;
		glUniform2f(g.blit_uloc_scale, -sw, -sh); CHKGL;
		break;
	default: assert(!"unhandled mode");
	}
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

static void process_run(const char* image_path)
{
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
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); CHKGL;
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CHKGL;
		assert((((g.source_width*pixel_size) & 3) == 0) && "FIXME unaligned rows; use glPixelStorei(GL_UNPACK_ALIGNMENT, 1)?");
		const GLvoid* data = g.source_image;
		glTexImage2D(GL_TEXTURE_2D, /*level=*/0, internal_format, g.source_width, g.source_height, /*border=*/0, g.source_format, GL_UNSIGNED_SHORT, data); CHKGL;
		glGenerateMipmap(GL_TEXTURE_2D); CHKGL;
		glBindTexture(GL_TEXTURE_2D, 0); CHKGL;
	}

	glGenFramebuffers(1, &g.framebuffer); CHKGL;

	glGenTextures(1, &g.dummy_fb_tex); CHKGL;
	glBindTexture(GL_TEXTURE_2D, g.dummy_fb_tex); CHKGL;
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CHKGL;
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CHKGL;
	glTexImage2D(GL_TEXTURE_2D, /*level=*/0, GL_RGB, g.source_width, g.source_height, /*border=*/0, GL_RGB, GL_UNSIGNED_BYTE, NULL); CHKGL;

	glBindTexture(GL_TEXTURE_2D, 0); CHKGL;

	g.n_trial_elems = 3 + g.source_n_channels;
	g.trial_stride = sizeof(uint16_t) * g.n_trial_elems;

	g.n_paint_elems = 2 + g.source_n_channels;
	g.paint_stride = sizeof(uint16_t) * g.n_paint_elems;

	glGenVertexArrays(1, &g.vao); CHKGL;

	glGenBuffers(1, &g.trial_vbo); CHKGL;
	glBindBuffer(GL_ARRAY_BUFFER, g.trial_vbo); CHKGL;
	glBufferData(GL_ARRAY_BUFFER, (1 << trial_batch_size_log2) * 3 * g.trial_stride, NULL, GL_DYNAMIC_DRAW); CHKGL;
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

	const int n_signal_u32s = 1 << (trial_batch_size_log2 - 5);
	assert(g.max_fragment_atomic_counters >= n_signal_u32s);
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
	"layout (location = 0) uniform vec2 u_scale;\n"
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
	"	vec2 c = a_pos * u_scale;\n"
	"	v_uv = c;\n"
	"	v_color = a_color;\n"
	"	v_signal_arrindex = uint(a_signal) >> 5u;\n"
	"	v_signal_mask = 1u << (uint(a_signal) & 31u);\n"
	"	vec2 pos = vec2(-1.0, -1.0) + c * vec2(2.0,  2.0);\n"
	"	gl_Position = vec4(pos, 0.0, 1.0);\n"
	"}\n"

	,

	// fragment
	"#version 460\n"
	"\n"
	, define_colortype_glsl ,
	"\n"
	"layout (location = 1) uniform sampler2D u_canvas_tex;\n"
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

	g.trial_uloc_scale = glGetUniformLocation(g.trial_prg, "u_scale");
	g.trial_uloc_canvas_tex = glGetUniformLocation(g.trial_prg, "u_canvas_tex");

	g.trial_aloc_pos    = glGetAttribLocation(g.trial_prg, "a_pos");
	g.trial_aloc_signal = glGetAttribLocation(g.trial_prg, "a_signal");
	g.trial_aloc_color  = glGetAttribLocation(g.trial_prg, "a_color");

	//glDisable(GL_CULL_FACE); CHKGL;
	glEnable(GL_BLEND); CHKGL;

	xoshiro256_seed(&g.rng, 0); // CFG

	g.next_primitve = 1;

	while (frame()) {
		g.frame_counter++;
		const int n_searches_per_draw = 10;
		for (int i = 0; i < n_searches_per_draw; i++) {
			process_search();
		}
		process_draw();
	}
}

int main(int argc, char** argv)
{
	prg = argv[0];
	if (argc < 2) usage(NULL);

	const char* cmd = argv[1];

	if (strcmp("process", cmd) == 0) {
		if (argc < 3) usagef("image is missing\n");
		const char* image_path = argv[2];
		process_run(image_path);
	} else if (strcmp("view", cmd) == 0) {
		assert(!"TODO");
	} else if (strcmp("render", cmd) == 0) {
		assert(!"TODO");
	} else {
		usagef("invalid cmd \"%s\"", cmd);
	}

	return EXIT_SUCCESS;

}
