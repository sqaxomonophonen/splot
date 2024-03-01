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

static inline struct triangle mk_triangle(double x1, double y1, double x2, double y2, double x3, double y3)
{
	return (struct triangle) {
		.A = mk_vec2(x1,y1),
		.B = mk_vec2(x2,y2),
		.C = mk_vec2(x3,y3),
	};
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

static inline double triangle_semiperimeter(struct triangle t)
{
	const double a = vec2_distance(t.A, t.B);
	const double b = vec2_distance(t.B, t.C);
	const double c = vec2_distance(t.C, t.A);
	return (a+b+c)/2.0;
}

static inline double triangle_inradius(struct triangle t)
{
	return triangle_area(t) / triangle_semiperimeter(t);
}

static inline double triangle_circumradius(struct triangle t)
{
	const double a = vec2_distance(t.A, t.B);
	const double b = vec2_distance(t.B, t.C);
	const double c = vec2_distance(t.C, t.A);
	return (a*b*c) / (4.0 * triangle_area(t));
}

static inline double triangle_score(struct triangle t)
{
	//return triangle_circumradius(t) / triangle_inradius(t);
	return triangle_inradius(t) / triangle_circumradius(t);
}


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

static struct {
	SDL_Window* window;
	int width;
	int height;
	GLint max_fragment_atomic_counters;
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
	SDL_GLContext glctx = SDL_GL_CreateContext(g.window);

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
				if (sym == SDLK_ESCAPE) return 0;
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

	SDL_GL_GetDrawableSize(g.window, &g.width, &g.height);

	glViewport(0, 0, g.width, g.height);
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	frame_number++;

	return 1;
}

static void mode_process(const char* image_path)
{
	init();

	int width, height, n_channels;
	assert(image_path != NULL);
	uint16_t* source_image = stbi_load_16(image_path, &width, &height, &n_channels, 0);
	if (source_image == NULL) {
		fprintf(stderr, "%s: could not read\n", image_path);
		exit(EXIT_FAILURE);
	}
	printf("%d×%dc%d\n", width, height, n_channels);

	uint16_t* canvas_image = malloc(width*height*n_channels*sizeof(*canvas_image ));
	uint64_t* canvas_cum_weight = malloc(width*height*sizeof(*canvas_cum_weight));
	int* canvas_cum_idx = malloc(width*height*sizeof(*canvas_cum_idx));

	GLuint canvas_tex;

	glGenTextures(1, &canvas_tex); CHKGL;
	glBindTexture(GL_TEXTURE_2D, canvas_tex); CHKGL;
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CHKGL;
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CHKGL;
	GLint internal_format;
	GLenum format;
	int pixel_size = -1;
	switch (n_channels) {
	case 1:
		internal_format = GL_R16;
		format = GL_RGB;
		pixel_size = 2;
		break;
	case 3:
		internal_format = GL_RGB16;
		format = GL_RGB;
		pixel_size = 6;
		break;
	case 4:
		internal_format = GL_RGBA16;
		format = GL_RGBA;
		pixel_size = 8;
		break;
	default:
		fprintf(stderr, "unhandled number of image channels: %d\n", n_channels);
		exit(EXIT_FAILURE);
	}
	assert((((width*pixel_size) & 3) == 0) && "FIXME unaligned rows; use glPixelStorei(GL_UNPACK_ALIGNMENT, 1)?");
	glTexImage2D(GL_TEXTURE_2D, /*level=*/0, internal_format, width, height, /*border=*/0, format, GL_UNSIGNED_SHORT, source_image); CHKGL;
	glBindTexture(GL_TEXTURE_2D, 0); CHKGL;

	GLuint fb;
	glGenFramebuffers(1, &fb); CHKGL;

	GLuint dummy_fb_tex;
	glGenTextures(1, &dummy_fb_tex); CHKGL;
	glBindTexture(GL_TEXTURE_2D, dummy_fb_tex); CHKGL;
	glTexImage2D(GL_TEXTURE_2D, /*level=*/0, GL_RED, width, height, /*border=*/0, GL_RED, GL_UNSIGNED_BYTE, NULL); CHKGL;

	glBindTexture(GL_TEXTURE_2D, 0); CHKGL;

	const int trial_batch_size_log2 = 10;

	const int n_trials_per_primitive = 1<<10; // TODO configurable?
	const int n_trial_elems = 3 + n_channels;
	const int trial_stride = sizeof(uint16_t) * n_trial_elems;

	const int n_paint_elems = 2 + n_channels;
	const int paint_stride = sizeof(uint16_t) * n_paint_elems;

	GLuint vao;
	glGenVertexArrays(1, &vao); CHKGL;

	GLuint trial_vbo;
	glGenBuffers(1, &trial_vbo); CHKGL;
	glBindBuffer(GL_ARRAY_BUFFER, trial_vbo); CHKGL;
	glBufferData(GL_ARRAY_BUFFER, (1 << trial_batch_size_log2) * 3 * trial_stride, NULL, GL_DYNAMIC_DRAW); CHKGL;
	glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;

	GLuint paint_vbo;
	glGenBuffers(1, &paint_vbo); CHKGL;
	glBindBuffer(GL_ARRAY_BUFFER, paint_vbo); CHKGL;
	const size_t paint_vbo_sz = 1<<20;
	glBufferData(GL_ARRAY_BUFFER, paint_vbo_sz, NULL, GL_DYNAMIC_DRAW); CHKGL;
	glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;

	const char* define_colortype_str = NULL;
	switch (n_channels) {
	case 1:
		define_colortype_str = "#define COLORTYPE float\n";
		break;
	case 3:
		define_colortype_str = "#define COLORTYPE vec3\n";
		break;
	case 4:
		define_colortype_str = "#define COLORTYPE vec4\n";
		break;
	default: assert(!"unhandled n_channels");
	}

	const int n_signal_u32s = 1 << (trial_batch_size_log2 - 5);
	assert(g.max_fragment_atomic_counters >= n_signal_u32s);
	char n_signal_u32s_str[1<<5];
	snprintf(n_signal_u32s_str, sizeof n_signal_u32s_str, "%d", n_signal_u32s);

	GLuint signal_buf;
	glGenBuffers(1, &signal_buf); CHKGL;
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, signal_buf);
	const size_t atomic_buffer_sz = n_signal_u32s * sizeof(GLuint);
	glBufferData(GL_ATOMIC_COUNTER_BUFFER, atomic_buffer_sz, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

	const char* paint_sources[] = {
	// vertex
	"#version 460\n"
	"\n"
	, define_colortype_str ,
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
	, define_colortype_str ,
	"\n"
	"in COLORTYPE v_color;\n"
	"\n"
	"layout (location = 0) out vec4 frag_color;\n"
	"\n"
	"void main()\n"
	"{\n"
	,
	(
	n_channels == 1 ?
	"	frag_color = vec4(v_color, v_color, v_color, 1.0);\n"
	: n_channels == 3 ?
	"	frag_color = vec4(v_color, 1.0);\n"
	: n_channels == 4 ?
	"	frag_color = v_color;\n"
	: "BANG"
	)
	,
	"}\n"
	};

	GLuint paint_prg = mk_render_program(3, 5, paint_sources);

	const GLint paint_uloc_scale = glGetUniformLocation(paint_prg, "u_scale");

	const GLint paint_aloc_pos    = glGetAttribLocation(paint_prg, "a_pos");
	const GLint paint_aloc_color  = glGetAttribLocation(paint_prg, "a_color");


	const char* trial_sources[] = {
	// vertex
	"#version 460\n"
	"\n"
	, define_colortype_str ,
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
	, define_colortype_str ,
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
	n_channels == 1 ?
	"	COLORTYPE d = texture(u_canvas_tex, v_uv).x - v_color;\n"
	"	if (d <= 0.0) {\n"
	: n_channels == 3 ?
	"	COLORTYPE d = texture(u_canvas_tex, v_uv).xyz - v_color;\n"
	"	if (d.x <= 0.0 || d.y <= 0.0 || d.z <= 0.0) {\n"
	: n_channels == 4 ?
	"	COLORTYPE d = texture(u_canvas_tex, v_uv) - v_color;\n"
	"	if (d.x <= 0.0 || d.y <= 0.0 || d.z <= 0.0 || d.w <= 0.0) {\n"
	: "BANG"
	)
	,

	"		atomicCounterOr(u_signal[v_signal_arrindex], v_signal_mask);\n"
	"	}\n"
	,
	(
	n_channels == 1 ?
	"	frag_color = vec4(d,d,d,1.0);\n"
	: n_channels == 3 ?
	"	frag_color = vec4(d,1.0);\n"
	: n_channels == 4 ?
	"	frag_color = d;\n"
	: "BANG"
	)
	,
	"}\n"

	};

	GLuint trial_prg = mk_render_program(3, 9, trial_sources);

	const GLint trial_uloc_scale = glGetUniformLocation(trial_prg, "u_scale");
	const GLint trial_uloc_canvas_tex = glGetUniformLocation(trial_prg, "u_canvas_tex");

	const GLint trial_aloc_pos    = glGetAttribLocation(trial_prg, "a_pos");
	const GLint trial_aloc_signal = glGetAttribLocation(trial_prg, "a_signal");
	const GLint trial_aloc_color  = glGetAttribLocation(trial_prg, "a_color");

	struct xoshiro256 rng;
	xoshiro256_seed(&rng, 0);

	uint16_t* vs = NULL;

	//glDisable(GL_CULL_FACE); CHKGL;
	glEnable(GL_BLEND); CHKGL;

	double best_score;
	uint16_t best_triangle[3*n_paint_elems];
	uint16_t* chosen_vs = NULL;

	int next_primitve = 1;

	uint64_t canvas_accum;
	int canvas_n_weights;
	int trial_counter;

	while (frame()) {
		if (next_primitve) {
			trial_counter = 0;
			next_primitve = 0;
			best_score = 0.0;

			glBindTexture(GL_TEXTURE_2D, canvas_tex); CHKGL;
			{
				int tw, th;
				glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &tw);
				glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &th);
				assert(tw == width);
				assert(th == height);
			}
			glGetTexImage(GL_TEXTURE_2D, 0, format, GL_UNSIGNED_SHORT, canvas_image);
			glBindTexture(GL_TEXTURE_2D, 0); CHKGL;

			uint16_t* p = canvas_image;
			canvas_accum = 0;
			uint64_t* cwp = canvas_cum_weight;
			int* cip = canvas_cum_idx;
			int pixel_index = 0;
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int s = 0;
					for (int i = 0; i < n_channels; i++) {
						// TODO weight RGB?
						uint16_t v = *(p++);
						//printf("%d\n", (int)v);
						s += v;
					}
					if (s > 0) {
						canvas_accum += s;
						*(cwp++) = canvas_accum;
						*(cip++) = pixel_index;
					}
					pixel_index++;
				}
			}
			canvas_n_weights = cwp - canvas_cum_weight;
		}

		arrsetlen(vs, 0);

		int batch_trial_index = 0;
		const int trial_batch_size = 1 << trial_batch_size_log2;
		for (batch_trial_index = 0; batch_trial_index < trial_batch_size && trial_counter < n_trials_per_primitive; batch_trial_index++, trial_counter++) {
			for (int point = 0; point < 3; point++) {
				uint64_t find = xoshiro256_next(&rng) % canvas_accum;

				int left = 0;
				int right = canvas_n_weights;
				int n_iterations = 0;
				while (left < right) {
					n_iterations++;
					int mid = (left + right) >> 1;
					uint64_t wk = canvas_cum_weight[mid];
					if (wk > find) {
						right = mid;
					} else {
						left = mid + 1;
					}
				}

				const int idx = canvas_cum_idx[right-1];
				const int px = idx % width;
				const int py = idx / width;
				uint16_t* pixel = &canvas_image[idx*n_channels];

				{
					uint16_t* v0 = arraddnptr(vs, 3+n_channels);
					uint16_t* v = v0;
					*(v++) = px;
					*(v++) = py;
					*(v++) = batch_trial_index;
					for (int i = 0; i < n_channels; i++) {
						*(v++) = pixel[i] >> 4;
					}
					assert((v-v0) == n_trial_elems);
				}
			}
		}
		const int batch_size = batch_trial_index;

		glBindFramebuffer(GL_FRAMEBUFFER, fb); CHKGL;
		glViewport(0, 0, width, height);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dummy_fb_tex, /*level=*/0); CHKGL;
		assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

		glUseProgram(trial_prg); CHKGL;

		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, signal_buf); CHKGL;
		{
			GLuint* a = glMapBufferRange(
				GL_ATOMIC_COUNTER_BUFFER,
				0, atomic_buffer_sz,
				GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT); CHKGL;
			memset(a, 0, atomic_buffer_sz);
			glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER); CHKGL;
		}

		glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, signal_buf, 0, atomic_buffer_sz); CHKGL;

		glBindBuffer(GL_ARRAY_BUFFER, trial_vbo); CHKGL;
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vs[0]) * arrlen(vs), vs); CHKGL;

		glBindVertexArray(vao); CHKGL;
		glEnableVertexAttribArray(trial_aloc_pos); CHKGL;
		glEnableVertexAttribArray(trial_aloc_signal); CHKGL;
		glEnableVertexAttribArray(trial_aloc_color); CHKGL;

		//printf("%d %d %d\n", trial_aloc_pos, trial_aloc_signal, trial_aloc_color);
		glVertexAttribPointer(trial_aloc_pos, 2, GL_UNSIGNED_SHORT, GL_TRUE, trial_stride, (void*)0); CHKGL;
		glVertexAttribPointer(trial_aloc_signal, 1, GL_UNSIGNED_SHORT, GL_FALSE, trial_stride, (void*)4); CHKGL;
		glVertexAttribPointer(trial_aloc_color,  n_channels, GL_UNSIGNED_SHORT, GL_TRUE,  trial_stride, (void*)6); CHKGL;


		glBindTexture(GL_TEXTURE_2D, canvas_tex); CHKGL;
		glUniform2f(trial_uloc_scale, 65536.0f / (float)width, 65536.0f / (float)height); CHKGL;
		glUniform1i(trial_uloc_canvas_tex, 0); CHKGL;

		//printf("%d\n", batch_size);
		glDrawArrays(GL_TRIANGLES, 0, 3*batch_size); CHKGL;

		{
			GLuint* a = glMapBufferRange(
				GL_ATOMIC_COUNTER_BUFFER,
				0, atomic_buffer_sz,
				GL_MAP_READ_BIT);

			{
				int i1 = 0;
				//printf("trials=[");
				for (int i0 = 0; i0 < ((batch_size+31)>>5); i0++) {
					for (; i1 < ((i0<<5)+32) && i1 < batch_size; i1++) {
						const int underflow = a[i0] & (1 << (i1&31));
						//printf("u_signal[%d]=%u\n", i0, a[i0]);
						//printf("%c", underflow ? '1' : '0');
					}
				}
				//printf("]\n");
			}

			{
				int i1 = 0;
				for (int i0 = 0; i0 < ((batch_size+31)>>5); i0++) {
					for (; i1 < ((i0<<5)+32) && i1 < batch_size; i1++) {
						const int underflow = a[i0] & (1 << (i1&31));
						if (underflow) continue;
						uint16_t* v0 = &vs[3*n_trial_elems*i1];
						//printf("%d\n", i1);
						uint16_t* v1 = v0;
						int color_weight = 0;
						for (int i2 = 0; i2 < 3; i2++) {
							assert(v1[2] == i1);
							v1 += n_trial_elems;
							for (int i3 = 0; i3 < n_channels; i3++) {
								// TODO gray scale weight?
								color_weight += v1[3+i3];
							}
						}

						struct triangle T = mk_triangle(
							v0[0*n_trial_elems+0], v0[0*n_trial_elems+1],
							v0[1*n_trial_elems+0], v0[1*n_trial_elems+1],
							v0[2*n_trial_elems+0], v0[2*n_trial_elems+1]);
						const double area = triangle_area(T);
						const double score = triangle_score(T);

						if (score > best_score) {
							best_score = score;
							//printf("new best %f\n", score);
							uint16_t* src = &vs[3*n_trial_elems*i1];
							uint16_t* dst = best_triangle;
							for (int i = 0; i < 3; i++) {
								*(dst++) = *(src++);
								*(dst++) = *(src++);
								src++;
								for (int j = 0; j < n_channels; j++) {
									*(dst++) = *(src++);
								}
							}
						}

					}
				}
			}

			glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER); CHKGL;
		}

		glBindTexture(GL_TEXTURE_2D, 0); CHKGL;

		glDisableVertexAttribArray(trial_aloc_color); CHKGL;
		glDisableVertexAttribArray(trial_aloc_signal); CHKGL;
		glDisableVertexAttribArray(trial_aloc_pos); CHKGL;
		glBindVertexArray(0); CHKGL;

		glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0); CHKGL;
		glUseProgram(0); CHKGL;

		glBindFramebuffer(GL_FRAMEBUFFER, 0); CHKGL;
		glViewport(0, 0, g.width, g.height);

		if (trial_counter == n_trials_per_primitive) {
			next_primitve = 1;
			if (best_score > 0) {
				uint16_t* vout = arraddnptr(chosen_vs, 3*n_paint_elems);
				memcpy(vout, best_triangle, sizeof best_triangle);
				printf("new chosen length: %zd\n", arrlen(chosen_vs)/(3*n_paint_elems));
				glBindBuffer(GL_ARRAY_BUFFER, paint_vbo); CHKGL;
				const size_t blitsz = sizeof(chosen_vs[0]) * arrlen(chosen_vs);
				assert(blitsz <= paint_vbo_sz);
				glBufferSubData(GL_ARRAY_BUFFER, 0, blitsz, chosen_vs); CHKGL;
				glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;

				// update canvas

				glBindFramebuffer(GL_FRAMEBUFFER, fb); CHKGL;
				glViewport(0, 0, width, height);
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, canvas_tex, /*level=*/0); CHKGL;
				assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
				glUseProgram(paint_prg); CHKGL;
				glBindVertexArray(vao); CHKGL;
				glBindBuffer(GL_ARRAY_BUFFER, paint_vbo); CHKGL;
				glEnableVertexAttribArray(paint_aloc_pos); CHKGL;
				glEnableVertexAttribArray(paint_aloc_color); CHKGL;

				glVertexAttribPointer(paint_aloc_pos, 2, GL_UNSIGNED_SHORT, GL_TRUE, paint_stride, (void*)0); CHKGL;
				glVertexAttribPointer(paint_aloc_color,  n_channels, GL_UNSIGNED_SHORT, GL_TRUE,  paint_stride, (void*)4); CHKGL;
				glUniform2f(paint_uloc_scale, 65536.0f / (float)width, 65536.0f / (float)height); CHKGL;

				glBlendFunc(GL_ONE, GL_ONE); CHKGL;
				glBlendEquation(GL_FUNC_REVERSE_SUBTRACT); CHKGL;

				glDrawArrays(GL_TRIANGLES, arrlen(chosen_vs)/n_paint_elems-3, 3); CHKGL;
				glBlendEquation(GL_FUNC_ADD); CHKGL;

				glDisableVertexAttribArray(paint_aloc_color); CHKGL;
				glDisableVertexAttribArray(paint_aloc_pos); CHKGL;
				glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;
				glBindVertexArray(0); CHKGL;
				glUseProgram(0); CHKGL;

				glBindFramebuffer(GL_FRAMEBUFFER, 0); CHKGL;
				glViewport(0, 0, g.width, g.height);
			}
		}

		if (arrlen(chosen_vs) > 0) {
			glUseProgram(paint_prg); CHKGL;
			glBindVertexArray(vao); CHKGL;
			glBindBuffer(GL_ARRAY_BUFFER, paint_vbo); CHKGL;
			glEnableVertexAttribArray(paint_aloc_pos); CHKGL;
			glEnableVertexAttribArray(paint_aloc_color); CHKGL;

			glVertexAttribPointer(paint_aloc_pos, 2, GL_UNSIGNED_SHORT, GL_TRUE, paint_stride, (void*)0); CHKGL;
			glVertexAttribPointer(paint_aloc_color,  n_channels, GL_UNSIGNED_SHORT, GL_TRUE,  paint_stride, (void*)4); CHKGL;
			glUniform2f(paint_uloc_scale, 65536.0f / (float)width, 65536.0f / (float)height); CHKGL;

			glBlendFunc(GL_ONE, GL_ONE); CHKGL;

			glDrawArrays(GL_TRIANGLES, 0, arrlen(chosen_vs)/n_paint_elems); CHKGL;

			glDisableVertexAttribArray(paint_aloc_color); CHKGL;
			glDisableVertexAttribArray(paint_aloc_pos); CHKGL;
			glBindBuffer(GL_ARRAY_BUFFER, 0); CHKGL;
			glBindVertexArray(0); CHKGL;
			glUseProgram(0); CHKGL;
		}

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
		mode_process(image_path);
	} else if (strcmp("view", cmd) == 0) {
		assert(!"TODO");
	} else if (strcmp("render", cmd) == 0) {
		assert(!"TODO");
	} else {
		usagef("invalid cmd \"%s\"", cmd);
	}

	return EXIT_SUCCESS;

}
