CFLAGS+=-std=c11
#CFLAGS+=-O0 -g
LDLIBS+=-lm
CFLAGS+=-O0 -g
#CFLAGS+=-O3
#CFLAGS+=-Wall

PKGS=sdl2
CFLAGS+=$(shell pkg-config --cflags ${PKGS})
LDLIBS+=$(shell pkg-config --libs ${PKGS})

OBJS=gl3w.o stb_image.o stb_ds.o

all: p0

p0: p0.o $(OBJS)

clean:
	rm -f *.o main
