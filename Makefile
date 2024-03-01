CFLAGS+=-std=c11
#CFLAGS+=-O0 -g
LDLIBS+=-lm
CFLAGS+=-O0 -g
#CFLAGS+=-O3
#CFLAGS+=-Wall

PKGS=sdl2
CFLAGS+=$(shell pkg-config --cflags ${PKGS})
LDLIBS+=$(shell pkg-config --libs ${PKGS})

all: main

main: main.o gl3w.o stb_image.o stb_ds.o

clean:
	rm -f *.o main
