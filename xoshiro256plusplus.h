#ifndef _XOSHIRO256PLUSPLUS_H_

#include <stdint.h>

struct xoshiro256 {
	uint64_t s[4];
};

static inline uint64_t _xoshiro256_rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

static inline void xoshiro256_seed(struct xoshiro256* x, uint64_t seed)
{
	for (int i = 0; i < 4; i++) {
		uint64_t z = (seed += 0x9e3779b97f4a7c15);
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		x->s[i] = z ^ (z >> 31);
	}

	// This is a fixed-increment version of Java 8's SplittableRandom
	// generator. See http://dx.doi.org/10.1145/2714064.2660195 and 
	// http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html
	// It is a very fast generator passing BigCrush, and it can be useful if
	// for some reason you absolutely want 64 bits of state.
}

static inline uint64_t xoshiro256_next(struct xoshiro256* x)
{
	const uint64_t result = _xoshiro256_rotl(x->s[0] + x->s[3], 23) + x->s[0];

	const uint64_t t = x->s[1] << 17;

	x->s[2] ^= x->s[0];
	x->s[3] ^= x->s[1];
	x->s[1] ^= x->s[2];
	x->s[0] ^= x->s[3];

	x->s[2] ^= t;

	x->s[3] = _xoshiro256_rotl(x->s[3], 45);

	return result;
}

// modified xoshiro source code above, original text below:

/*  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

// This is xoshiro256++ 1.0, one of our all-purpose, rock-solid generators.
// It has excellent (sub-ns) speed, a state (256 bits) that is large
// enough for any parallel application, and it passes all tests we are
// aware of.

// For generating just floating-point numbers, xoshiro256+ is even faster.

// The state must be seeded so that it is not everywhere zero. If you have
// a 64-bit seed, we suggest to seed a splitmix64 generator and use its
// output to fill s. */

#define _XOSHIRO256PLUSPLUS_H_
#endif
