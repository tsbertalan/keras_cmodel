#ifndef MM_UTILS_H
#define MM_UTILS_H
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "mm_utils.h"

void zero_array(double *A, const int rows, const int cols);

void mmult(
    const double *A, const int rA, const int cA,
    const double *B, const int rB, const int cB,
    double *O // O must have shape [rA, cB], or we'll segfault.
);

void bias_add(
    const double *xA, const int rows, const int cols,
    const double *b,
    double *O // O and b both must be shape [rows, cols].
);

void activate(double *mat, const int rows, const int cols, const short act_type);

void print_array(char *name, const double *A, int rows, int cols);

int test();
#endif // MM_UTILS_H