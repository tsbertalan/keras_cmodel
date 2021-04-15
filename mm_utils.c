#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "mm_utils.h"

void zero_array(double *A, const int rows, const int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            A[i * cols + j] = 0;
        }
    }
}

void mmult(
    const double *A, const int rA, const int cA,
    const double *B, const int rB, const int cB,
    double *O // O must have shape [rA, cB], or we'll segfault.
)
{
    // printf("(assertion: %d == %d)\n", cA, rB);
    assert(cA == rB); // We don't actually need rB after this.
    zero_array(O, rA, cB);
    int i, j, k;
    for (i = 0; i < rA; i++)
    {
        for (j = 0; j < cB; j++)
        {
            for (k = 0; k < cA; k++)
            {
                O[i * cB + j] += A[i * cA + k] * B[k * cB + j];
            }
        }
    }
}

void bias_add(
    const double *xA, const int rows, const int cols,
    const double *b,
    double *O // O and b both must be shape [rows, cols].
)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            O[i * rows + j] = xA[i * cols + j] + b[0 * cols + j];
        }
    }
}

void activate(double *mat, const int rows, const int cols, const short act_type)
{
    int i, j;
    double x;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            x = mat[i * cols + j];
            if (act_type == 0)
            {
                mat[i * cols + j] = x > 0 ? x : 0;
            }
            else if (act_type == 1)
            {
                mat[i * cols + j] = tanh(x);
            }
            // else assume linear/none
        }
    }
}

void print_array(char *name, const double *A, int rows, int cols)
{
    printf("%s", name);
    printf(" = [\n");
    for (int i = 0; i < rows; i++)
    {
        printf(" [");
        for (int j = 0; j < cols; j++)
        {
            printf("%f, ", A[i * cols + j]);
        }
        printf("]\n");
    }
    printf("];\n\n");
}

#define TEST_BATCH_SIZE 3
#define TEST_INPUT_SIZE 2
#define TEST_OUTPUT_SIZE 3

int test()
{

    const int x_shape[2] = {TEST_BATCH_SIZE, TEST_INPUT_SIZE};
    const double x[TEST_BATCH_SIZE * TEST_INPUT_SIZE] = {1, 2, 3, 4, 5, 6};

    const int A_shape[2] = {TEST_INPUT_SIZE, TEST_OUTPUT_SIZE};
    const double A[TEST_INPUT_SIZE * TEST_OUTPUT_SIZE] = {1, 2, 3, 4, 5, 6};

    const int xA_shape[2] = {TEST_BATCH_SIZE, TEST_OUTPUT_SIZE};
    double xA[TEST_BATCH_SIZE * TEST_OUTPUT_SIZE] = {0, 0, 0};

    const int b_shape[2] = {1, TEST_OUTPUT_SIZE};
    const double b[1 * TEST_OUTPUT_SIZE] = {-1, -1, -1};

    const int result_shape[2] = {TEST_BATCH_SIZE, TEST_OUTPUT_SIZE};
    double result[TEST_BATCH_SIZE * TEST_OUTPUT_SIZE] = {0, 0, 0};

    print_array("x", x, x_shape[0], x_shape[1]);
    print_array("A", A, A_shape[0], A_shape[1]);

    mmult(
        x, x_shape[0], x_shape[1],
        A, A_shape[0], A_shape[1],
        xA);

    print_array("x * A", xA, xA_shape[0], xA_shape[1]);

    print_array("b", b, b_shape[0], b_shape[1]);

    bias_add(xA, xA_shape[0], xA_shape[1], b, result);

    print_array("x * A + b", result, result_shape[0], result_shape[1]);
}