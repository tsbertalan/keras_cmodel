
#include "MLP.h"
#include "mm_utils.h"

void setup()
{
    kernel_0_shape[0] = 2;
    kernel_0_shape[1] = 3;
    ;
    kernel_0[0] = -0.5934772491455078125;
    kernel_0[1] = -0.3144419193267822265625;
    kernel_0[2] = -0.062902756035327911376953125;
    kernel_0[3] = -0.45576083660125732421875;
    kernel_0[4] = -0.900450050830841064453125;
    kernel_0[5] = -1.04640471935272216796875;
    ;

    bias_0_shape[0] = 1;
    bias_0_shape[1] = 3;
    ;
    bias_0[0] = 0.00317639834247529506683349609375;
    bias_0[1] = 0.0031798654235899448394775390625;
    bias_0[2] = 0.003179051913321018218994140625;
    ;

    xA_0_shape[0] = 4;
    xA_0_shape[1] = 3;
    ;
    xA_0[0] = 0;
    xA_0[1] = 0;
    xA_0[2] = 0;
    xA_0[3] = 0;
    xA_0[4] = 0;
    xA_0[5] = 0;
    xA_0[6] = 0;
    xA_0[7] = 0;
    xA_0[8] = 0;
    xA_0[9] = 0;
    xA_0[10] = 0;
    xA_0[11] = 0;
    ;

    preactivation_1_shape[0] = 4;
    preactivation_1_shape[1] = 3;
    ;
    preactivation_1[0] = 0;
    preactivation_1[1] = 0;
    preactivation_1[2] = 0;
    preactivation_1[3] = 0;
    preactivation_1[4] = 0;
    preactivation_1[5] = 0;
    preactivation_1[6] = 0;
    preactivation_1[7] = 0;
    preactivation_1[8] = 0;
    preactivation_1[9] = 0;
    preactivation_1[10] = 0;
    preactivation_1[11] = 0;
    ;

    kernel_1_shape[0] = 3;
    kernel_1_shape[1] = 1;
    ;
    kernel_1[0] = -0.2823113501071929931640625;
    kernel_1[1] = -1.10363280773162841796875;
    kernel_1[2] = -1.20179641246795654296875;
    ;

    bias_1_shape[0] = 1;
    bias_1_shape[1] = 1;
    ;
    bias_1[0] = -0.00317725609056651592254638671875;
    ;

    xA_1_shape[0] = 4;
    xA_1_shape[1] = 1;
    ;
    xA_1[0] = 0;
    xA_1[1] = 0;
    xA_1[2] = 0;
    xA_1[3] = 0;
    ;
}

void MLP(
    const double *preactivation_0,
    double *preactivation_2)
{
    const int preactivation_0_shape[2] = {4, 2};
    const int preactivation_2_shape[2] = {4, 1};

    mmult(
        preactivation_0, preactivation_0_shape[0], preactivation_0_shape[1],
        kernel_0, kernel_0_shape[0], kernel_0_shape[1],
        xA_0);
    bias_add(
        xA_0, xA_0_shape[0], xA_0_shape[1],
        bias_0,
        preactivation_1);
    activate(
        preactivation_1,
        preactivation_1_shape[0],
        preactivation_1_shape[1],
        1);

    mmult(
        preactivation_1, preactivation_1_shape[0], preactivation_1_shape[1],
        kernel_1, kernel_1_shape[0], kernel_1_shape[1],
        xA_1);
    bias_add(
        xA_1, xA_1_shape[0], xA_1_shape[1],
        bias_1,
        preactivation_2);
    activate(
        preactivation_2,
        preactivation_2_shape[0],
        preactivation_2_shape[1],
        2);
}
