
        #ifndef MLP_H
        #define MLP_H
        int kernel_0_shape[2];
double kernel_0[10];

int bias_0_shape[2];
double bias_0[10];

int xA_0_shape[2];
double xA_0[1280];

int preactivation_1_shape[2];
double preactivation_1[1280];

int kernel_1_shape[2];
double kernel_1[100];

int bias_1_shape[2];
double bias_1[10];

int xA_1_shape[2];
double xA_1[1280];

int preactivation_2_shape[2];
double preactivation_2[1280];

int kernel_2_shape[2];
double kernel_2[10];

int bias_2_shape[2];
double bias_2[1];

int xA_2_shape[2];
double xA_2[128];

void setup();

        
        void MLP(
            const double *preactivation_0,
            double *preactivation_3
        );
        
        #endif //MLP_H
        