
        #ifndef MLP_H
        #define MLP_H
        int kernel_0_shape[2];
double kernel_0[6];

int bias_0_shape[2];
double bias_0[3];

int xA_0_shape[2];
double xA_0[12];

int preactivation_1_shape[2];
double preactivation_1[12];

int kernel_1_shape[2];
double kernel_1[3];

int bias_1_shape[2];
double bias_1[1];

int xA_1_shape[2];
double xA_1[4];

void setup();

        
        void MLP(
            const double *preactivation_0,
            double *preactivation_2
        );
        
        #endif //MLP_H
        