#include "MLP.h"
#include "mm_utils.h"

int main() {
    setup();
    double inputs[] = {
        1, 2,
        3, 4,
        1, 2,
        3, 4,
    };
    double outputs[8];
    MLP(inputs, outputs);
    print_array("outputs", outputs, 4, 2);
}