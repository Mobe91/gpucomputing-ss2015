#include <cuda_runtime.h>
#include "SOM.h"

__host__ void updateSOMGPU(SOM &som, float *inputDescriptor, int indexOfBMU);