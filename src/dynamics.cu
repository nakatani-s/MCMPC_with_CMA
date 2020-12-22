/*
only include "../include/dynamics.cuh"
*/
#include "../include/dynamics.cuh"

__host__ __device__ void calc_Linear_example(float *state, float input, float *param, float *ret)
{
    float temp[dim_state];
    temp[0] = param[0] * state[0] + param[1] * state[1] + param[2] * state[2] + param[9] * input;
    temp[1] = param[3] * state[0] + param[4] * state[1] + param[5] * state[2] + param[10] * input;
    temp[2] = param[6] * state[0] + param[7] * state[1] + param[8] * state[2] + param[11] * input;
    
    for(int d = 0; d < dim_state; d++){
        ret[d] = temp[d];
    }
}