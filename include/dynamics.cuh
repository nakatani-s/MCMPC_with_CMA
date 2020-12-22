/*
for calculating the state equation of dynamical system
*/
#include<math.h>
#include<cuda.h>
#include "params.cuh"

#ifndef DYNAMICS_CUH
#define DYNAMICS_CUH
__host__ __device__ void calc_Linear_example(float *state, float input, float *param, float *ret);
#endif
