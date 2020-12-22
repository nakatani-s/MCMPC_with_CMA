/* 
    MCMPC.cuh
*/
#include<cuda.h>
#include "params.cuh"
#include "DataStructure.cuh"
#include "dynamics.cuh"


__constant__ float  d_constraints[NUM_CONST], d_matrix[dim_weight_matrix]/*, d_hat_Q[dim_hat_Q], d_param[dim_param]*/;
__shared__ float W_comp[THREAD_PER_BLOCKS], L_comp[THREAD_PER_BLOCKS], values[HORIZON];
__shared__ int best_thread_id_this_block;

// #ifndef MCMPC_CUH
// #define MCMPC_CUH
__global__ void setup_kernel(curandState *state,int seed);
__global__ void setup_init_Covariance(float *Mat);
unsigned int countBlocks(unsigned int a, unsigned int b);
void weighted_mean(Data1 *h_Data, int Blocks, float *Us_host);
//__global__ void MCMPC_GPU_Linear_Example(float *state, curandState *devs, Data1 *d_Datas, float var, int Blocks, float *d_cov);
//__global__ void MCMPC_GPU_Linear_Example(float x, float y, float w, curandState *devs, Data1 *d_Datas, float var, int Blocks, float *d_cov);
__global__ void MCMPC_GPU_Linear_Example(float x, float y, float w, curandState *devs, Data1 *d_Datas, float var, int Blocks, float *d_cov, float *d_param, float *d_matrix);
// #endif
