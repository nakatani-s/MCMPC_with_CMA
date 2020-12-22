/*
include "MCMPC.cuh"
*/
#include<stdio.h>
#include "../include/MCMPC.cuh" 

__global__ void setup_kernel(curandState *state,int seed) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, id, 0, &state[id]);
}

unsigned int countBlocks(unsigned int a, unsigned int b) {
	unsigned int num;
	num = a / b;
	if (a < b || a % b > 0)
		num++;
	return num;
}

void weighted_mean(Data1 *h_Data, int Blocks, float *Us_host)
{
    float total_weight = 0.0f;
    float temp[HORIZON] = {};
    for(int i = 0; i < Blocks; i++){
        if(isnan(h_Data[i].W))
        {
            total_weight += 0.0f;
        }else{
            total_weight += h_Data[i].W;
        }
    }

    for(int i = 0; i < HORIZON; i++)
    {
        for(int k = 0; k < Blocks; k++)
        {
            if(isnan(h_Data[k].W))
            {
                temp[i] += 0.0f;
            }else{
                temp[i] += h_Data[k].W * h_Data[k].Input[i] / total_weight;
            }
            Us_host[i] = temp[i]; 
        }
    }
}

__device__ float generate_u(int t, float mean, float var, float *d_cov, float *z)
{
    int count_index;
    count_index = t * HORIZON;
    float ret, sec_term;
    for(int k = 0; k < HORIZON; k++)
    {
        sec_term += d_cov[count_index + k]*z[k];
    }
    ret = mean + var * sec_term;
    return ret;
}

__device__ float gen_u(unsigned int id, curandState *state, float ave, float vr) {
    float u;
    curandState localState = state[id];
    u = curand_normal(&localState) * vr + ave;
    return u;
}

__global__ void setup_init_Covariance(float *Mat)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    //float values;
    /*if(threadIdx.x == 0 && blockIdx.x ==0)
    {
        values[threadIdx.x] = 1.0f;
    }*/
    if(threadIdx.x == blockIdx.x)
    {
        Mat[id] = 1.0f;
        values[threadIdx.x] = 1.0f;
    }else{
        Mat[id] = 0.0f;
        values[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    /*if(threadIdx.x == 0)
    {
       for(int i =0; i < blockDim.x; i++)
           Mat[id] = values[i];
    }  */   
    
}

__global__ void MCMPC_GPU_Linear_Example(float x, float y, float w, curandState *devs, Data1 *d_Datas, float var, int Blocks, float *d_cov, float *d_param, float *d_matrix)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seq;
    seq = id;
    float qx = 0.0f;
    float total_cost = 0.0f;
    float u[HORIZON]= { };
    float block_var;
    // int Powers;
    //printf("hoge id=%d\n", id);
    float d_state_here[dim_state] = {x,y,w};
    float get_state[dim_state] = {};
    float z[HORIZON] = { };

    for(int t = 0; t < HORIZON; t++)
    {
        block_var = var;
        for(int t_x = 0; t_x < HORIZON; t_x++)
        {
            z[t_x] = gen_u(seq, devs, 0, 1.0f);
            seq += 3;
        }
        u[t] = generate_u(t, d_Datas[0].Input[t], var, d_cov, z);
        if(u[t]<-4.0f){
           u[t] = -4.0f;
        }
        if(u[t] > 4.0f){
           u[t] = 4.0f;
        }
        //printf("hoge id=%d @ %f %f\n", id, u[t], z[t]);
        calc_Linear_example(d_state_here, u[t], d_param, get_state);
        //printf("hoge id=%d @ %f %f %f\n", id, u[t], d_param[0], get_state[1]);
        qx += d_matrix[0] * get_state[0] * get_state[0];
        qx += d_matrix[1] * get_state[0] * get_state[1];
        qx += d_matrix[3] * get_state[0] * get_state[1];
        qx += d_matrix[4] * get_state[1] * get_state[1];
        for(int h = 0; h < dim_state; h++){
           d_state_here[h] = get_state[h];
        }
        
        total_cost += qx;

        qx = 0.0f;
    }

    float KL_COST, S, lambda;
    lambda = HORIZON * dim_state;
    S = total_cost / lambda;
    KL_COST = exp(-S);
    W_comp[threadIdx.x] = KL_COST;
    L_comp[threadIdx.x] = total_cost;
    __syncthreads();
    if(threadIdx.x == 0)
    {
        best_thread_id_this_block = 0;
        for(int y = 1; y < blockDim.x; y++){
            if(L_comp[y] < L_comp[best_thread_id_this_block])
            {
                best_thread_id_this_block = y;
            }
        }
    }
    __syncthreads();
    if(threadIdx.x == best_thread_id_this_block)
    {
        Data1 block_best;
        block_best.L = L_comp[best_thread_id_this_block];
        block_best.W = W_comp[best_thread_id_this_block];
        for(int z = 0; z < HORIZON; z++)
        {
            block_best.Input[z] = u[z];
        }
        d_Datas[blockIdx.x] = block_best;

    } 
}
