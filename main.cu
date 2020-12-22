#include<iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>

#include "include/params.cuh"
#include "include/DataStructure.cuh"
#include "include/MCMPC.cuh"
#include "include/init.cuh"
#include "include/cuSolverForMCMPC.cuh"

#define Linear

int main(int argc, char **argv)
{
    /*データ書き込みファイルの定義*/
    FILE *fp;
    time_t timeValue;
    struct tm *timeObject;
    time( &timeValue );
    timeObject = localtime( &timeValue );
    char filename1[35];
    sprintf(filename1,"data_system_%d%d_%d%d.txt",timeObject->tm_mon + 1, timeObject->tm_mday, timeObject->tm_hour,timeObject->tm_min);
    fp = fopen(filename1,"w");


    float params[dim_param], state[dim_state], /*h_constraint[NUM_CONST],*/ h_matrix[dim_weight_matrix];
    Mat_sys_A( params );
    init_state( state );
    // init_constraint( h_constraint );
    init_Weight_matrix( h_matrix );
    cudaMemcpyToSymbol(d_param, &params, dim_param * sizeof(float));
    cudaMemcpyToSymbol(d_matrix, h_matrix, dim_weight_matrix * sizeof(float));

#ifdef Linear
    float opt[HORIZON], Error[HORIZON];
    init_opt( opt );
#endif


    /* GPUの設定 */
    unsigned int numBlocks, randomBlocks, randomNums/*, minId_cpu*/;
    int Blocks;
    randomNums = N_OF_SAMPLES * dim_U * HORIZON;
    randomBlocks = countBlocks(randomNums, THREAD_PER_BLOCKS);
    numBlocks = countBlocks(N_OF_SAMPLES, THREAD_PER_BLOCKS);
    printf("#NumBlocks = %d\n", numBlocks);
    Blocks = numBlocks;

    /* CPU to GPU dataExchanger */
    Data1 *h_dataFromBlocks;
    Data1 *d_dataFromBlocks;

    h_dataFromBlocks = (Data1 *)malloc(sizeof(Data1)*numBlocks);
    cudaMalloc(&d_dataFromBlocks, sizeof(Data1) * numBlocks);

    /* curand の設定 */
    curandState *devStates;
    cudaMalloc((void **)&devStates, randomNums * sizeof(curandState));
    setup_kernel<<<randomBlocks, THREAD_PER_BLOCKS>>>(devStates,rand());
    cudaDeviceSynchronize();

    /* Covariance の定義 */
    float *h_hat_Q, *Diag_D;
    float *device_cov;
    float *device_diag_eig = NULL;
    h_hat_Q = (float *)malloc(sizeof(float)*dim_hat_Q);
    Diag_D = (float *)malloc(sizeof(float)*dim_hat_Q);
    cudaMalloc(&device_cov, sizeof(float)*dim_hat_Q);
    cudaMalloc(&device_diag_eig, sizeof(float)*dim_hat_Q);
    /*cudaMalloc(&d_hat_Q, sizeof(float)*dim_hat_Q);*/

    setup_init_Covariance<<<HORIZON, HORIZON>>>(d_hat_Q);

    /* 準最適制御入力列 */
    float *Us_host, *Us_device;
    Us_host = (float *)malloc(sizeof(float) * HORIZON);
    for(int i = 0; i < HORIZON; i++){
        Us_host[i] = 0.0f;
    }
    cudaMalloc(&Us_device, sizeof(float) * HORIZON);


    float var;
    float now_u;
    for(int i = 0; i < Blocks; i++){
        for(int k = 0; k < HORIZON; k++){
            h_dataFromBlocks[i].Input[k] = 0.0f;
        }
    }

    for(int time = 0; time < TIME; time++){
        for(int repeat = 0; repeat < Recalc; repeat++){
            var = Variavility;
            cudaMemcpy(d_dataFromBlocks, h_dataFromBlocks, sizeof(Data1)*numBlocks, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            // MCMPC_GPU<<<numBlocks, THREAD_PER_BLOCKS>>>(state, devStates, d_dataFromBlocks, var, Blocks, d_hat_Q);
            MCMPC_GPU_Linear_Example<<<numBlocks, THREAD_PER_BLOCKS>>>(state, devStates, d_dataFromBlocks, var, Blocks, d_hat_Q);
            cudaDeviceSynchronize();
            cudaMemcpy(h_dataFromBlocks, d_dataFromBlocks, sizeof(Data1) * numBlocks, cudaMemcpyDeviceToHost);
            weighted_mean(h_dataFromBlocks, Blocks, Us_host);
            cudaMemcpy(Us_device, Us_host, sizeof(float) * HORIZON, cudaMemcpyHostToDevice);
            calc_Var_Cov_matrix<<<HORIZON, HORIZON>>>(device_cov, d_dataFromBlocks, Us_device, Blocks);
            cudaDeviceSynchronize();
            cudaMemcpy(h_hat_Q, device_cov, sizeof(float)*dim_hat_Q, cudaMemcpyDeviceToHost);
            get_eigen_values(h_hat_Q, Diag_D);
            cudaMemcpy(device_diag_eig, h_hat_Q, sizeof(float)*dim_hat_Q, cudaMemcpyHostToDevice);
            cudaMemcpy(device_cov, Diag_D, sizeof(float)*dim_hat_Q, cudaMemcpyHostToDevice);
            pwr_matrix_answerB<<<HORIZON,HORIZON>>>(device_cov, device_diag_eig);
            cudaDeviceSynchronize();
            pwr_matrix_answerA<<<HORIZON,HORIZON>>>(device_diag_eig, device_cov);
            cudaDeviceSynchronize();
            cudaMemcpy(d_hat_Q, device_diag_eig, sizeof(float)*dim_hat_Q, cudaMemcpyDeviceToHost);

            fprintf(fp,"%f %f %f %f %f %f %f %f %f %f\n",Us_host[0], Us_host[1],
                    Us_host[2], Us_host[3], Us_host[4], Us_host[5], Us_host[6], Us_host[7], Us_host[8], Us_host[9]);

#ifdef Linear
            float RSME;
            for(int d = 0; d < HORIZON; d++){
                Error[d] = Us_host[d] - opt[d];
                RSME += powf(Error[d],2);
            }
            printf("RSME == %f\n", RSME / HORIZON);
#endif
        }
        now_u = Us_host[0];
        calc_Linear_example(state, now_u, params, state);
        for(int i = 0; i < Blocks; i++){
            for(int k = 0; k < HORIZON - 1; k++){
                h_dataFromBlocks[i].Input[k] = Us_host[k+1];
            }
            h_dataFromBlocks[i].Input[HORIZON-1] = Us_host[HORIZON - 1];
        }
    }
    fclose(fp);
    // fclose(hp);
    cudaDeviceReset();
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}