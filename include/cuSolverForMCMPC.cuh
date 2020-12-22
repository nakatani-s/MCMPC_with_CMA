/*
*/
#include <stdio.h>
#include <stdlib.h>
#include<cuda.h>
#include<assert.h>
#include<cuda_runtime.h>
#include<cusolverDn.h>
#include<cublas_v2.h>
#include "params.cuh"
#include "DataStructure.cuh"

__shared__ float row[HORIZON];
__global__ void calc_Var_Cov_matrix(float *d_mat,Data1 *d_Data, float *Us_dev, int Blocks);
__global__ void pwr_matrix_answerB(float *A, float *B);
__global__ void pwr_matrix_answerA(float *A, float *B);
void get_eigen_values(float *A, float *D);