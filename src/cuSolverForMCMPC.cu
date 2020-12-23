/* 

*/

#include "../include/cuSolverForMCMPC.cuh"

__global__ void make_Diagonalization(float *vec, float *mat)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(threadIdx.x == blockIdx.x)
    {
        mat[id] = sqrtf(vec[threadIdx.x]);
    }else{
        mat[id] = 0.0f;
    }
}


__global__ void calc_Var_Cov_matrix(float *d_mat,Data1 *d_Data, float *Us_dev, int Blocks)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    float pows;
    //int counter = 0;
    //int denominator;
    //denominator = 1/(Blocks - 1);
    //printf("hoge::id=%d %d@ =%f $ = %f\n",id,threadIdx.x,d_Data[5].Input[threadIdx.x],Us_dev[0]);
    for(int z = 0; z < Blocks; z++)
    {
        pows +=  (d_Data[z].Input[threadIdx.x] - Us_dev[threadIdx.x]) * (d_Data[z].Input[blockIdx.x] - Us_dev[blockIdx.x]);
    }
    if(threadIdx.x == blockIdx.x && pows < 0.00001f){
        pows += (Blocks -1);
        //pows = d_mat[id];
    }
    __syncthreads();
    
    d_mat[id] = pows  /(Blocks - 1);

    //unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    //float values;
    /*if(threadIdx.x == 0 && blockIdx.x ==0)
    {
        values[threadIdx.x] = 1.0f;
    }*/
    /*if(threadIdx.x == blockIdx.x)
    {
        for(int z = 0; z < Blocks; z++)
        {
            d_mat[id] +=  (d_Data[z].Input[threadIdx.x] - Us_dev[threadIdx.x]) * (d_Data[z].Input[blockIdx.x] - Us_dev[blockIdx.x]);
        }
        //d_mat[id] = (d_Data[z].Input[threadIdx.x] - Us_dev[threadIdx.x]) * (d_Data[z].Input[blockIdx.x] - Us_dev[blockIdx.x]);;
        //values[threadIdx.x] = 1.0f;
    }else{
        d_mat[id] = 0.0f;
        //values[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    d_mat[id] = d_mat[id]  /(Blocks - 1);
    /*if(threadIdx.x == 0)
    {
       for(int i =0; i < blockDim.x; i++)
           Mat[id] = values[i];
    }  */  
}

// A * B → B
__global__ void pwr_matrix_answerB(float *A, float *B)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    int row_index, column_index;
    float pows;
    if(blockIdx.x == 0)
    {
        row_index = (int)blockDim.x * blockIdx.x;
    }else{
        row_index = ((int)blockDim.x * blockIdx.x) -1;
    }
    if(threadIdx.x == 0)
    {
        column_index = (int)blockDim.x * threadIdx.x;
    }else{
        column_index = ((int)blockDim.x * threadIdx.x) -1;
    }
    for(int k = 0; k < HORIZON; k++){
        //row[id] += A[column_index + k] * B[row_index + k];
        pows += A[column_index + k] * B[row_index + k];
    }
    __syncthreads();
    B[id] = pows;
    /*if(threadIdx.x == 0)
    {
        B[id] = row[id];
    }*/

}

// A * B →　A
__global__ void pwr_matrix_answerA(float *A, float *B)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    int row_index, column_index;
    float pows;
    if(blockIdx.x == 0)
    {
        row_index = (int)blockDim.x * blockIdx.x;
    }else{
        row_index = ((int)blockDim.x * blockIdx.x) -1;
    }
    if(threadIdx.x == 0)
    {
        column_index = (int)blockDim.x * threadIdx.x;
    }else{
        column_index = ((int)blockDim.x * threadIdx.x) -1;
    }
    for(int k = 0; k < HORIZON; k++){
        //row[id] += A[column_index + k] * B[row_index + k];
        pows += A[column_index + k] * B[row_index + k];
    }

    __syncthreads();
    A[id] = pows;
    /*if(threadIdx.x == 0)
    {
        A[id] = row[id];
    }*/

}
__global__ void tanspose(float *Out, float *In)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    int In_index = blockIdx.x + blockDim.x * threadIdx.x;

    Out[id] = In[In_index];
    __syncthreads();
}
void get_eigen_values(float *A, float *D)
{
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    const int m = HORIZON;
    const int lda = m;

    float eig_vec[m];

    float *d_A = NULL;
    float *d_W = NULL;
    int *devInfo = NULL;
    float *d_work = NULL;
    int lwork = 0;

    int info_gpu = 0;

    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(float) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_W, sizeof(double) * m);
    cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    cusolver_status = cusolverDnSsyevd_bufferSize(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_W,
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

    cusolver_status = cusolverDnSsyevd(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_W,
        d_work,
        lwork,
        devInfo);

    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(eig_vec, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(D, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    make_Diagonalization<<<HORIZON,HORIZON>>>(d_W, d_A);
    cudaMemcpy(A, d_A, sizeof(float)*lda*m, cudaMemcpyDeviceToHost);

    if (d_A    ) cudaFree(d_A);
    if (d_W    ) cudaFree(d_W);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
}
