/*
define parameters for MCMPC and Systems
*/

#include <math.h>

#ifndef PARAMS_CUH
#define PARAMS_CUH

#define TIME 1
// #define dim_param 10
#define dim_param 12
#define dim_state 3
#define dim_U 1

#define dim_weight_matrix 5
#define NUM_CONST 4
#define interval 0.01
#define Variavility 1.0

#define Recalc 1
#define N_OF_SAMPLES 3000
#define THREAD_PER_BLOCKS 50
#define HORIZON 10

const int dim_hat_Q = HORIZON * HORIZON;

#endif
