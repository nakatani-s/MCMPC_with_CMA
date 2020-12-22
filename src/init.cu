/*
#include "../include/~.cuh"
*/ 
#include <math.h>

// include header files described by editter
#include "../include/init.cuh"

void Mat_sys_A(float *a)
{
    /*a[0] = 0.0f;
    a[1] = -1.6658f;
    a[2] = -11.9340f;
    a[3] = 3.5377e-8;
    a[4] = 0.0f;
    a[5] = 43.0344f;
    a[6] = 44.7524f;
    a[7] = -9.1392e-5;

    a[8] = 9.434f;
    a[9] = -35.3774f;*/

    a[0] = -1.0f;
    a[1] = 1.0f;
    a[2] = -1.0f;
    a[3] = 1.0f;
    a[4] = 0.0f;
    a[5] = 3.0f;
    a[6] = 1.0f;
    a[7] = 0.0f;
    a[8] = -1.0f;

    a[9] = 1.0f;
    a[10] = 2.0f;
    a[11] = 1.0f;
}

void init_state(float *st)
{
    // float st[8];
    /*st[0] = 0.5; //cart_position
    st[1] = 0.01; // Theta_1
    st[2] = 0.01; // Theta_2
    st[3] = 0.01; // Theta_3
    st[4] = 0.0f; //cart_velocity
    st[5] = 0.0f; // dTheta_1
    st[6] = 0.0f; // dTheta_2
    st[7] = 0.0f; // dTheta_3 */

    /*st[0] = 0.0f;
    st[1] = M_PI;
    st[2] = 0.0f;
    st[3] = 0.0f;*/

    st[0] = 1.0f;
    st[1] = 1.0f;
    st[3] = -1.0f;
}

void init_Weight_matrix(float * matrix)
{
    matrix[0] = 1.0f;
    matrix[1] = -1.0f;
    matrix[2] = 0.0f;
    matrix[3] = -1.0f;
    matrix[4] = 1.0f;

    matrix[5] = 1.0f;
}

void init_opt( float *opt )
{
    opt[0] = -3.6521f;
    opt[1] = 0.84605f;
    opt[2] = 4.03018f;
    opt[3] = -2.5105f;
    opt[4] = 1.45366f;
    opt[5] = -0.55267f;
    opt[6] = 0.13774f;
    opt[7] = 0.02762f;
    opt[8] = -0.07211f;
    opt[9] = -0.01287f;
}