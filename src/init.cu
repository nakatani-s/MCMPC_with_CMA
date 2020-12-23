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

    /*a[0] = -1.0f;
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
    a[11] = 1.0f;*/

    a[0] = 0.0f;
    a[1] = 1.0f;
    a[2] = 0.0f;
    a[3] = 0.0f;
    a[4] = -1.1364f;
    a[5] = 0.2273f;
    a[6] = 0.0f;
    a[7] = -0.1339f;
    a[8] = -0.1071f;

    a[9] = 0.0f;
    a[10] = 0.0f;
    a[11] = 0.0893f;
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

    st[0] = 2.98f;
    st[1] = 0.7f;
    st[3] = 0.0f;
}

void init_Weight_matrix(float * matrix)
{
    /*matrix[0] = 1.0f;
    matrix[1] = 0.0f;
    matrix[2] = 0.0f;
    matrix[3] = 0.0f;
    matrix[4] = 1.0f;

    matrix[5] = 1.0f;*/
    matrix[0] = 2.0f;
    matrix[1] = 1.0f;
    matrix[2] = 0.1f;
    matrix[3] = 1.0f;
}

void init_opt( float *opt )
{
    /*opt[0] = -3.6009f;
    opt[1] = 1.0275f;
    opt[2] = 4.1634f;
    opt[3] = -2.7379f;
    opt[4] = 1.1454f;
    opt[5] = -0.2345f;
    opt[6] = 0.00768f;
    opt[7] = 0.1061f;
    opt[8] = -0.05222f;
    opt[9] = -0.0335f;*/
    opt[0] = -2.69f;
    opt[1] = 2.3787f;
    opt[2] = -2.0953f;
    opt[3] = 1.8364f;
    opt[4] = -1.5989f;
    opt[5] = 1.38f;
    opt[6] = -1.1772f;
    opt[7] = 0.9881f;
    opt[8] = -0.8106f;
    opt[9] = 0.6424f;
    opt[10] = -0.4818f;
    opt[11] = 0.327f;
    opt[12] = -0.1778f;
    opt[13] = 0.0428f;
    opt[14] = 0.0027f;
}
