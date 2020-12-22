/*
initialize all param for dynamical systems and setting of MCMPC

*/

void init_sys_param(float *get_param);
void Mat_sys_A(float *a);
void init_state(float *st);
void init_Weight_matrix(float * matrix);

void init_opt( float *opt );