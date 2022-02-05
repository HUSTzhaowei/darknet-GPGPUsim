#ifndef ZW_GEMM_H
#define ZW_GEMM_H
#ifdef GPU
void mysgemm(int TA, int TB, int M, int N, int K, float alpha, 
float *A, float *B, float beta, float *C);
#endif
#endif