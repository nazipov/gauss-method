__kernel void second_step(__global float* A, __global float* X, 
    int M, int K) 
{
  int i = get_global_id(0);
  float T = 0;

  T = A[(M - 1) + M * K] / A[K + M * K];
  if (i == K) {
    X[K] = T;
  } else {
    A[(M - 1) + M * i] = A[(M - 1) + M * i] - A[K + M * i] * T;
  }
}