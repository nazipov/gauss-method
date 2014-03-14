__kernel void first_step(__global float* A, __global float* C, 
    const unsigned int K, const unsigned int N, const unsigned int M) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  float T = 0;

  if (j > K) {
    T = A[K + M * j] / A[K + M * K];

    C[i + M * j] = A[i + M * j] - A[i + M * K] * T;
  } else {
    C[i + M * j] = A[i + M * j];
  }
}