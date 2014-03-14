__kernel void validate(__global float* A, __global float* X, __global float* B_delta, const unsigned int N) 
{
  int j = get_global_id(0);
  float value = 0;

  for (int k = 0; k < N; k++) {
    value = value + A[k + j * (N + 1)] * X[k];
  }
  B_delta[j] = A[N + j * (N + 1)] - value;
}