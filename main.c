#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

// Тестируем на ошибку, если да - то выходим
#define TEST_CL_ERR(M) if (ret != CL_SUCCESS) { fprintf(stderr, "[ERROR] " M " errno=%d\n", ret); exit(1); }


// Переменные OpenCL
cl_device_id device_id = NULL;
cl_platform_id platform_id = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret;
cl_device_type device_type = CL_DEVICE_TYPE_CPU;

// Переменные СЛАУ
int N;
int M;
float * A; // Матрица А|B
float * A_orig; // Матрица А|B, для проверки
float * C; // Временная матрица (для пямого хода)
float * X; // Решение
float * delta_B; // Ошибка 

// Инициализируем матрицу А|B
void init_matrix() {
  A = (float *)malloc(sizeof(float) * N * M);
  A_orig = (float *)malloc(sizeof(float) * N * M);
  C = (float *)malloc(sizeof(float) * N * M);
  X = (float *)malloc(sizeof(float) * N);
  delta_B = (float *)malloc(sizeof(float) * N);

  for(int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
      *(A_orig + i * M + j) = rand() % 100 + 1;
      *(A + i * M + j) = *(A_orig + i * M + j);
      *(C + i * M + j) = 0;
    }
  }

  for(int i = 0; i < N; i++) {
    *(X + i) = 0;
    *(delta_B + i) = 0;
  }
}

// Сохраняем результаты
void save_results() {
  // printf("Input matrix\n");
  // for(int i = 0; i < N; i++) {
  //   for(int j = 0; j < M; j++) {
  //     printf("%f\t", *(A_orig + i * M + j));
  //   }
  //   printf("\n");
  // }

  // printf("Triangle\n");
  // for(int i = 0; i < N; i++) {
  //   for(int j = 0; j < M; j++) {
  //     printf("%f\t", *(A + i * M + j));
  //   }
  //   printf("\n");
  // }

  // printf("Solve\n");
  // for(int i = 0; i < N; i++) {
  //   printf("X[%d]=%f\n", i, *(X + i));
  // }
}

// Инициализируем OpenCL
void init_opencl() {
  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  TEST_CL_ERR("clGetPlatformIDs");
  ret = clGetDeviceIDs(platform_id, device_type, 1, &device_id, &ret_num_devices);
  TEST_CL_ERR("clGetDeviceIDs");
}

void free_opencl() {
  // TODO
}

// Прямой ход решения СЛАУ
void first_step() {
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_mem memobjA = NULL;
  cl_mem memobjC = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  size_t globalThreads[2] = {M, N};
  int step;

  // Загружаем исходники для прямого хода
  FILE *fp;
  char fileName[] = "./first_step.cl";
  char *source_str;
  size_t source_size;

  fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose( fp );

  context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
  TEST_CL_ERR("clCreateContext");

  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  TEST_CL_ERR("clCreateCommandQueue");
 
  memobjA = clCreateBuffer(context, CL_MEM_READ_WRITE, N * M * sizeof(float), NULL, &ret);
  TEST_CL_ERR("clCreateBuffer"); 
  memobjC = clCreateBuffer(context, CL_MEM_READ_WRITE, N * M * sizeof(float), NULL, &ret);
  TEST_CL_ERR("clCreateBuffer"); 
  ret = clEnqueueWriteBuffer(command_queue, memobjA, CL_TRUE, 0,
                            N * M * sizeof(float), A, 0, NULL, NULL);
  TEST_CL_ERR("clEnqueueWriteBuffer"); 
  ret = clEnqueueWriteBuffer(command_queue, memobjC, CL_TRUE, 0,
                            N * M * sizeof(float), C, 0, NULL, NULL);
  TEST_CL_ERR("clEnqueueWriteBuffer");
 
  program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                      (const size_t *)&source_size, &ret);
  TEST_CL_ERR("clCreateProgramWithSource");
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  TEST_CL_ERR("clBuildProgram");

  /* Create OpenCL Kernel */
  kernel = clCreateKernel(program, "first_step", &ret);
  TEST_CL_ERR("clCreateKernel");

  /* Set OpenCL Kernel Arguments */
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjA);
  TEST_CL_ERR("clSetKernelArg");
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobjC);
  TEST_CL_ERR("clSetKernelArg");
  ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&N);
  TEST_CL_ERR("clSetKernelArg");
  ret = clSetKernelArg(kernel, 4, sizeof(int), (void *)&M);
  TEST_CL_ERR("clSetKernelArg");

  for (int i = 0; i <= N - 2; i++) {
    ret = clSetKernelArg(kernel, 2, sizeof(int), (void *)&i);
    TEST_CL_ERR("clSetKernelArg");
    if (i % 2 == 0) {
      ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjA);
      ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobjC);
    } else {
      ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjC);
      ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobjA);
    }
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalThreads, NULL, NULL, 0, NULL);
    TEST_CL_ERR("clEnqueueNDRangeKernel");

    ret = clFinish(command_queue);
    TEST_CL_ERR("clFinish");
  }

  if ((N - 2) % 2 == 0) {
    ret = clEnqueueReadBuffer(command_queue, memobjC, CL_TRUE, 0,
      N * M * sizeof(float), A, 0, NULL, NULL);
    TEST_CL_ERR("clEnqueueReadBuffer");
  } else {
    ret = clEnqueueReadBuffer(command_queue, memobjA, CL_TRUE, 0,
      N * M * sizeof(float), A, 0, NULL, NULL);
    TEST_CL_ERR("clEnqueueReadBuffer");
  }

  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(memobjA);
  ret = clReleaseMemObject(memobjC);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
  free(source_str);
}

// Обраный ход решения СЛАУ
void second_step() {
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_mem memobjA = NULL;
  cl_mem memobjX = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  size_t globalThreads[2] = {N};
  int step;

  // Загружаем исходники для прямого хода
  FILE *fp;
  char fileName[] = "./second_step.cl";
  char *source_str;
  size_t source_size;

  fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose( fp );

  // OpenCL
  context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
  TEST_CL_ERR("clCreateContext");
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  TEST_CL_ERR("clCreateCommandQueue");
 
  memobjA = clCreateBuffer(context, CL_MEM_READ_WRITE, N * M * sizeof(float), NULL, &ret);
  TEST_CL_ERR("clCreateBuffer");
  memobjX = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &ret);
  TEST_CL_ERR("clCreateBuffer");
  ret = clEnqueueWriteBuffer(command_queue, memobjA, CL_TRUE, 0,
                            N * M * sizeof(float), A, 0, NULL, NULL);
  TEST_CL_ERR("clEnqueueWriteBuffer");

  program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                      (const size_t *)&source_size, &ret);
  TEST_CL_ERR("clCreateProgramWithSource");
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  TEST_CL_ERR("clBuildProgram");

  /* Create OpenCL Kernel */
  kernel = clCreateKernel(program, "second_step", &ret);
  TEST_CL_ERR("clCreateKernel");

  /* Set OpenCL Kernel Arguments */
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjA);
  TEST_CL_ERR("clSetKernelArg");
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobjX);
  TEST_CL_ERR("clSetKernelArg");
  ret = clSetKernelArg(kernel, 2, sizeof(int), (void *)&M);
  TEST_CL_ERR("clSetKernelArg");

  for (int i = N - 1; i >= 0; i--) {
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&i);
    TEST_CL_ERR("clSetKernelArg");
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, globalThreads, NULL, NULL, 0, NULL);
    TEST_CL_ERR("clEnqueueNDRangeKernel");
    ret = clFinish(command_queue);
    TEST_CL_ERR("clFinish");
  }

  ret = clEnqueueReadBuffer(command_queue, memobjX, CL_TRUE, 0,
    N * sizeof(float), X, 0, NULL, NULL);
  TEST_CL_ERR("clEnqueueReadBufferclFinish");

  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(memobjA);
  ret = clReleaseMemObject(memobjX);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
  free(source_str);
}

void validate() {
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_mem memobjA = NULL;
  cl_mem memobjX = NULL;
  cl_mem memobjB_delta = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  size_t globalThreads[2] = {N};
  int step;

  // Загружаем исходники для прямого хода
  FILE *fp;
  char fileName[] = "./validate.cl";
  char *source_str;
  size_t source_size;

  fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose( fp );

  // OpenCL
  context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
  TEST_CL_ERR("clCreateContext");
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  TEST_CL_ERR("clCreateCommandQueue");

  memobjA = clCreateBuffer(context, CL_MEM_READ_WRITE, N * M * sizeof(float), NULL, &ret);
  TEST_CL_ERR("clCreateBuffer");
  memobjX = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &ret);
  TEST_CL_ERR("clCreateBuffer");
  memobjB_delta = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &ret);
  TEST_CL_ERR("clCreateBuffer");
  ret = clEnqueueWriteBuffer(command_queue, memobjA, CL_TRUE, 0,
                            N * M * sizeof(float), A, 0, NULL, NULL);
  TEST_CL_ERR("clEnqueueWriteBuffer");
  ret = clEnqueueWriteBuffer(command_queue, memobjX, CL_TRUE, 0,
                            N * sizeof(float), X, 0, NULL, NULL);
  TEST_CL_ERR("clEnqueueWriteBuffer");

  program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                      (const size_t *)&source_size, &ret);
  TEST_CL_ERR("clCreateProgramWithSource");
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  TEST_CL_ERR("clBuildProgram");

  /* Create OpenCL Kernel */
  kernel = clCreateKernel(program, "validate", &ret);
  TEST_CL_ERR("clBuildProgram");

  /* Set OpenCL Kernel Arguments */
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjA);
  TEST_CL_ERR("clSetKernelArg");
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobjX);
  TEST_CL_ERR("clSetKernelArg");
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memobjB_delta);
  TEST_CL_ERR("clSetKernelArg");
  ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&N);
  TEST_CL_ERR("clSetKernelArg");

  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, globalThreads, NULL, NULL, 0, NULL);
  TEST_CL_ERR("clEnqueueNDRangeKernel");

  ret = clFinish(command_queue);
  TEST_CL_ERR("clFinish");

  ret = clEnqueueReadBuffer(command_queue, memobjB_delta, CL_TRUE, 0,
                          N * sizeof(float), delta_B, 0, NULL, NULL);
  TEST_CL_ERR("clEnqueueReadBuffer");

  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(memobjA);
  ret = clReleaseMemObject(memobjX);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
  free(source_str);
}

int main(int argc, char *argv[])
{
  struct timeval start, end;
  long seconds, useconds;
  double mtime;

  srand (time(NULL));

  if (argc > 1) {
    N = atoi(argv[1]);
  } else {
    N = 5;
  }
  M = N + 1;

  if (argc > 2) {
    device_type = CL_DEVICE_TYPE_GPU;
  }

  init_matrix();
  init_opencl();

  gettimeofday(&start, NULL);

  first_step();
  second_step();

  gettimeofday(&end, NULL);

  validate();

  save_results();

  free_opencl();

  float max_err = 0;
  float min_err = 0;
  float avg_err = 0;
  float B, dB, err;

  for(int i = 0; i < N; i++) {
    B = *(A + i * M + M - 1);
    dB = *(delta_B + i);
    err = fabsf(dB / B);

    if (err >= max_err) max_err = err;
    if (err < min_err) min_err = err;
    avg_err += err;
  }
  avg_err = avg_err / N;

  mtime = (end.tv_sec - start.tv_sec) * 1000.0;      // sec to ms
  mtime += (end.tv_usec - start.tv_usec) / 1000.0;   // us to ms

  printf("DeviceType=%d, N=%d, Max(dB/B)=%f, Min(dB/B)=%f, Avg(dB/B)=%f, Time=%5.6f ms\n", 
    device_type, N, max_err, min_err, avg_err, mtime);

  return 0;
}