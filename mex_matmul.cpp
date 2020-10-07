/************************************************************************
This file is part of nothing.
Copyright (c) 2020 Pierre.
Author: Pierre

"mex_matmul" is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Intersect is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Intersect.  If not, see <https://www.gnu.org/licenses/>.
************************************************************************/

// compile with : mex mex_matmul.cpp -D"CXXFLAGS -std=c++11" -I"/usr/local/cuda-8.0/include/" -L"/usr/local/cuda-8.0/lib64/" -lstdc++ -lcudart -lcublas
// test with : a = rand(10000, 12000); b = rand(12000, 11000); c = mex_matmul(a, b);


#include "mex.h"
#include <assert.h>
#include <time.h>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
using namespace std;

typedef std::chrono::high_resolution_clock myclock;
typedef myclock::time_point timepoint;
using namespace std::chrono;

#include <cublas_v2.h>
//~ #include <curand.h>
#include <cuda_runtime.h>


void gpuCublasMmul(cublasHandle_t &handle,const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    // do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


void do_processing(int m, int k, int n, float *A, float *B, float *C) {
	
	// Allocate memory for each vector on GPU
	float *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, sizeof(float)*m*k);
	cudaMalloc(&d_b, sizeof(float)*k*n);
	cudaMalloc(&d_c, sizeof(float)*m*n);
	
	cudaMemcpy(d_a, A, sizeof(float)*m*k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, sizeof(float)*k*n, cudaMemcpyHostToDevice);
	
	cublasHandle_t handle;
	cublasCreate(&handle);
	gpuCublasMmul(handle, d_a, d_b, d_c, m, k, n); // nr_rows_A, nr_cols_A, nr_cols_B
	cudaDeviceSynchronize();
	
	cudaMemcpy(C, d_c, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cublasDestroy(handle);
	
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	timepoint t1, t2; duration<double> time_span;
	t1 = myclock::now();

	int nrA = mxGetM(prhs[0]);
	int ncA = mxGetN(prhs[0]);
	int nrB = mxGetM(prhs[1]);
	int ncB = mxGetN(prhs[1]);
	assert(ncA == nrB);
	
	float *A, *B, *C;
	A = new float[nrA*ncA];
	B = new float[nrB*ncB];
	
	if (mxIsDouble(prhs[0])) {
		//~ mexPrintf("Data type is double\n");
		
		double *ptr_A = (double *) mxGetData(prhs[0]);
		double *ptr_B = (double *) mxGetData(prhs[1]);		
		for (int j=0; j < nrA*ncA; j++) A[j] = (float)ptr_A[j];
		for (int j=0; j < nrB*ncB; j++) B[j] = (float)ptr_B[j];
		//~ std::transform(ptr_A, ptr_A + nrA*ncA, A, [](double x) { return (float)x; });
		//~ std::transform(ptr_B, ptr_B + nrB*ncB, B, [](double x) { return (float)x; });
		
	} else {
		
		if (mxIsSingle(prhs[0])) {
			float *ptr_A = (float *) mxGetData(prhs[0]);
			float *ptr_B = (float *) mxGetData(prhs[1]);
			for (int j=0; j < nrA*ncA; j++) A[j] = (float)ptr_A[j]; // not continuous, so, no direct use possible, and no possible copy with memcpy
			for (int j=0; j < nrB*ncB; j++) B[j] = (float)ptr_B[j];
		}
		
	}
	
	C = new float[nrA*ncB];
	do_processing(nrA, ncA, ncB, A, B, C);

	mxArray *res = mxCreateDoubleMatrix(nrA, ncB, mxREAL);
	double *ptr_C = (double *) mxGetData(res);
	for (int j=0; j < nrA*ncB; j++) ptr_C[j] = (double)C[j];
	//~ std::transform(ptr_C, ptr_C + nrA*ncB, C, [](double x) { return (float)x; });
	plhs[0] = res;
	
	delete[] A;
	delete[] B;
	delete[] C;
	
	t2 = myclock::now(); time_span = t2-t1; mexPrintf("Elapsed time = %f s\n", time_span.count());
	
}
