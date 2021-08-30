#pragma once

#include <immintrin.h>
#include <vector>

#include "sparse_tensor.h"

void sparse_gemm_bsc_f32(
    int M, int N, int K,
    const float* A,
    const float* B,
    const int64_t* rowidxs,
    const int64_t* colptr,
    int ncolptr,
    int blocksize[2],
    float* C,
    bool with_relu    
){
  int M_NBLK = 4;
  assert(M % M_NBLK == 0);
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);
  __m512 d = _mm512_setzero_ps();
  #pragma omp parallel for collapse(2)   
  for (int mb = 0; mb < M / M_NBLK; mb++){
      for (int b_col = 0; b_col < ncolptr-1; b_col++) { // N dim
	      __m512 c[M_NBLK];
	      for (int i = 0; i < M_NBLK; i++) c[i] = _mm512_setzero_ps(); 
	      for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col+1]; b_row_idx++){
	          int b_row = rowidxs[b_row_idx];
		  __m512 a[M_NBLK];
		  for (int i = 0; i < M_NBLK; i++) a[i] = _mm512_set1_ps(A[(mb*M_NBLK+i)*K+b_row]);
		  __m512 b = _mm512_load_ps(&B[1*16]);
		  for (int i = 0; i < M_NBLK; i++) c[i] = _mm512_fmadd_ps(b, a[i], c[i]);
	      }
	      for (int i = 0; i < M_NBLK; i++){
		  if (with_relu){
		      c[i] = _mm512_max_ps(c[i], d);
		  }
	          _mm512_store_ps(C + (mb*M_NBLK+i)*N + b_col*16, c[i]);
	      }
      }
}
}

#ifdef __AVX512BF16__
void sparse_gemm_bsc_bf16(
    int M, int N, int K,
    const short* A,
    const short* B,
    const int64_t* rowidx,
    const int64_t* colptr,
    int ncolptr,
    int blocksize[2],
    const float* C,
    bool with_relu
) {
  int M_NBLK = 4;
  assert(M % M_NBLK == 0);
  assert(K % blocksize[0] == 0);
  assert(N % blocksize[1] == 0);
  #pragma omp parallel for collapse(2)
  for (int mb = 0; mb < M / M_NBLK; mb++){
      int nnz_idx = 0;
      for (int b_col = 0; b_col < ncolptr-1; b_col++) { // N dim
              __m512 c[M_NBLK];
              if (with_relu){
                  __m512 d = _mm512_setzero_ps();
              }
              for (int i = 0; i < M_NBLK; i++) c[i] = _mm512_setzero_ps();
              for (int b_row_idx = colptr[b_col]; b_row_idx < colptr[b_col+1]; b_row_idx++, nnz_idx++){
                  int b_row = rowidxs[b_row_idx];
                  __m512bh a[M_NBLK];
                  for (int i = 0; i < M_NBLK; i++) a[i] = _mm512bh(_mm512_set1_ps(A[(mb*M_NBLK+i)*K+b_row]));
                  __m512bh b = _mm512bh(_mm512_load_ps(&B->data[nnz_idx*16]));
                  for (int i = 0; i < M_NBLK; i++) c[i] = _mm512_dpbf16_ps(b, a[i], c[i]);
              }
              for (int i = 0; i < M_NBLK; i++){
                  if (with_relu){
                      c[i] = _mm512_max_ps(c[i], d);
                  }
                  _mm512_store_ps(C + m*N + b_col*16, c);
              }
        }
}
}
#endif
