#include "matmul.cuh"

int main()
{
    MatMulWrapper<float> A(100, 100, kRandomInit);
    MatMulWrapper<float> B(100, 100, kRandomInit);
    MatMulWrapper<float> C(100, 100, kEmptyInit);


    
}