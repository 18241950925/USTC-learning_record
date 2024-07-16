#include <iostream>
#include <cuda_runtime.h>
// 定义核函数
__global__ void matrixMultiplication(int M, int N, int P, int K, int* D, int* S, int* result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < P) {
        int sum = 0;
        // 计算乘积矩阵的元素值
        for (int k = 0; k < K; k++) {
            int s_row = S[k * 3];
            int s_col = S[k * 3 + 1];
            int s_val = S[k * 3 + 2];
            if (s_col == col) {
                sum += D[row * N + s_row] * s_val;
            }
        }
        result[row * P + col] = sum;
    }
}
int main() {
    int M, N, P, K;
    std::cin >> M >> N >> P >> K;
    // 分配主机内存并读取输入矩阵
    int* h_D = new int[M * N];
    int* h_S = new int[K * 3];
    int* h_result = new int[M * P];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cin >> h_D[i * N + j];
        }
    }
    for (int k = 0; k < K; k++) {
        std::cin >> h_S[k * 3] >> h_S[k * 3 + 1] >> h_S[k * 3 + 2];
    }
    // 分配设备内存并复制数据到设备内存
    int* d_D;
    int* d_S;
    int* d_result;
    cudaMalloc((void**)&d_D, sizeof(int) * M * N);
    cudaMalloc((void**)&d_S, sizeof(int) * K * 3);
    cudaMalloc((void**)&d_result, sizeof(int) * M * P);
    cudaMemcpy(d_D, h_D, sizeof(int) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S, h_S, sizeof(int) * K * 3, cudaMemcpyHostToDevice);
    // 定义网格大小和线程块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    // 调用核函数
    matrixMultiplication << <gridSize, blockSize >> > (M, N, P, K, d_D, d_S, d_result);
    // 复制结果到主机内存
    cudaMemcpy(h_result, d_result, sizeof(int) * M * P, cudaMemcpyDeviceToHost);
    // 打印结果
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            std::cout << h_result[i * P + j] << " ";
        }
        std::cout << std::endl;
    }
    // 释放内存
    delete[] h_D;
    delete[] h_S;
    delete[] h_result;
    cudaFree(d_D);
    cudaFree(d_S);
    cudaFree(d_result);
    return 0;
}