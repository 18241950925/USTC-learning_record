#include <iostream>
#include <cuda_runtime.h>
// ����˺���
__global__ void matrixMultiplication(int M, int N, int P, int K, int* D, int* S, int* result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < P) {
        int sum = 0;
        // ����˻������Ԫ��ֵ
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
    // ���������ڴ沢��ȡ�������
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
    // �����豸�ڴ沢�������ݵ��豸�ڴ�
    int* d_D;
    int* d_S;
    int* d_result;
    cudaMalloc((void**)&d_D, sizeof(int) * M * N);
    cudaMalloc((void**)&d_S, sizeof(int) * K * 3);
    cudaMalloc((void**)&d_result, sizeof(int) * M * P);
    cudaMemcpy(d_D, h_D, sizeof(int) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S, h_S, sizeof(int) * K * 3, cudaMemcpyHostToDevice);
    // ���������С���߳̿��С
    dim3 blockSize(16, 16);
    dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    // ���ú˺���
    matrixMultiplication << <gridSize, blockSize >> > (M, N, P, K, d_D, d_S, d_result);
    // ���ƽ���������ڴ�
    cudaMemcpy(h_result, d_result, sizeof(int) * M * P, cudaMemcpyDeviceToHost);
    // ��ӡ���
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            std::cout << h_result[i * P + j] << " ";
        }
        std::cout << std::endl;
    }
    // �ͷ��ڴ�
    delete[] h_D;
    delete[] h_S;
    delete[] h_result;
    cudaFree(d_D);
    cudaFree(d_S);
    cudaFree(d_result);
    return 0;
}