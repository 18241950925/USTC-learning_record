#include <iostream>
#include <cuda_runtime.h>
// 定义块尺寸
#define BLOCK_SIZE 16
// 定义核函数
__global__ void matrixMultiplication(int M, int N, int P, int K, int *D, int *S, int *result)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    // 使用共享内存
    __shared__ int shared_D[BLOCK_SIZE][BLOCK_SIZE];
    int sum = 0;
    for (int i = 0; i < N; i += BLOCK_SIZE)
    {
        // 拷贝 D 数据到共享内存
        if (row < M && tx + i < N)
        {
            shared_D[ty][tx] = D[row * N + tx + i];
        }
        else
        {
            shared_D[ty][tx] = 0;
        }
        __syncthreads();
        for (int j = 0; j < BLOCK_SIZE && col < P; j++)
        {
            int ll = 0, rr = K - 1;
            int temp = -1;
            // 二分查找
            while (ll <= rr)
            {
                int mid = (ll + rr) / 2;
                if (col < S[mid * 3 + 1])
                {
                    rr = mid - 1;
                }
                else if (col > S[mid * 3 + 1])
                {
                    ll = mid + 1;
                }
                else
                {
                    temp = mid;
                    break;
                }
            }
            if (temp >= 0)
            {
                while (temp >= 0 && col == S[temp * 3 + 1])
                {
                    int s_row = S[temp * 3];
                    int s_val = S[temp * 3 + 2];
                    sum += shared_D[ty][j] * s_val;
                    temp--;
                }
            }
            __syncthreads();
        }
    }
    if (row < M && col < P)
    {
        result[row * P + col] = sum;
    }
}
int main()
{
    int M, N, P, K;
    std::cin >> M >> N >> P >> K;
    // 分配主机内存并读取输入矩阵
    int *h_D = new int[M * N];
    int *h_S = new int[K * 3];
    int *h_result = new int[M * P];
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cin >> h_D[i * N + j];
        }
    }
    for (int k = 0; k < K; k++)
    {
        std::cin >> h_S[k * 3] >> h_S[k * 3 + 1] >> h_S[k * 3 + 2];
    }
    // 分配设备内存并复制数据到设备内存
    int *d_D;
    int *d_S;
    int *d_result;
    cudaMalloc((void **)&d_D, sizeof(int) * M * N);
    cudaMalloc((void **)&d_S, sizeof(int) * K * 3);
    cudaMalloc((void **)&d_result, sizeof(int) * M * P);
    cudaMemcpyAsync(d_D, h_D, sizeof(int) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_S, h_S, sizeof(int) * K * 3, cudaMemcpyHostToDevice);
    // 定义网格大小和线程块大小
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    // 调用核函数
    matrixMultiplication<<<gridSize, blockSize>>>(M, N, P, K, d_D, d_S, d_result);
    // 复制结果到主机内存
    cudaMemcpyAsync(h_result, d_result, sizeof(int) * M * P, cudaMemcpyDeviceToHost);
    // 同步主机和设备之间的内存传输
    cudaDeviceSynchronize();
    // 打印结果
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < P; j++)
        {
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