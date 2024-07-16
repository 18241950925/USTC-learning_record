#include <iostream>
#include <cuda_runtime.h>

// 定义核函数
__global__ void matrixMultiplication(int M, int N, int P, int K, int *D, int *values, int *rows, int *cols, int *nums, int *result)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int first_index = cols[col];
    int num = nums[col];
    if (row < M && col < P)
    {
        int sum = 0;
        if (num > 0)
        {
            for (int k = first_index; k < first_index + num; k++)
            {
                int s_row = rows[k];
                int s_val = values[k];
                sum += D[row * N + s_row] * s_val;
            }
        }
        result[row * P + col] = sum;
    }
}
int main()
{
    int M, N, P, K;
    std::cin >> M >> N >> P >> K;
    // 分配主机内存并读取输入矩阵
    int *h_D = new int[M * N];
    // 使用CSR格式存储稀疏矩阵
    int *h_value = new int[K];
    int *h_row = new int[K]; // 对应的行号
    int *h_col = new int[K]; // 该列的第一个元素的value下标
    int *h_num = new int[K]; // 每一列的个数
    std::fill(h_col, h_col + K, -1);
    std::fill(h_num, h_num + K, 0);
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
        int temp;
        std::cin >> h_row[k];
        std::cin >> temp;
        std::cin >> h_value[k];
        if (h_col[temp] < 0)
            h_col[temp] = k;
        h_num[temp]++;
    }
    // 分配设备内存并复制数据到设备内存
    int *d_D;
    int *d_value;
    int *d_row;
    int *d_col;
    int *d_num;
    int *d_result;
    cudaMalloc((void **)&d_D, sizeof(int) * M * N);
    cudaMalloc((void **)&d_value, sizeof(int) * K);
    cudaMalloc((void **)&d_row, sizeof(int) * K);
    cudaMalloc((void **)&d_col, sizeof(int) * K);
    cudaMalloc((void **)&d_num, sizeof(int) * K);
    cudaMalloc((void **)&d_result, sizeof(int) * M * P);
    cudaMemcpy(d_D, h_D, sizeof(int) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, h_value, sizeof(int) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_col, sizeof(int) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, h_row, sizeof(int) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_num, h_num, sizeof(int) * K, cudaMemcpyHostToDevice);

    // 定义网格大小和线程块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    // 调用核函数
    matrixMultiplication<<<gridSize, blockSize>>>(M, N, P, K, d_D, d_value, d_row, d_col, d_num, d_result);
    // cudaDeviceSynchronize();
    //  复制结果到主机内存
    cudaMemcpy(h_result, d_result, sizeof(int) * M * P, cudaMemcpyDeviceToHost);
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
    delete[] h_value;
    delete[] h_col;
    delete[] h_row;
    delete[] h_num;
    delete[] h_result;

    cudaFree(d_D);
    cudaFree(d_row);
    cudaFree(d_num);
    cudaFree(d_col);
    cudaFree(d_value);
    cudaFree(d_result);
    return 0;
}