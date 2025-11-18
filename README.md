# Exp - 5 - Bubble Sort and Merge sort in CUDA 
<h3>NAME: KARTHIKEYAN M</h3>
<h3>REG NO: 212223110020</h3>
<h3>DATE: 17.11.25 </h3>

## AIM:
To Implement Bubble Sort and Merge Sort on the GPU using CUDA to enhance the performance of sorting tasks by parallelizing comparisons and swaps within the sorting algorithm.

## EQUIPMENTS REQUIRED:
- Hardware
   - PCs with NVIDIA GPU & CUDA NVCC
   - Google Colab with NVCC Compiler, CUDA Toolkit installed


## PROCEDURE:

1. **Initialize the CUDA Environment**:
   - Set up the necessary hardware and software for CUDA programming, including an NVIDIA GPU and NVCC compiler (Google Colab with CUDA Toolkit if using a cloud environment).

2. **Define Bubble Sort and Merge Sort Kernels**:
   - **Bubble Sort Kernel**: Define a CUDA kernel to perform Bubble Sort using a single block and parallelizing comparisons and swaps across threads. Each thread performs a comparison and swap if necessary, iterating over multiple passes.
   - **Merge Sort Kernel**: Define a CUDA kernel to perform Merge Sort in stages, where each thread merges sub-arrays, doubling the size of merged sub-arrays in each iteration.

3. **Memory Allocation on Device**:
   - Allocate device memory for the arrays to be sorted (using `cudaMalloc`).
   - Copy data from the host (CPU) array to the device (GPU) using `cudaMemcpy`.

4. **Kernel Execution and Synchronization**:
   - Launch the Bubble Sort kernel or Merge Sort kernel on the GPU. Use appropriate block and grid dimensions, such as specifying block size and calculating the required number of blocks.
   - Use `__syncthreads()` to synchronize threads after each pass in the sorting kernels, ensuring correct data access.

5. **CPU Sorting Functions**:
   - Implement Bubble Sort and Merge Sort for the CPU to compare performance. Measure execution time using high-resolution clock timers from the `chrono` library.

6. **Run Sorting Algorithms**:
   - Test both algorithms with arrays of different sizes (500 and 1000 elements) and configurations (different block sizes).
   - For each sorting algorithm, execute both CPU and GPU implementations and measure execution times.

7. **Measure and Compare Performance**:
   - Record the GPU execution time using `cudaEventRecord` for precise time tracking.
   - Compare the CPU and GPU execution times for both sorting algorithms, noting the effect of different block sizes and array sizes.

8. **Copy Data Back to Host and Clean Up**:
   - After the GPU kernel completes, copy the sorted array back to the host.
   - Free the allocated device memory to prevent memory leaks.

9. **Output Results**:
   - Print the execution times of Bubble Sort and Merge Sort on both CPU and GPU, formatted in a table for comparison.
   - Analyze the performance improvement observed due to parallelization on the GPU.
## PROCEDURE:

Tasks:

a. Modify the Kernel:

Implement Bubble Sort and Merge Sort using CUDA by assigning each comparison and swap task to individual threads.
Ensure the kernel checks boundaries to avoid out-of-bounds access, particularly for edge cases.
b. Performance Analysis:

Measure the execution time of the CUDA Bubble Sort with different array sizes (e.g., 512, 1024, 2048 elements).
Experiment with various block sizes (e.g., 16, 32, 64 threads per block) to analyze their effect on execution time and efficiency.
c. Comparison:

Compare the performance of the CUDA-based Bubble Sort and Merge Sort with a CPU-based Bubble Sort and Merge Sort implementation.
Discuss the differences in execution time and explain the limitations of Bubble Sort and Merge Sort when parallelized on the GPU.
## PROGRAM:
```
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

__global__ void bubbleSortKernel(int *d_arr, int n) {
    int temp;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (blockIdx.x > 0) return;

    for (int i = 0; i < n - 1; i++) {
        if (idx < n - 1 - i) {
            if (d_arr[idx] > d_arr[idx + 1]) {
                temp = d_arr[idx];
                d_arr[idx] = d_arr[idx + 1];
                d_arr[idx + 1] = temp;
            }
        }
        __syncthreads();
    }
}

__device__ void merge(int *arr, int left, int mid, int right, int *temp) {
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}

__global__ void mergeSortKernel(int *d_arr, int *d_temp, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int size = 1; size < n; size *= 2) {
        int left = 2 * size * tid;
        if (left < n) {
            int mid = min(left + size - 1, n - 1);
            int right = min(left + 2 * size - 1, n - 1);

            merge(d_arr, left, mid, right, d_temp);
        }

        __syncthreads();

        if (tid == 0) {
            int *temp = d_arr;
            d_arr = d_temp;
            d_temp = temp;
        }

        __syncthreads();
    }
}

void mergeHost(int *arr, int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    i = 0;
    j = 0;
    k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

void bubbleSort(int *arr, int n, int blockSize, int numBlocks) {
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    bubbleSortKernel<<<numBlocks, blockSize>>>(d_arr, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    printf("Bubble Sort (GPU) took %f milliseconds\n", milliseconds);
}

void mergeSort(int *arr, int n, int blockSize, int numBlocks) {
    int *d_arr, *d_temp;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mergeSortKernel<<<numBlocks, blockSize>>>(d_arr, d_temp, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);

    printf("Merge Sort (GPU) took %f milliseconds\n", milliseconds);
}

void bubbleSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("Bubble Sort (CPU) took %f milliseconds\n", duration.count());
}

void mergeSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int size = 1; size < n; size *= 2) {
        int left = 0;
        while (left + size < n) {
            int mid = left + size - 1;
            int right = min(left + 2 * size - 1, n - 1);

            mergeHost(arr, left, mid, right);
            left += 2 * size;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("Merge Sort (CPU) took %f milliseconds\n", duration.count());
}

int main() {
    int n_array[] = {500, 1000};
    for (int i = 0; i < 2; i++) {
        int n = n_array[i];
        int *arr = (int*)malloc(n * sizeof(int));

        int blockSize_array[] = {16, 32};

        for (int i = 0; i < 2; i++) {
            int blockSize = blockSize_array[i];
            int numBlocks = (n + blockSize - 1) / blockSize;

            printf("\nArray Size:%d\nBlock Size:%d\nNum Blocks:%d\n", n, blockSize, numBlocks);

            for (int i = 0; i < n; i++) {
                arr[i] = rand() % 1000;
            }

            bubbleSortCPU(arr, n);

            for (int i = 0; i < n; i++) {
                arr[i] = rand() % 1000;
            }

            bubbleSort(arr, n, blockSize, numBlocks);

            for (int i = 0; i < n; i++) {
                arr[i] = rand() % 1000;
            }

            mergeSortCPU(arr, n);

            for (int i = 0; i < n; i++) {
                arr[i] = rand() % 1000;
            }

            mergeSort(arr, n, blockSize, numBlocks);
            printf("\n");
        }
        free(arr);
    }
    return 0;
}


```
## OUTPUT:
<img width="495" height="758" alt="image" src="https://github.com/user-attachments/assets/1beb1f96-905e-4c09-ab40-abd2949701a8" />

## RESULT:
Thus, the program has been executed using CUDA to implement Bubble Sort and Merge Sort on the GPU using CUDA and analyze the efficiency of this sorting algorithm when parallelized.
