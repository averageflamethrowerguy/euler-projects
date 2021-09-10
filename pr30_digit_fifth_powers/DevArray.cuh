//
// Created by smooth_operator on 9/7/21.
//

#ifndef PR26_RECIPROCAL_CYCLES_DEVARRAY_CUH
#define PR26_RECIPROCAL_CYCLES_DEVARRAY_CUH

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

template <class T>
class DevArray
{
    // public functions
public:
    explicit DevArray()
            : start_(0),
              size(0)
    {}

    // constructor
    explicit __device__ __host__ DevArray(size_t size) {
        allocate(size);
    }

    // destructor
    void __host__ __device__ clear() {
        free();
    }

    size_t size;
    size_t sizeUsed;

    // get data
    const __host__ __device__ T* getData() const {
        return start_;
    }
    __host__ __device__ T* getData() {
        return start_;
    }

    // set
    // note that this sets exclusively to start; doesn't do pointer arithmetic
    void set(const T* src, size_t set_size) {
        // we will repeatedly increase array size until we are able to store the src array
        while (size < set_size) {
            doubleArray();
        }

        size_t copySize = set_size;
        // we will set sizeUsed as the maximum of sizeUsed and the memory to copy
        sizeUsed = std::max(sizeUsed, copySize);
        cudaError_t result = cudaMemcpy(start_, src, copySize * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess) {
            throw std::runtime_error("failed to copy to device memory");
        }
    }

    // get
    void get(T* dest, size_t get_size) {
        size_t min = std::min(get_size, size);
        cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            throw std::runtime_error("failed to copy to host memory");
        }
    }

    // pushes a new element to an array
    void __host__ __device__ push_back(T element) {
        // we will double the size of the array if we don't have enough space for the next element
        if (sizeUsed == size) {
            doubleArray();
        }

        // now we will allocate the next element
        start_[sizeUsed] = element;
        sizeUsed++;
    }

    // private functions
private:
    // allocate memory on the device
    __host__ __device__ void allocate(size_t mem_size) {
        size = mem_size;
        sizeUsed = 0;
        cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));
        if (result != cudaSuccess) {
            start_ = 0;
            size = 0;
            sizeUsed = 0;
        }
    }

    // free memory on the device
    __host__ __device__ void free() {
        if (start_ != 0) {
            cudaFree(start_);
            start_ = 0;
            size = 0;
            sizeUsed = 0;
        }
    }

    // double memory for the pointer
    __host__ __device__ void doubleArray() {
        T* newStart_ = nullptr;
        size_t newSize = 2 * size;
        cudaError_t result = cudaMalloc((void**)&newStart_, newSize * sizeof(T));
        for (int i = 0; i < size; i++) {
            newStart_[i] = start_[i];
        }

        size_t tempSizedUsed = sizeUsed;

        free();

        // sets a bunch of the pointers and the sizes
        size = newSize;
        start_ = newStart_;
        sizeUsed = tempSizedUsed;
    }

    T* start_;
};

#endif //PR26_RECIPROCAL_CYCLES_DEVARRAY_CUH