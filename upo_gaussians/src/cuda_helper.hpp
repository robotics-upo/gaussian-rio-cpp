#pragma once
#include <upo_gaussians/types.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_WARP_SIZE 32

#define UPO_CUDA_NUM_THREADS     256
#define UPO_CUDA_WARPS_PER_BLOCK (UPO_CUDA_NUM_THREADS/CUDA_WARP_SIZE)

namespace upo_gaussians::detail {

	template <typename T>
	__device__ inline constexpr unsigned capToWarpSize(T val)
	{
		return val <= CUDA_WARP_SIZE ? val : CUDA_WARP_SIZE;
	}

	template <typename T>
	struct Wrapper {
		alignas(alignof(T)) char raw[sizeof(T)];

		UPO_HOST_DEVICE auto& operator *()       { return *(      T*)raw; }
		UPO_HOST_DEVICE auto& operator *() const { return *(const T*)raw; }
	};

	static inline void cudaCheck(cudaError_t err)
	{
		if (err != cudaSuccess) {
			std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorName(err));
			std::abort();
		}
	}

	template <typename T>
	class GpuArray {
		static_assert(alignof(T) <= 256, "Unsupported alignment");

		T* m_ptr{};

	public:
		constexpr GpuArray() = default;
		GpuArray(GpuArray const&) = delete;
		constexpr GpuArray(GpuArray&& rhs) : m_ptr{rhs.m_ptr} { rhs.m_ptr = nullptr; }
		~GpuArray() { clear(); }

		constexpr operator T*()                          { return m_ptr; }
		constexpr operator T const*()              const { return m_ptr; }
		constexpr operator bool()                  const { return m_ptr != nullptr; }
		constexpr T&       operator [](size_t idx)       { return m_ptr[idx]; }
		constexpr T const& operator [](size_t idx) const { return m_ptr[idx]; }

		void reserve(size_t sz) {
			clear();

			void* ptr;
			cudaCheck(cudaMallocManaged(&ptr, sizeof(T)*sz));
			m_ptr = (T*)ptr;
		}

		void clear() {
			if (m_ptr) {
				cudaFree(m_ptr);
				m_ptr = nullptr;
			}
		}
	};

}
