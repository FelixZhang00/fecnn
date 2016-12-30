#ifndef MY_CUDA_HELPER_H
#define MY_CUDA_HELPER_H

#include "common.h"

using namespace fecnn;

namespace fecnn {

//////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernels
//////////////////////////////////////////////////////////////////////////////////////////////////

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int FECNN_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int FECNN_GET_BLOCKS(const int N) {
  return (N + FECNN_CUDA_NUM_THREADS - 1) / FECNN_CUDA_NUM_THREADS;
}


#define CUDA_NUM_THREADS 512

#define MAX_NUM_BLOCKS 2880

// N 表示任务总数
inline int CUDA_GET_BLOCKS(const size_t N) {
    return min(MAX_NUM_BLOCKS, int((N + size_t(CUDA_NUM_THREADS) - 1) / CUDA_NUM_THREADS));
}

// 所有线程并行执行，这样需要循环几次
inline size_t CUDA_GET_LOOPS(const size_t N) {
    size_t total_threads = CUDA_GET_BLOCKS(N)*CUDA_NUM_THREADS;
    return (N + total_threads -1)/ total_threads;
}

__global__ void Accuracy_MultinomialLogistic(
    size_t N, int C,
    const StorageT *pred, const StorageT *label,
    StorageT *loss) {

	CUDA_KERNEL_LOOP(idx,N){
        int l = int(GPUStorage2ComputeT(label[idx]));
        int baseID = idx * C;
        int elementID = baseID + l;

        ComputeT prob = GPUStorage2ComputeT(pred[elementID]); // 预测中猜中elementID时的概率为prob
        loss[idx] = GPUCompute2StorageT(1); // 预测正确则为1
        for (int d = 0; d < C; ++d) {
            if (GPUStorage2ComputeT(pred[baseID + d]) > prob) { // 如果prob不是C个多项中最大的那个，说明猜错了！
                loss[idx] = GPUCompute2StorageT(0); // 预测失败则为0
            }
        }
    }
}

// 详细公式见：winter1516_lecture3.pdf
__global__ void Loss_MultinomialLogistic(
    size_t N, int C,
    const StorageT* pred, const StorageT* label,
    StorageT *loss) {
	CUDA_KERNEL_LOOP(idx,N){
		int l = int(GPUStorage2ComputeT(label[idx]));
		int baseID = idx * C;
		int elementID = baseID + l;

		ComputeT prob = max(GPUStorage2ComputeT(pred[elementID]), ComputeT_MIN);  // 取出正确预测的概率值prob
		ComputeT res = -log(prob);
		loss[idx] = GPUCompute2StorageT(res);
	}
}



// 参考：http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
// 下面是残差回传。
__global__ void LossGrad_MultinomialLogistic_StableSoftmax(
    size_t N, int C,
    ComputeT scale,
    const StorageT *pred, const StorageT *label, 
    StorageT *diff) {

	CUDA_KERNEL_LOOP(idx,N){
        int l = int(GPUStorage2ComputeT(label[idx]));
        int baseID = idx * C;
        int elementID = baseID + l; // 正确类别的位置

        for (int d = 0; d < C; ++d) {
            int k = baseID + d;
            diff[k] = GPUCompute2StorageT(GPUStorage2ComputeT(diff[k]) +
                                          scale *
                                          GPUStorage2ComputeT(pred[k]));
        }
        diff[elementID] = GPUCompute2StorageT(
            GPUStorage2ComputeT(diff[elementID]) - scale);
    }
}


__global__ void Kernel_set_value(int N, StorageT* GPUdst, StorageT value){
    CUDA_KERNEL_LOOP(index,N){
        GPUdst[index] = value;
    }
}


 
__global__ void Kernel_bsa2b(int num, StorageT* a, StorageT* b){
	CUDA_KERNEL_LOOP(index,num){
		b[index] -= a[index]; 
	}
}

void bsa2b(size_t N, StorageT* a, StorageT* b){
	 Kernel_bsa2b<<<FECNN_GET_BLOCKS(N), FECNN_CUDA_NUM_THREADS>>>(N,a,b);
}


__global__ void Kernel_update_SGDL1(size_t CUDA_NUM_LOOPS, size_t N, int nNets, ComputeT decay, ComputeT momentum, ComputeT lr, const StorageT* weights, StorageT* gradients){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        ComputeT w  = GPUStorage2ComputeT(weights[idx]);
        ComputeT h  = GPUStorage2ComputeT(gradients[idx]);
        ComputeT g;
        if (w>0)        g = decay;
        else if (w<0)   g = -decay;
        else            g = 0;
        for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]);
        gradients[idx] = GPUCompute2StorageT(momentum * h + lr * g);
    }
}

// N表示weight_numel或bais_numel。更新“Wd_velocity”
__global__ void Kernel_update_SGDL2(size_t N, 
				int nNets, ComputeT decay, ComputeT momentum, ComputeT lr, 
				const StorageT* weights, StorageT* gradients){
	CUDA_KERNEL_LOOP(idx,N){
		ComputeT w  = GPUStorage2ComputeT(weights[idx]);
		ComputeT h  = GPUStorage2ComputeT(gradients[idx]);
		ComputeT g  = decay * w;     // L2 regularization
		for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]); //加上对应的diff部分
		gradients[idx] = GPUCompute2StorageT(momentum * h + lr * g);
	}
}


void update_solver(SolverAlgorithm solver, Regularizer regularizer, size_t N, int nNets, ComputeT decay, ComputeT momentum,ComputeT lr, const StorageT* weights, StorageT* gradients){
    switch (solver){
        case SGD:
            if (regularizer==L1)
                Kernel_update_SGDL1<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,nNets,decay,momentum,lr,weights,gradients);
            else
                Kernel_update_SGDL2<<<FECNN_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
                	N,nNets,decay,momentum,lr,weights,gradients);
            break;
        default:
            break;     
    }
    checkCUDA(__FILE__,__LINE__,cudaGetLastError());
}


// fixme 只适用于GPU模式
void fecnn_copy(int N, StorageT* X, StorageT* Y) {
	checkCUDA(__FILE__,__LINE__, cudaMemcpy(Y, X, sizeof(StorageT) * N, cudaMemcpyDefault));
}

}// namespace fecnn

#endif  // MY_CUDA_HELPER_H