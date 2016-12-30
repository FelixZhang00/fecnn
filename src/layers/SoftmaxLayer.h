#ifndef SOFTMAX_LAYER_H_
#define SOFTMAX_LAYER_H_

#include "Layer.h"

using namespace fecnn;

namespace fecnn {

 
__global__ void kernel_channel_max(int num, int channels, StorageT* data, StorageT* out) {
  CUDA_KERNEL_LOOP(index, num) {
    StorageT maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[index * channels + c], maxval);
    }
    out[index] = maxval;
  }
}
__global__ void kernel_channel_subtract(int count,int num, int channels,
     StorageT* channel_max, StorageT* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels;
    data[index] -= channel_max[n];
  }
}

__global__ void kernel_exp(int count, StorageT* data, StorageT* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

__global__ void kernel_channel_sum(int num, int channels,
    StorageT* data, StorageT* channel_sum) {
  CUDA_KERNEL_LOOP(index, num) {
    int n = index;
    StorageT sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[n * channels + c];
    }
    channel_sum[index] = sum;
  }
}

__global__ void kernel_channel_div(int count,int num,int channels,
    StorageT* channel_sum, StorageT* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels;
    data[index] /= channel_sum[n];
  }
}


// BP 部分

__global__ void kernel_channel_dot(int num, int channels,
    StorageT* data_1, StorageT* data_2,
    StorageT* channel_dot) {
  CUDA_KERNEL_LOOP(index, num) {
    int n = index;
    StorageT dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[n * channels + c]
          * data_2[n * channels + c]);
    }
    channel_dot[index] = dot;
  }
}

__global__ void mul_kernel(int n,StorageT* a,
    StorageT* b, StorageT* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}


class SoftmaxLayer : public Layer {
public:
    bool stable_gradient;

    Blob<StorageT> sacled; // 一个大小为num（即batch_size）大小的向量,用于保存计算的中间结果
    // Blob<StorageT> sum_multiplier; //sum_multiplier_这里都是1，用于辅助计算，可以看作一个行向量 [K]

    SoftmaxLayer(std::string name_): Layer(name_), stable_gradient(false){};

    SoftmaxLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,           TrainingTesting)
        SetValue(json, stable_gradient, false)
    };

    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  "); // 在基类Layer中默认设置train_me为false
        std::cout<<name<<std::endl;

        if (in.size()==0) { std::cout<<std::endl<<"SoftmaxLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"SoftmaxLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        for (int i=0;i<out.size();++i){
            out[i]->need_diff = in[i]->need_diff;
            memoryBytes += out[i]->Malloc(in[i]->dim);
        }

        std::vector<int> scale_dims = {in[0]->dim[0],1};
        sacled.Reshape(scale_dims);

        return memoryBytes;
    };
    void forward(Phase phase_){
        for (int i=0;i<in.size();++i){
                // 需要处理数据的总长度，即批次*当前层神经元的个数。
                int count=in[i]->count();
                // 每个并行处理单元的数据长度，及当前层神经元的个数。
                int channels = in[i]->sizeofitem();
                // 批次。
                int batchsize= count/channels;
                // 一个大小为num（即batch_size）大小的向量,用于保存计算的中间结果。
                StorageT* scale_data = sacled.dataGPU;

                fecnn_copy(count,in[i]->dataGPU,out[i]->dataGPU);

                kernel_channel_max<<<FECNN_GET_BLOCKS(batchsize),
                    FECNN_CUDA_NUM_THREADS>>>(batchsize, channels , out[i]->dataGPU,
                    scale_data);

                kernel_channel_subtract<<<FECNN_GET_BLOCKS(count),
                    FECNN_CUDA_NUM_THREADS>>>(count, batchsize, channels,
                    scale_data, out[i]->dataGPU); 

                kernel_exp<<<FECNN_GET_BLOCKS(count), FECNN_CUDA_NUM_THREADS>>>(
                    count, out[i]->dataGPU, out[i]->dataGPU); 

                kernel_channel_sum<<<FECNN_GET_BLOCKS(batchsize),
                    FECNN_CUDA_NUM_THREADS>>>(batchsize, channels, out[i]->dataGPU,
                    scale_data);    

                kernel_channel_div<<<FECNN_GET_BLOCKS(count),
                    FECNN_CUDA_NUM_THREADS>>>(count, batchsize, channels,
                    scale_data, out[i]->dataGPU);   
        }
    };
    void backward(Phase phase_){
        for (int i=0;i<in.size();++i){
            // if bottom still needs to compute gradients
            if (in[i]->need_diff){
                    int count=in[i]->count();
                    int channels = in[i]->sizeofitem();
                    int batchsize= count/channels;
                    StorageT* scale_data = sacled.dataGPU;

                    fecnn_copy(count,out[i]->diffGPU,in[i]->diffGPU);
                    // 此处计算点积    
                    kernel_channel_dot<<<FECNN_GET_BLOCKS(batchsize),
                        FECNN_CUDA_NUM_THREADS>>>(batchsize, channels,
                        out[i]->diffGPU, out[i]->dataGPU, scale_data);
                    // 此处计算大括号内的减法    
                    kernel_channel_subtract<<<FECNN_GET_BLOCKS(count),
                         FECNN_CUDA_NUM_THREADS>>>(count, batchsize, channels,
                         scale_data, in[i]->diffGPU);   
                    // //此处计算大括号外和a的乘法        
                    mul_kernel<<<FECNN_GET_BLOCKS(count), FECNN_CUDA_NUM_THREADS>>>(
                       count, in[i]->diffGPU, out[i]->dataGPU, in[i]->diffGPU);  

            }

        }
    };
};


}// namespace fecnn

#endif  // SOFTMAX_LAYER_H_