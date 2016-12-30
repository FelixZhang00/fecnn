#ifndef ACTIVATION_LAYER_H_
#define ACTIVATION_LAYER_H_

#include "Layer.h"

using namespace fecnn;

namespace fecnn {

 
__global__ void TanhForward(const int n, StorageT* in, StorageT* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = tanh(in[index]);
  }
}
__global__ void TanhBackward(const int n,  StorageT* out_diff,
     StorageT* out_data, StorageT* in_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    StorageT tanh_x = out_data[index];
    in_diff[index] = out_diff[index] * (1 - tanh_x * tanh_x);
  }
}

__global__ void SigmoidForward(const int n, StorageT* in, StorageT* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 1. / (1. + exp(-in[index]));
  }
}
__global__ void SigmoidBackward(const int n,  StorageT* out_diff,
     StorageT* out_data, StorageT* in_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    StorageT sigmoid_x = out_data[index];
    in_diff[index] = out_diff[index] * sigmoid_x * (1 - sigmoid_x);
  }
}


__global__ void ReLUForward(const int n, StorageT* in, StorageT* out,
    StorageT negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

__global__ void ReLUBackward(const int n, StorageT* in_diff,
    StorageT* out_data, StorageT* out_diff, StorageT negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    in_diff[index] = out_diff[index] * ((out_data[index] > 0)
        + (out_data[index] <= 0) * negative_slope);
  }
}

class ActivationLayer : public Layer {
public:
    ActivateMode mode;
    ActivationLayer(std::string name_): Layer(name_){};

    ActivationLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, mode,                ReLU)
        SetValue(json, phase,               TrainingTesting)
    };

    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        if (in.size()==0) { std::cout<<std::endl<<"ActivationLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"ActivationLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        for (int i=0;i<out.size();++i){
            out[i]->need_diff = in[i]->need_diff;
            memoryBytes += out[i]->Malloc(in[i]->dim);
        }
        return memoryBytes;
    };
    void forward(Phase phase_){
        switch(mode){
            case ReLU:
            for (int i=0;i<in.size();++i){
                    // 默认使用relu
                    const int count = in[i]->count();
                    ReLUForward<<<FECNN_GET_BLOCKS(count), FECNN_CUDA_NUM_THREADS>>>(
                        count, in[i]->dataGPU, out[i]->dataGPU, 0);
            }
            break;
            case Sigmoid:
            for (int i=0;i<in.size();++i){
                    const int count = in[i]->count();
                    SigmoidForward<<<FECNN_GET_BLOCKS(count), FECNN_CUDA_NUM_THREADS>>>(
                        count,in[i]->dataGPU,out[i]->dataGPU);
            }
            break;
            case Tanh:
            for (int i=0;i<in.size();++i){
                    const int count = in[i]->count();
                    TanhForward<<<FECNN_GET_BLOCKS(count), FECNN_CUDA_NUM_THREADS>>>(
                        count,in[i]->dataGPU,out[i]->dataGPU);
            }
            break;
        }
    };
    void backward(Phase phase_){
        switch(mode){
            case ReLU:
            for (int i=0;i<in.size();++i){
                // if bottom still needs to compute gradients
                if (in[i]->need_diff){
                    // 默认使用relu
                    const int count = in[i]->count();
                    ReLUBackward<<<FECNN_GET_BLOCKS(count), FECNN_CUDA_NUM_THREADS>>>(
                        count, in[i]->diffGPU, out[i]->dataGPU, out[i]->diffGPU, 0);
                }
            }
            break;
            case Sigmoid:
            for (int i=0;i<in.size();++i){
                // if bottom still needs to compute gradients
                if (in[i]->need_diff){
                    const int count = in[i]->count();
                    SigmoidBackward<<<FECNN_GET_BLOCKS(count), FECNN_CUDA_NUM_THREADS>>>(
                        count, out[i]->diffGPU, out[i]->dataGPU, in[i]->diffGPU);
                }
            }
            break;
            case Tanh:
            for (int i=0;i<in.size();++i){
                // if bottom still needs to compute gradients
                if (in[i]->need_diff){
                    const int count = in[i]->count();
                    TanhBackward<<<FECNN_GET_BLOCKS(count), FECNN_CUDA_NUM_THREADS>>>(
                        count, out[i]->diffGPU, out[i]->dataGPU, in[i]->diffGPU);
                }
            }
            break;
        }
    };
};


}// namespace fecnn

#endif  // ACTIVATION_LAYER_H_