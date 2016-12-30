#ifndef POOLING_LAYER_H_
#define POOLING_LAYER_H_

#include "Layer.h"

using namespace fecnn;

namespace fecnn {
 
__global__ void MaxPoolForward(int nthreads,
    StorageT* in_data, int num, int channels,
    int width, int height, int pooled_width, int pooled_height,int kernel_w,int kernel_h,
    int stride_w, int stride_h, int pad_w,int pad_h,
    StorageT* out_data,int* mask){

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int pw = index % pooled_width;
        const int ph = (index / pooled_width) % pooled_height;
        const int c = (index / pooled_width / pooled_height) % channels;
        const int n = index / pooled_width / pooled_height / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        const int hend = min(hstart + kernel_h, height);
        const int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        StorageT maxval = -FLT_MAX;
        int maxidx = -1;
        const StorageT* const bottom_slice =
            in_data + (n * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            if (bottom_slice[h * width + w] > maxval) {
              maxidx = h * width + w;
              maxval = bottom_slice[maxidx];
            }
          }
        }
        out_data[index] = maxval;
        mask[index] = maxidx;   
    }
}

__global__ void MaxPoolBackward(int nthreads,  StorageT* out_diff,
     int*  mask, int num,
     int channels, int width,int height,
     int pooled_width,int pooled_height,int kernel_w,
     int kernel_h,int stride_w,int stride_h,
     int pad_w,int pad_h, StorageT* in_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    StorageT gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const StorageT* const out_diff_slice = out_diff + offset;

    const int* const mask_slice = mask + offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (mask_slice[ph * pooled_width + pw] == h * width + w) {
          gradient += out_diff_slice[ph * pooled_width + pw];
        }
      }
    }
    in_diff[index] = gradient;  // 将上层的误差翻转到对应的位置（forward时最大的位置处），其他位置误差为0
  } 
}

class PoolingLayer : public Layer {
public:
    // Pool mode;
    std::vector<int> window;
    std::vector<int> padding;
    std::vector<int> stride;

    // 使用max-pool方法会用到，保存前馈过程中pool时的最大元素的位置。
    Blob<int> maxIdxs;

    void init(){

    };

    PoolingLayer(std::string name_,std::vector<int> window_, std::vector<int> padding_, std::vector<int> stride_): Layer(name_), window(window_), padding(padding_), stride(stride_){
        init();
    };

    PoolingLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,               TrainingTesting)
        SetOrDie(json, window               )
        std::vector<int> zeros = std::vector<int>(window.size(),0);
        SetValue(json, padding,             zeros)
        SetValue(json, stride,              window)

        init();
    };

    size_t Malloc(Phase phase_){
        size_t memoryBytes=0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        if (in.size()==0) { std::cout<<std::endl<<"PoolingLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"PoolingLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        for (int i=0;i<out.size();++i){
            out[i]->need_diff = in[i]->need_diff;

            // compute the size to allocate memory
            std::vector<int> dimOut(in[i]->dim.size());
            dimOut[0] = in[i]->dim[0]; // size of mini-bath
            dimOut[1] = in[i]->dim[1]; // channels
            for (int d=2;d<in[i]->dim.size();++d){
              dimOut[d] = 1 + static_cast<int>(ceil(static_cast<float>(in[i]->dim[d] + 2*padding[d-2] - window[d-2])/stride[d-2]));
            }

            
            // @TODO 加上判断是否是MAX-mode
            // @TODO 需要加上内存统计
            maxIdxs.Reshape(dimOut);

            memoryBytes += out[i]->Malloc(dimOut);
        }
        return memoryBytes;
    };
    void forward(Phase phase_){
        for (int i=0;i<in.size();++i){
                int count = out[i]->count();
                int *mask = maxIdxs.dataGPU;

                MaxPoolForward<<<FECNN_GET_BLOCKS(count), FECNN_CUDA_NUM_THREADS>>>(
                    count, in[i]->dataGPU, in[i]->dim[0], in[i]->dim[1],
                    in[i]->dim[2], in[i]->dim[3], out[i]->dim[2], out[i]->dim[3], window[0],
                    window[1], stride[0], stride[1], padding[0], padding[1], out[i]->dataGPU,
                    mask);
        }
    };
    void backward(Phase phase_){
        for (int i=0;i<in.size();++i){
            // if bottom still needs to compute gradients
            if (in[i]->need_diff){
                    int count = in[i]->count();
                    int* mask = maxIdxs.dataGPU;

                    MaxPoolBackward<<<FECNN_GET_BLOCKS(count), FECNN_CUDA_NUM_THREADS>>>(
                        count, out[i]->diffGPU, mask, out[i]->dim[0], in[i]->dim[1],
                        in[i]->dim[2], in[i]->dim[3], out[i]->dim[2], out[i]->dim[3],
                        window[0], window[1], stride[0], stride[1], padding[0], padding[1], 
                        in[i]->diffGPU);
            }
        }
    };
    ~PoolingLayer(){
    };
};

}// namespace fecnn

#endif  // POOLING_LAYER_H_