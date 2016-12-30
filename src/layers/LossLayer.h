#ifndef LOSS_LAYER_H_
#define LOSS_LAYER_H_

#include "Layer.h"

using namespace fecnn;

namespace fecnn {

// 该层用于计算损失函数。目标函数可以调整。
class LossLayer : public Layer {
    StorageT* loss_values; // 存储batchsize个数据
    size_t loss_numel; // =batchsize
    int numExamples; // =batchsize
    ComputeT scale;  //= loss_weight / loss_numel
public:
    ComputeT result; //准确率
    ComputeT loss;  // 损失函数的值


    LossObjective mode;
    ComputeT loss_weight; // 默认为1

    LossLayer(std::string name_, LossObjective mode_, ComputeT loss_weight_)
        : Layer(name_), mode(mode_), loss_weight(loss_weight_),
          loss_values(NULL) {
        train_me = false;
    };

    LossLayer(JSON* json): loss_values(NULL){
        SetOrDie(json, name)
        SetValue(json, phase,       TrainingTesting) 
        SetOrDie(json, mode)
        SetValue(json, loss_weight, 1)

        train_me = false;
    };

    ~LossLayer() {
        if (loss_values != NULL)
            checkCUDA(__FILE__,__LINE__, cudaFree(loss_values));
    };

    size_t Malloc(Phase phase_) {
        std::cout << (train_me ? "* " : "  ");
        std::cout << name << std::endl;

        size_t memoryBytes = 0;

        numExamples = in[0]->dim[0];

        switch (mode) {
            case MultinomialLogistic_StableSoftmax:
            case MultinomialLogistic:
                if (!(in.size() == 2 || in.size() == 3)) {
                    std::cout <<
                    "LossLayer: MultinomialLogistic should have 2 or 3 ins" <<
                    std::endl;
                    FatalError(__LINE__);
                }
            if (!same_dim_EC(in[0]->dim, in[1]->dim)) {
                std::cout <<
                "LossLayer: MultinomialLogistic should have the same dimensions except channels" <<
                std::endl;
                FatalError(__LINE__);
            }
            if (in[1]->dim[1] != 1) {
                std::cout <<
                "LossLayer: MultinomialLogistic in[1] should have only 1 channel" <<
                std::endl;
                FatalError(__LINE__);
            }
            if (in.size() == 3 && !(numel(in[0]->dim) == numel(in[2]->dim) ||
                                    sizeofitem(in[0]->dim) ==
                                    numel(in[2]->dim))) {
                std::cout <<
                "LossLayer: MultinomialLogistic in[2] size should be either the same with in[0] or should be the same with sizeofitem for in[0]" <<
                std::endl;
                FatalError(__LINE__);
            }
            loss_numel = numExamples * numspel(in[0]->dim);
            break;
            case SmoothL1:
            break;
            case Contrastive:
            break;
            case EuclideanSSE:
                break;
            case HingeL1:
                break;
            case HingeL2:
                break;
            case SigmoidCrossEntropy:
                break;
            case Infogain:
                break;
        }
        scale = loss_weight / loss_numel;

        memoryBytes += loss_numel * sizeofStorageT;
        checkCUDA(__FILE__,__LINE__, cudaMalloc(&loss_values, memoryBytes));

        // std::cout<<"loss_weights.size()="<<loss_weights.size()<<std::endl;
        
        // 输出 felixlog [4]={64,10,1,1}[4]={64,1,1,1}
        // std::cout<<"felixlog "; veciPrint(in[0]->dim); veciPrint(in[1]->dim); std::cout<<std::endl; 

        return memoryBytes;
    };

    void display() {
        std::cout << " loss = " << loss;
        std::cout << " * " << loss_weight;
        // std::cout << "  loss_numel= "<<loss_numel<<std::endl; //loss_numel=64
        if (mode == MultinomialLogistic_StableSoftmax ||
            mode == MultinomialLogistic)
            std::cout << "  eval = " << result;
        std::cout << "   ";
    };

    void eval(){
        ComputeT resultSum;

        switch(mode){
            case MultinomialLogistic_StableSoftmax:
            case MultinomialLogistic:
                Accuracy_MultinomialLogistic<<<FECNN_GET_BLOCKS(loss_numel),CUDA_NUM_THREADS>>>(
                        loss_numel, 
                        in[0]->dim[1], 
                        in[0]->dataGPU, in[1]->dataGPU,
                        loss_values);
                checkCUBLAS(__FILE__,__LINE__, GPUasum(cublasHandle, loss_numel,
                                              loss_values, 1, &resultSum));
                result += resultSum / loss_numel;

                Loss_MultinomialLogistic<<<FECNN_GET_BLOCKS(loss_numel),CUDA_NUM_THREADS>>>(
                        loss_numel, 
                        in[0]->dim[1], 
                        in[0]->dataGPU, in[1]->dataGPU,
                        loss_values);
                break;
            case SmoothL1:
                break;
            case Contrastive:
                break;
            case EuclideanSSE:
                break;
            case HingeL1:
                break;
            case HingeL2:
                break;
            case SigmoidCrossEntropy:
                break;
            case Infogain:
                break;
        }

        ComputeT lossSum;
        checkCUBLAS(__FILE__,__LINE__, GPUasum(cublasHandle, loss_numel,
                                      loss_values, 1, &lossSum));
        loss += lossSum/loss_numel;
    };


    void backward(Phase phase_){
        // either write this in Cuda or get both the prediction and ground truth to CPU and do the computation and write the diff back to GPU
        if (in[0]->need_diff){
            switch(mode){
                case MultinomialLogistic_StableSoftmax:
                // std::cout<<"scale="<<scale<<std::endl; //scale=0.015625	
                // std::cout<<"in[0]->dim[1]="<<in[0]->dim[1]<<" numspel(in[0]->dim)="<<numspel(in[0]->dim)<<std::endl;
                    LossGrad_MultinomialLogistic_StableSoftmax<<<
                        FECNN_GET_BLOCKS(loss_numel), CUDA_NUM_THREADS>>>(
                            loss_numel, 		 // = 64	 
                            in[0]->dim[1],  	 // = 10	
                            scale,				 // = loss_weight/loss_numel	
                            in[0]->dataGPU, in[1]->dataGPU, 
                            in[0]->diffGPU);
                    break;
                case MultinomialLogistic:
                    break;
                case SmoothL1:
                    break;
                case Contrastive:
                    break;
                case EuclideanSSE:
                    break;
                case HingeL1:
                    break;
                case HingeL2:
                    break;
                case SigmoidCrossEntropy:
                    break;
                case Infogain:
                    break;
            }

        }
    };
};

}// namespace fecnn

#endif  // LOSS_LAYER_H_