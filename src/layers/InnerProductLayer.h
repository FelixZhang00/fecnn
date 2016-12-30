#ifndef INNERPRODUCT_LAYER_H_
#define INNERPRODUCT_LAYER_H_

#include "Layer.h"

using namespace fecnn;

namespace fecnn {

// 全连接层
class InnerProductLayer : public Layer {
    int num_input; // 每组数据，神经元的个数
    int num_items; // =batch_size
public:
    int num_output; // 从配置文件中读出
    bool bias_term;

    StorageT* bias_multGPU; // a std::vector with size # of mini-batch training example

    InnerProductLayer(std::string name_,
                    int num_output_,
                    bool bias_term_=true,
                    ComputeT weight_lr_mult_=1.0,   Filler weight_filler_=Xavier, ComputeT weight_filler_param_=0.0,
                    ComputeT bias_lr_mult_=2.0,     Filler bias_filler_=Constant,   ComputeT  bias_filler_param_=0.0): Layer(name_),num_output(num_output_), bias_multGPU(NULL), bias_term(bias_term_){
        weight_filler = weight_filler_;
        weight_filler_param = weight_filler_param_;
        bias_filler = bias_filler_;
        bias_filler_param = bias_filler_param_;
        weight_lr_mult = weight_lr_mult_;
        bias_lr_mult   = bias_lr_mult_;
        train_me = true;
    };

    InnerProductLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,               TrainingTesting)
        SetValue(json, train_me,            true)
        SetValue(json, weight_lr_mult,      1.0)
        SetValue(json, weight_filler,       Xavier)
        SetValue(json, weight_filler_param, 0.0)
        SetValue(json, bias_lr_mult,        2.0)
        SetValue(json, bias_filler,         Constant)
        SetValue(json, bias_filler_param,   0.0)
        SetValue(json, weight_decay_mult,   1.0)
        SetValue(json, bias_decay_mult,     1.0)
        SetValue(json, bias_term,           true)
        SetOrDie(json, num_output           )

    };

    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        train_me = train_me && phase_ != Testing;

        std::cout<< (train_me? "* " : "  ");
        std::cout<<name;

        if (in.size()==0) { std::cout<<std::endl<<"InnerProductLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"InnerProductLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        num_input = sizeofitem(in[0]->dim);
        num_items = in[0]->dim[0];

        weight_dim.resize(2);
        weight_dim[0] = num_output;
        weight_dim[1] = num_input;
        weight_numel = numel(weight_dim);

        if (bias_term){        
            bias_dim.resize(1);
            bias_dim[0] = num_output;
            bias_numel   = numel(bias_dim);
        }else{
            bias_numel   = 0;
        }


        if (weight_numel>0){
            std::cout<<" weight"; veciPrint(weight_dim);
            checkCUDA(__FILE__,__LINE__, cudaMalloc(&weight_dataGPU, weight_numel * sizeofStorageT) );
            memoryBytes += weight_numel * sizeofStorageT;
        }

        if (bias_numel>0){
            std::cout<<" bias"; veciPrint(bias_dim);
            checkCUDA(__FILE__,__LINE__, cudaMalloc(&bias_dataGPU, bias_numel * sizeofStorageT) );
            memoryBytes += bias_numel * sizeofStorageT;
            checkCUDA(__FILE__,__LINE__, cudaMalloc(&bias_multGPU, num_items * sizeofStorageT) );
            Kernel_set_value<<<FECNN_GET_BLOCKS(num_items),FECNN_CUDA_NUM_THREADS>>>(num_items, bias_multGPU, CPUCompute2StorageT(1));
            memoryBytes += num_items * sizeofStorageT;
        }
        std::cout<<std::endl;

        for (int i=0;i<out.size();++i){
            out[i]->need_diff = train_me || in[i]->need_diff; // if one of them need the grad
            std::vector<int> dimOut(in[i]->dim.size());
            dimOut[0] = in[i]->dim[0];
            dimOut[1] = num_output;
            for (int d=2;d<in[i]->dim.size();++d)
                dimOut[d] = 1;

            memoryBytes += out[i]->Malloc(dimOut);

        }
        return memoryBytes;
    };

    void forward(Phase phase_){
        for (int i=0;i<in.size();++i){
            // http://rpm.pbone.net/index.php3/stat/45/idpl/12463013/numer/3/nazwa/cublasSgemm
            checkCUBLAS(__FILE__,__LINE__, 
            	GPUgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
            		num_output, num_items, num_input,       // M,N,K
            		oneComputeT, weight_dataGPU, num_input, // A [M,K]
            		in[i]->dataGPU, num_input,              // B [K,N]
            		zeroComputeT, out[i]->dataGPU, num_output) ); // C [M,N]
            if (bias_numel>0){
                checkCUBLAS(__FILE__,__LINE__, 
                	GPUgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                		num_output, num_items, 1, 
                		oneComputeT, bias_dataGPU, num_output, 
                		bias_multGPU, 1, 
                		oneComputeT, out[i]->dataGPU, num_output) );
            }
        }
    };

    void backward(Phase phase_){
        for (int i=0;i<in.size();++i){
            if (in[i]->need_diff){
                // 计算残差
                checkCUBLAS(__FILE__,__LINE__, 
                	GPUgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                		num_input, num_items, num_output, 
                		oneComputeT, weight_dataGPU, num_input, 
                		out[i]->diffGPU, num_output, 
                		oneComputeT, in[i]->diffGPU, num_input) );
            }
        }

        for (int i=0;i<in.size();++i){
            if (train_me){ // 计算权重的斜率
                ComputeT beta = ComputeT(1);

                if (weight_numel>0){
                    checkCUBLAS(__FILE__,__LINE__, 
                    	GPUgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
                    		num_input, num_output, num_items, 
                    		oneComputeT, in[i]->dataGPU,  num_input, 
                    		out[i]->diffGPU, num_output, 
                    		&beta, weight_diffGPU, num_input) );
                }
                if (bias_numel>0){
                    checkCUBLAS(__FILE__,__LINE__, 
                    	GPUgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    		num_output, 1, num_items, 
                    		oneComputeT, out[i]->diffGPU, num_output, 
                    		bias_multGPU, num_items, 
                    		&beta, bias_diffGPU,num_output) );
                }
            }
        }
    };

    ~InnerProductLayer(){
        if (bias_multGPU!=NULL) checkCUDA(__FILE__,__LINE__, cudaFree(bias_multGPU));
    };
};

}// namespace fecnn

#endif  // INNERPRODUCT_LAYER_H_