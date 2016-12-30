#ifndef LAYER_H_
#define LAYER_H_

#include "../common/common.h"
#include "../common/JSON.h"
#include "../common/my_cuda_helper.h"
#include "../io/tensor.h"
#include "../io/response.h"

using namespace fecnn;

namespace fecnn {

class Layer {
public:
    StorageT *weight_dataGPU;
    StorageT *weight_diffGPU; // 中间形式的权重的斜率.指向weight_histGPU的中间
    StorageT *weight_histGPU; // 由Solver的solve算出最终适合权重更新的斜率（相当于Wd_velocity），再由Layer的update方法用来更新weight_dataGPU

    StorageT *bias_dataGPU;
    StorageT *bias_diffGPU;
    StorageT *bias_histGPU;

    std::vector<Response *> in;
    std::vector<Response *> out;

    std::mt19937 rng;
    cublasHandle_t cublasHandle;

    // parameters:
    int GPU;

    std::string name;
    Phase phase;
    bool train_me; // user specify whether they want to tune this layer

    ComputeT weight_lr_mult;
    Filler weight_filler;
    ComputeT weight_filler_param;
    std::vector<int> weight_dim;
    size_t weight_numel;
    ComputeT weight_decay_mult;

    ComputeT bias_lr_mult;
    Filler bias_filler;
    ComputeT bias_filler_param;
    std::vector<int> bias_dim;
    size_t bias_numel;
    ComputeT bias_decay_mult;

    Layer() : phase(TrainingTesting), train_me(false), weight_dataGPU(NULL),
              weight_diffGPU(NULL), weight_histGPU(NULL), bias_dataGPU(NULL),
              bias_diffGPU(NULL), bias_histGPU(NULL), weight_numel(0),
              bias_numel(0), weight_decay_mult(ComputeT(1)),
              bias_decay_mult(ComputeT(1)) {
        checkCUBLAS(__FILE__,__LINE__, cublasCreate(&cublasHandle));
        std::random_device rd;
        rng.seed(rd());
    };

    Layer(std::string name_) : name(name_), phase(TrainingTesting),
                               train_me(false), weight_dataGPU(NULL),
                               weight_diffGPU(NULL), weight_histGPU(NULL),
                               bias_dataGPU(NULL), bias_diffGPU(NULL),
                               bias_histGPU(NULL), weight_numel(0),
                               bias_numel(0), weight_decay_mult(ComputeT(1)),
                               bias_decay_mult(ComputeT(1)) {
        checkCUBLAS(__FILE__,__LINE__, cublasCreate(&cublasHandle));
        std::random_device rd;
        rng.seed(rd());
    };

    virtual ~Layer() {
        if (weight_dataGPU != NULL)
            checkCUDA(__FILE__,__LINE__, cudaFree(weight_dataGPU));

        if (bias_dataGPU != NULL) checkCUDA(__FILE__,__LINE__, cudaFree(bias_dataGPU));
    };

    ComputeT ameanWeightData() {
        if (weight_dataGPU == NULL) return -1;
        ComputeT result;
        size_t n = numel(weight_dim);
        checkCUBLAS(__FILE__,__LINE__,
                    GPUasum(cublasHandle, n, weight_dataGPU, 1, &result));
        result /= ComputeT(n);
        return result;
    };

    ComputeT ameanWeightDiff() {
        if (weight_diffGPU == NULL) return -1;
        ComputeT result;
        size_t n = numel(weight_dim);
        checkCUBLAS(__FILE__,__LINE__,
                    GPUasum(cublasHandle, n, weight_diffGPU, 1, &result));
        result /= ComputeT(n);
        return result;
    };

    int checkNaNWeight(){
        return fecnn::checkNaN(weight_dataGPU, numel(weight_dim));
    };

    int checkNaNWeightDiff(){
        return fecnn::checkNaN(weight_diffGPU, numel(weight_dim));
    };    

    ComputeT ameanBiasData() {
        if (bias_dataGPU == NULL) return -1;
        ComputeT result;
        size_t n = numel(bias_dim);
        checkCUBLAS(__FILE__,__LINE__,
                    GPUasum(cublasHandle, n, bias_dataGPU, 1, &result));
        result /= ComputeT(n);
        return result;
    };

    ComputeT ameanBiasDiff() {
        if (bias_diffGPU == NULL) return -1;
        ComputeT result;
        size_t n = numel(bias_dim);
        checkCUBLAS(__FILE__,__LINE__,
                    GPUasum(cublasHandle, n, bias_diffGPU, 1, &result));
        result /= ComputeT(n);
        return result;
    };

    int checkNaNBias(){
        return fecnn::checkNaN(bias_dataGPU, numel(bias_dim));
    };

    int checkNaNBiasDiff(){
        return fecnn::checkNaN(bias_diffGPU, numel(bias_dim));
    };       

    void addIn(Response *r) { in.push_back(r); };

    void addOut(Response *r) { out.push_back(r); };

    virtual size_t Malloc(Phase phase_) {    // by default, do nothing
        std::cout << (train_me ? "* " : "  ");
        std::cout << name << std::endl;
        return 0;
    };

    virtual void forward(Phase phase_) { };  // by default, do nothing
    virtual void backward(Phase phase_) { }; // by default, do nothing
    virtual void display() { };

    virtual bool isDataLayer() { return false; };

    void fillGPU(StorageT *GPUmem, std::vector<int> dim, Filler filler,
                 ComputeT param = 0) {
        int n = numel(dim);
        StorageT *CPUbuf = new StorageT[n];
        switch (filler) {
            case Xavier: {
                int fan_in = ComputeT(n / dim[0]);
                ComputeT scale = sqrt(ComputeT(3) / fan_in);

                //均匀分布uniform
                std::uniform_real_distribution<ComputeT> distribution(-scale,
                                                                      scale);
                for (StorageT *p = CPUbuf; p != CPUbuf + n; ++p) {
                    *p = CPUCompute2StorageT(distribution(rng));
                }
            }
            break;
            case Gaussian: {
                //正态分布normal
                std::normal_distribution<ComputeT> distribution(0, param);
                for (StorageT *p = CPUbuf; p != CPUbuf + n; ++p) {
                    *p = CPUCompute2StorageT(distribution(rng));
                }
            }
            break;
            case Constant: {
                StorageT paramStorageT = CPUCompute2StorageT(param);
                for (StorageT *p = CPUbuf; p != CPUbuf + n; ++p) {
                    *p = paramStorageT;
                }
            }
            break;
        }
        checkCUDA(__FILE__,__LINE__, cudaMemcpy(GPUmem, CPUbuf, n * sizeofStorageT,
                                       cudaMemcpyHostToDevice));

        delete[] CPUbuf;
    }

    void randInit() {
        if (weight_dataGPU != NULL) fillGPU(weight_dataGPU, weight_dim, weight_filler, weight_filler_param);
        if (bias_dataGPU != NULL) fillGPU(bias_dataGPU, bias_dim, bias_filler, bias_filler_param);
    };

    void clearDiff() {
        if (weight_diffGPU != NULL)
            checkCUDA(__FILE__,__LINE__, cudaMemset(weight_diffGPU, 0,
                                           sizeofStorageT * weight_numel));
        if (bias_diffGPU != NULL)
            checkCUDA(__FILE__,__LINE__, cudaMemset(bias_diffGPU, 0,
                                           sizeofStorageT * bias_numel));
    };

    void clearHist() {
        if (weight_diffGPU != NULL)
            checkCUDA(__FILE__,__LINE__, cudaMemset(weight_histGPU, 0,
                                           sizeofStorageT * weight_numel));
        if (bias_diffGPU != NULL)
            checkCUDA(__FILE__,__LINE__, cudaMemset(bias_histGPU, 0,
                                           sizeofStorageT * bias_numel));
    };

    void setWeights(std::vector<Tensor<StorageT> *> weights) {
        for (int i = 0; i < weights.size(); ++i) {
            if (weight_dataGPU != NULL &&
                weights[i]->name == name + ".weight") {
                if (numel(weight_dim) == numel(weights[i]->dim)) {
                    if (!same_dim(weight_dim, weights[i]->dim)) {
                        std::cout << "[Warning] " << name <<
                        ".weight is loaded with mismatched dimensions ";
                        std::cout << "need";
                        veciPrint(weight_dim);
                        std::cout << " vs. file";
                        veciPrint(weights[i]->dim);
                        std::cout << std::endl;
                    }
                    std::cout << " " << name << ".weight";
                    veciPrint(weights[i]->dim);
                    std::cout << " is set." << std::endl;
                    weights[i]->writeGPU(weight_dataGPU);
                } else {
                    std::cout << "[Warning] " << name <<
                    ".weight is found but not loaded because the numels are mismatched: ";
                    std::cout << "need";
                    veciPrint(weight_dim);
                    std::cout << " vs. file";
                    veciPrint(weights[i]->dim);
                    std::cout << std::endl;
                }
            }
            if (bias_dataGPU != NULL && weights[i]->name == name + ".bias") {
                if (numel(bias_dim) == numel(weights[i]->dim)) {
                    if (!same_dim(bias_dim, weights[i]->dim)) {
                        std::cout << "[Warning] " << name <<
                        ".bias is loaded with mismatched dimensions ";
                        std::cout << "need";
                        veciPrint(bias_dim);
                        std::cout << " vs. file";
                        veciPrint(weights[i]->dim);
                        std::cout << std::endl;
                    }
                    std::cout << " " << name << ".bias";
                    veciPrint(weights[i]->dim);
                    std::cout << " is set." << std::endl;
                    weights[i]->writeGPU(bias_dataGPU);
                } else {
                    std::cout << "[Warning] " << name <<
                    ".bias is found but not loaded because the numels are mismatched: ";
                    std::cout << "need";
                    veciPrint(bias_dim);
                    std::cout << " vs. file";
                    veciPrint(weights[i]->dim);
                    std::cout << std::endl;
                }

            }
        }

    };

    void saveWeights(FILE *fp) {
        if (weight_dataGPU != NULL) {
            Tensor <StorageT> *t = new Tensor<StorageT>(
                name + ".weight", weight_dim);
            t->readGPU(weight_dataGPU);
            t->write(fp);
            delete t;
        }

        if (bias_dataGPU != NULL) {
            Tensor <StorageT> *t = new Tensor<StorageT>(
                name + ".bias", bias_dim);
            t->readGPU(bias_dataGPU);
            t->write(fp);
            delete t;
        }

    };


    void setDiffs(std::vector<Tensor<StorageT> *> weights) {
        for (int i = 0; i < weights.size(); ++i) {
            if (weight_diffGPU != NULL &&
                weights[i]->name == name + ".weight_diff") {
                std::cout << " " << name << ".weight_diff";
                veciPrint(weights[i]->dim);
                std::cout << " is set." << std::endl;
                weights[i]->writeGPU(weight_diffGPU);
            }
            if (bias_diffGPU != NULL &&
                weights[i]->name == name + ".bias_diff") {
                std::cout << " " << name << ".bias_diff";
                veciPrint(weights[i]->dim);
                std::cout << " is set." << std::endl;
                weights[i]->writeGPU(bias_diffGPU);
            }
        }
    };

    void saveDiffs(FILE *fp) {
        if (weight_diffGPU != NULL) {
            Tensor <StorageT> *t = new Tensor<StorageT>(
                name + ".weight_diff", weight_dim);
            t->readGPU(weight_diffGPU);
            t->write(fp);
            delete t;
        }

        if (bias_diffGPU != NULL) {
            Tensor <StorageT> *t = new Tensor<StorageT>(
                name + ".bias_diff", bias_dim);
            t->readGPU(bias_diffGPU);
            t->write(fp);
            delete t;
        }

    };

    void update() {
        if (train_me) {
            if (weight_numel > 0 && weight_histGPU != NULL)
                bsa2b(weight_numel, weight_histGPU, weight_dataGPU);
            if (bias_numel > 0 && bias_histGPU != NULL)
                bsa2b(bias_numel, bias_histGPU, bias_dataGPU);
        }
    };

};


}// namespace fecnn

#endif  // LAYER_H_