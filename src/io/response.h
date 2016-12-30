#ifndef RESPONSE_H
#define RESPONSE_H

#include "../common/common.h"
#include "../common/JSON.h"
#include "../common/my_cuda_helper.h"

using namespace fecnn;

namespace fecnn {
//////////////////////////////////////////////////////////////////////////////////////////////////
// Response and Layer
//////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
class Blob{
public:
    std::vector<int> dim;
    T* dataGPU;
    int count_;

    Blob():dim(),dataGPU(),count_(0){}

    inline size_t count() const { return fecnn::numel(dim);};

    size_t sizeofitem(){ return fecnn::sizeofitem(dim); };

    size_t numBytes(){ return sizeof(T)*(fecnn::numel(dim)); };

    size_t Malloc(std::vector<int> dim_){
        size_t memoryBytes = 0;
        checkCUDA(__FILE__,__LINE__, cudaMalloc(&dataGPU, numel(dim) * sizeof(T)) );
        memoryBytes += numel(dim) * sizeof(T);
        return memoryBytes;
    }

    void Reshape(std::vector<int>& shape){
        count_=1;
        dim.resize(shape.size());
        for (int i = 0; i < shape.size(); ++i) {
            count_ *= shape[i];
            dim[i] = shape[i];
        }
        checkCUDA(__FILE__,__LINE__, cudaMalloc(&dataGPU, numel(dim) * sizeof(T)) );
    }

    ~Blob(){
        if (dataGPU!=NULL) checkCUDA(__FILE__,__LINE__, cudaFree(dataGPU));
    }
};

class Response{
public:
    std::string name;
    cublasHandle_t cublasHandle;

    StorageT* dataGPU;
    StorageT* diffGPU; // 存储残差
    bool need_diff;
    std::vector<int> dim;


    inline size_t count() const { return fecnn::numel(dim);};

    size_t sizeofitem(){ return fecnn::sizeofitem(dim); };

    size_t numBytes(){ return sizeofStorageT*(fecnn::numel(dim)); };

    Response(std::string name_, bool need_diff_=false): name(name_), dataGPU(NULL), diffGPU(NULL), need_diff(need_diff_){

    };

    size_t Malloc(std::vector<int> dim_){
        size_t memoryBytes = 0;
        // 比如softmax层的in和out可能取同样的名称，不需要再分配内存了。
        // train的layer和test的layer会输出到同样的response中，要避免第二次malloc
        if (dataGPU==NULL){ 

            dim = dim_;

            std::cout<<"                                                                               ";
            std::cout<< (need_diff? "* " : "  ");

            std::cout<<name; veciPrint(dim);

            std::cout<<std::endl;

            checkCUDA(__FILE__,__LINE__, cudaMalloc(&dataGPU, numel(dim) * sizeofStorageT) );
            memoryBytes += numel(dim) * sizeofStorageT;

            if (need_diff){
                checkCUDA(__FILE__,__LINE__, cudaMalloc(&diffGPU, numel(dim) * sizeofStorageT) );
                memoryBytes += numel(dim) * sizeofStorageT;
            }
        }else{
            if (!same_dim(dim, dim_)){

                std::cerr<<std::endl<<"Response["<< name <<"] Malloc dimension mis-matched: ";
                veciPrint(dim);
                std::cerr<<" vs ";
                veciPrint(dim_);
                std::cerr<<std::endl;

                if (numel(dim)!=numel(dim_)) FatalError(__LINE__);
            }
        }
        return memoryBytes;
    };


    ~Response(){
        if (dataGPU!=NULL) checkCUDA(__FILE__,__LINE__, cudaFree(dataGPU));
        if (diffGPU!=NULL) checkCUDA(__FILE__,__LINE__, cudaFree(diffGPU));
    };

    void clearDiff(){
        if (diffGPU!=NULL){
            checkCUDA(__FILE__,__LINE__, cudaMemset(diffGPU, 0, sizeofStorageT * numel(dim)));
        }
    };

    // 只在debug模式下使用
    ComputeT ameanData(){
        if (dataGPU!=NULL){
            ComputeT result;
            size_t n = numel(dim);
            //std::cout<<"n="<<n<<std::endl;
            //std::cout<<"cublasHandle="<<cublasHandle<<std::endl;
            //std::cout<<"dataGPU="<<dataGPU<<std::endl;
            checkCUBLAS(__FILE__,__LINE__, GPUasum(cublasHandle, n, dataGPU, 1, &result));
            result /= ComputeT(n);
            return result;
        }else{
            return -1;
        }
    };
    ComputeT ameanDiff(){
        if (diffGPU!=NULL){
            ComputeT result;
            size_t n = numel(dim);
            checkCUBLAS(__FILE__,__LINE__, GPUasum(cublasHandle, n, diffGPU, 1, &result));
            result /= ComputeT(n);
            return result;
        }else{
            return -1;
        }
    };
};



}// namespace fecnn

#endif  // RESPONSE_H