#ifndef COMMON_H_
#define COMMON_H_

// 需要与处理的数据类型保持一致
#define StorageT float
#define ComputeT float
#define sizeofStorageT 4
#define sizeofComputeT 4
#define CPUStorage2ComputeT(x) (x)
#define CPUCompute2StorageT(x) (x)
#define GPUStorage2ComputeT(x) (x)
#define GPUCompute2StorageT(x) (x)
#define GPUgemm cublasSgemm
#define GPUasum cublasSasum
#define ISNAN(x) (std::isnan(x)) //测试某个浮点数是否是 非数字
#define ComputeT_MIN FLT_MIN //1.17549e-38


//////////////////////////////////////////////////////////////////////////////////////////////////
// Includes
//////////////////////////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <typeinfo>
#include <typeindex>
#include <thread>
#include <chrono>
#include <future>
#include <cublas_v2.h>
#include <curand.h>
#include <sys/time.h>

namespace fecnn {
//////////////////////////////////////////////////////////////////////////////////////////////////
// Type definition
//////////////////////////////////////////////////////////////////////////////////////////////////

enum Filler { Xavier, Gaussian, Constant }; // 权重或偏执的初始化方式
enum Pool { Max, Average, Sum };
enum ActivateMode{Sigmoid,ReLU,Tanh};
enum LossObjective { MultinomialLogistic_StableSoftmax, MultinomialLogistic, SmoothL1, Contrastive, EuclideanSSE, HingeL1, HingeL2, SigmoidCrossEntropy, Infogain }; //关于Loss层各种实现的解释可以参考caffe，参见：http://blog.csdn.net/u011762313/article/details/47361571
enum Phase { Training, Testing, TrainingTesting }; // TrainingTesting表示两个phase均可用时
enum LRPolicy { LR_fixed, LR_step, LR_exp, LR_inv, LR_multistep, LR_poly, LR_sigmoid };
enum SolverAlgorithm { SGD, AdaDelta, AdaGrad, Adam, NAG, RMSprop};
enum Regularizer { L2, L1 };


ComputeT anyval;
ComputeT oneval = 1;
ComputeT zeroval = 0;
const void* one = static_cast<void *>(&oneval);
const void* zero = static_cast<void *>(&zeroval);
const ComputeT* oneComputeT = &oneval;
const ComputeT* zeroComputeT = &zeroval;

//////////////////////////////////////////////////////////////////////////////////////////////////
// Debugging utility
//////////////////////////////////////////////////////////////////////////////////////////////////

void FatalError(const int lineNumber=0) {
    std::cerr << "FatalError";
    // if (fileName!="") std::cerr<<" at FILE "<<fileName;
    if (lineNumber!=0) std::cerr<<" at LINE "<<lineNumber;
    std::cerr << ". Program Terminated." << std::endl;
    cudaDeviceReset();
    exit(EXIT_FAILURE);
}

void checkCUDA(const char* fileName,const int lineNumber, cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA failure at FILE "<<fileName<<" at LINE " << lineNumber << ": " << status << std::endl;
        FatalError();
    }
}

void checkCUBLAS(const char* fileName,const int lineNumber, cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS failure at FILE "<<fileName<<" at LINE " << lineNumber << ": ";
        switch (status) {
            case CUBLAS_STATUS_SUCCESS:             std::cerr << "CUBLAS_STATUS_SUCCESS" << std::endl; break;
            case CUBLAS_STATUS_NOT_INITIALIZED:     std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED" << std::endl; break;
            case CUBLAS_STATUS_ALLOC_FAILED:        std::cerr << "CUBLAS_STATUS_ALLOC_FAILED" << std::endl; break;
            case CUBLAS_STATUS_INVALID_VALUE:       std::cerr << "CUBLAS_STATUS_INVALID_VALUE" << std::endl; break;
            case CUBLAS_STATUS_ARCH_MISMATCH:       std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH" << std::endl; break;
            case CUBLAS_STATUS_MAPPING_ERROR:       std::cerr << "CUBLAS_STATUS_MAPPING_ERROR" << std::endl; break;
            case CUBLAS_STATUS_EXECUTION_FAILED:    std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED" << std::endl; break;
            case CUBLAS_STATUS_INTERNAL_ERROR:      std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR" << std::endl; break;
            case CUBLAS_STATUS_NOT_SUPPORTED:       std::cerr << "CUBLAS_STATUS_NOT_SUPPORTED" << std::endl; break;
            case CUBLAS_STATUS_LICENSE_ERROR:       std::cerr << "CUBLAS_STATUS_LICENSE_ERROR" << std::endl; break;
        }
        FatalError();
    }
    checkCUDA(fileName,lineNumber,cudaGetLastError());
}

unsigned long long get_timestamp() {
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (unsigned long long)now.tv_sec * 1000000;
}

unsigned long long ticBegin;

// 计时开始
unsigned long long tic() {
    ticBegin = get_timestamp();
    return ticBegin;
}

// 计时结束
unsigned long long toc() {
    unsigned long long ticEnd = get_timestamp();
    unsigned long long delta = ticEnd - ticBegin;
    // std::cout << "Time passes " << delta << " microseconds" <<std::endl;
    std::cout << "Time passes " << delta/1000000.0 << " seconds" <<std::endl;
    ticBegin = ticEnd;
    return delta;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// HALF computation ultility
//////////////////////////////////////////////////////////////////////////////////////////////////

static __inline__ __device__ __host__ int ishnan(half h) {
    // When input is NaN, exponent is all ones and mantissa is non-zero.
    return (h.x & 0x7c00U) == 0x7c00U && (h.x & 0x03ffU) != 0;
}

half cpu_float2half(float f) {
    half ret;

    unsigned x = *((int*)(void*)(&f));
    unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
    unsigned sign, exponent, mantissa;

    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000) {
        ret.x = 0x7fffU;
        return ret;
    }

    sign = ((x >> 16) & 0x8000);

    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefff) {
        ret.x = sign | 0x7c00U;
        return ret;
    }
    if (u < 0x33000001) {
        ret.x = (sign | 0x0000);
        return ret;
    }

    exponent = ((u >> 23) & 0xff);
    mantissa = (u & 0x7fffff);

    if (exponent > 0x70) {
        shift = 13;
        exponent -= 0x70;
    } else {
        shift = 0x7e - exponent;
        exponent = 0;
        mantissa |= 0x800000;
    }
    lsb = (1 << shift);
    lsb_s1 = (lsb >> 1);
    lsb_m1 = (lsb - 1);

    // Round to nearest even.
    remainder = (mantissa & lsb_m1);
    mantissa >>= shift;
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
        ++mantissa;
        if (!(mantissa & 0x3ff)) {
            ++exponent;
            mantissa = 0;
        }
    }

    ret.x = (sign | (exponent << 10) | mantissa);

    return ret;
}


float cpu_half2float(half h) {
    unsigned sign = ((h.x >> 15) & 1);
    unsigned exponent = ((h.x >> 10) & 0x1f);
    unsigned mantissa = ((h.x & 0x3ff) << 13);

    if (exponent == 0x1f) {  /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) {  /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1;  /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }

    int temp = ((sign << 31) | (exponent << 23) | mantissa);

    return *((float*)((void*)&temp));
}


bool operator <(const half& x, const half& y) {
    return cpu_half2float(x) < cpu_half2float(y);
}

std::ostream& operator<< (std::ostream& stream, const half& x) {
    stream << cpu_half2float(x);
    return stream;
}


//////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
//////////////////////////////////////////////////////////////////////////////////////////////////

bool is_file_exist(const std::string& fileName){
    std::ifstream infile(fileName);
    return infile.good();
}

// 将以字节为单位的内存转换成合适的内存单位表示
void memorySizePrint(size_t bytes){
    if (bytes<512){
        std::cout<<bytes<<" Bytes";
    }else if (bytes<512.0*1024){
        std::cout<<(bytes/1024.0)<<" KB";
    }else if (bytes<512.0*1024*1024){
        std::cout<<(bytes/(1024.0*1024.0))<<" MB";
    }else if (bytes<512.0*1024*1024*1024){
        std::cout<<(bytes/(1024.0*1024.0*1024.0))<<" GB";
    }else if (bytes<512.0*1024*1024*1024*1024){
        std::cout<<(bytes/(1024.0*1024.0*1024.0*1024.0))<<" TB";
    }else{
        std::cout<<(bytes/(1024.0*1024.0*1024.0*1024.0*1024.0))<<" PB";
    }
}

// 打印整形列表的内容
void veciPrint(const std::vector<int>& v){
    std::cout<<"["<<v.size()<<"]={";
    if (v.size()>0) std::cout<<v[0];
    if (v.size()>1){
        for (int i=1;i<v.size();++i){
            std::cout<<","<<v[i];
        }
    }
    std::cout<<"}";
}

// @TODO 没有输出v.size()
void vecfPrint(const std::vector<ComputeT>& v){
    std::cout<<"[";
    if (v.size()>0) std::cout<<v[0];
    if (v.size()>1){
        for (int i=1;i<v.size();++i){
            std::cout<<","<<v[i];
        }
    }
    std::cout<<"]";
}

// 指定列表的大小，将可变参数的内容填入列表中。
std::vector<int> veci(int n, ...){
    std::vector<int> v;
    if (n==0) return v;
    va_list ap;
    va_start(ap, n);
    for(int i = 0; i < n; i++) {
        v.push_back(va_arg(ap, int));
    }
    va_end(ap);
    return v;
}

std::vector<std::string> vecs(int n, ...){
    std::vector<std::string> v;
    if (n==0) return v;
    va_list ap;
    va_start(ap, n);
    for(int i = 0; i < n; i++) {
        v.push_back(std::string(va_arg(ap, char*)));
    }
    va_end(ap);
    return v;
}

// 将一段用 “,” 分隔的字符串封装成列表
std::vector<std::string> getStringVector(std::string input){
    std::vector<std::string> ret;
    while (input.size()>0){
        int e = input.find(",");
        if (e==std::string::npos){
            e = input.size();
        }
        std::string first = input.substr(0,e);
        ret.push_back(first);
        if(e+1<input.size())
            input=input.substr(e+1);
        else
            break;
    }
    return ret;
}

// input=[0,1,2],[0,1,2,3,4,5]
std::vector<std::vector<int> > getIntVectorVector(std::string input){
    //remove all space
    input.erase(remove_if(input.begin(), input.end(), (int(*)(int))isspace), input.end());

    std::vector<std::vector<int> > ret;
    while (input.size()>0){
        int e;
        if (input[0]=='['){
            ret.resize(ret.size()+1);
            e=0;
        }else if (input[0]==','){
            e=0;
        }else if (input[0]==']'){
            e=0;
        }else{
            e = input.find(",");
            if (e==std::string::npos){
                e = input.size();
            }
            int f = input.find("]");
            if (f==std::string::npos){
                f = input.size();
            }
            e = min(e,f);
            std::string first = input.substr(0,e);
            ret[ret.size()-1].push_back(stoi(first));
        }
        if(e+1<input.size())
            input=input.substr(e+1);
        else
            break;
    }
    return ret;
}

// matlab中的一个函数
// returns the number of elements, n, in array A,
size_t numel(const std::vector<int>& dim){
    size_t res = 1;
    for (int i=0;i<dim.size();++i) res *= (size_t)(dim[i]);
    return res;
}

size_t sizeofitem(const std::vector<int>& dim){
    size_t res = 1;
    for (int i=1;i<dim.size();++i) res *= (size_t)(dim[i]); //第0维表示数据的个数batch，所以剩下的维度相乘表示item的大小
    return res;
}

size_t numspel(const std::vector<int>& dim){
    size_t res = 1;
    for (int i=2;i<dim.size();++i) res *= (size_t)(dim[i]);
    return res;
}

bool same_dim(const std::vector<int>& dimA, const std::vector<int>& dimB){
    if (dimA.size()!=dimB.size()) return false;
    for (int i=0;i<dimA.size();++i){
        if (dimA[i]!=dimB[i]) return false;
    }
    return true;
}

bool same_dim_EC(const std::vector<int>& dimA, const std::vector<int>& dimB){
    if (dimA.size()!=dimB.size()) return false;
    if (dimA[0]!=dimB[0]) return false;
    for (int i=2;i<dimA.size();++i)
        if (dimA[i]!=dimB[i])
            return false;
    return true;
}

size_t checkNaN(StorageT* dataGPU, size_t n){
    StorageT* CPUmem = new StorageT[n];
    cudaMemcpy(CPUmem, dataGPU, n*sizeofStorageT, cudaMemcpyDeviceToHost);
    size_t countNaN = 0;
    for (size_t i=0;i<n;++i) if (ISNAN(CPUmem[i])) ++countNaN;
    if (countNaN>0){
        std::cout<<"        checkNaN result: "<<countNaN<<" out of "<<n<<" ("<< 100*ComputeT(countNaN)/n<< "\045) values are NaN, "<<n-countNaN<<" are not NaN."; //<<std::endl;
    }
    delete [] CPUmem;
    return countNaN;
}

// 返回一组随机的下标值
//returns a row vector containing a random permutation of the integers from 0 to n exclusive. 
std::vector<size_t> randperm(size_t n, std::mt19937& rng){
    std::vector<size_t> v(n);
    for (size_t i=0;i<n;++i) v[i]=i;

    shuffle ( v.begin(), v.end(), rng );
    return v;
}

// 返回一个列表，其中第一个元素表示列表v中最小元素的编号，依次类推，最后一个元素表示列表v中最大元素的编号
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}

std::string int_to_str(const int i) {
    std::ostringstream s;
    s << i;
    return s.str();
}




}// namespace fecnn

#endif  // COMMON_H_