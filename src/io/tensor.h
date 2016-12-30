#ifndef TENSOR_H_
#define TENSOR_H_

#include "../common/common.h"

using namespace fecnn;

namespace fecnn {

//////////////////////////////////////////////////////////////////////////////////////////////////
// File format
//////////////////////////////////////////////////////////////////////////////////////////////////

uint8_t typeID(std::type_index t){
    if (t==typeid(half))        return uint8_t(0);
    if (t==typeid(float))       return uint8_t(1);
    if (t==typeid(double))      return uint8_t(2);
    if (t==typeid(uint8_t))     return uint8_t(3);
    if (t==typeid(uint16_t))    return uint8_t(4);
    if (t==typeid(uint32_t))    return uint8_t(5);
    if (t==typeid(uint64_t))    return uint8_t(6);
    if (t==typeid(int8_t))      return uint8_t(7);
    if (t==typeid(int16_t))     return uint8_t(8);
    if (t==typeid(int32_t))     return uint8_t(9);
    if (t==typeid(int64_t))     return uint8_t(10);
    if (t==typeid(char))        return uint8_t(11);
    if (t==typeid(bool))        return uint8_t(12);
    FatalError(__LINE__);       return uint8_t(255);
}


template <class T>
class Tensor{ //多维数据阵列
public:
    std::vector<int> dim; //存放维度信息，比如一个3*4*5的立方体，dim=[3,4,5]
    T* CPUmem;  // 存放在CPU上的内存
    std::string name;

    // compile will check if your time is not correct for writeGPU and readGPU
    void writeGPU(T* GPUmem){
        cudaMemcpy(GPUmem, CPUmem, numel()*sizeof(T), cudaMemcpyHostToDevice);
    };

    void readGPU(T* GPUmem){
        cudaMemcpy(CPUmem, GPUmem, numel()*sizeof(T), cudaMemcpyDeviceToHost);
    };

    Tensor(): CPUmem(NULL){};

    size_t numel(){ return fecnn::numel(dim); };

    size_t numBytes(){ return sizeof(T)*numel(); };

    int numofitems(){ return dim[0]; };

    size_t sizeofitem(){ return fecnn::sizeofitem(dim); };

    ~Tensor(){
        if (CPUmem!=NULL)   delete[] CPUmem;
    };

    void initialize(T val){
        for (size_t i=0;i<numel();++i){
            CPUmem[i]=val;
        }
    };


    T* read(FILE* fp,int batch_size=1){
        if (CPUmem!=NULL){
            delete[] CPUmem;
            CPUmem = NULL;
        }

        size_t read_cnt;

        uint8_t myTypeid = typeID(typeid(T));
        uint32_t myTypesizeof = uint32_t(sizeof(T));

        uint8_t fpTypeid;       read_cnt = fread((void*)(&fpTypeid), sizeof(uint8_t), 1, fp);       if (read_cnt!=1) return NULL;
        uint32_t fpTypesizeof;  read_cnt = fread((void*)(&fpTypesizeof), sizeof(uint32_t), 1, fp);  if (read_cnt!=1) return NULL;

        // 限定tensor中的数据格式必须为float型的
        if (myTypeid!=fpTypeid || myTypesizeof!=fpTypesizeof){
            std::cerr<<"Tensor read error！"<<std::endl; FatalError(__LINE__);
        }else{
            int lenName;
            read_cnt = fread((void*)(&lenName), sizeof(int), 1, fp);
            if (read_cnt!=1) return NULL;
            name.resize(lenName);
            if (lenName>0){
                read_cnt = fread((void*)(name.data()), sizeof(char), lenName, fp);
                if (read_cnt!=lenName) return NULL;
            }
            int nbDims;
            read_cnt = fread((void*)(&nbDims), sizeof(int), 1, fp);
            if (read_cnt!=1) return NULL;
            dim.resize(nbDims);
            if (nbDims>0){
                read_cnt = fread((void*)(&dim[0]), sizeof(int), nbDims, fp);
                if (read_cnt!=nbDims) return NULL;
            }

            size_t n = numel();
            Malloc(batch_size);
            read_cnt = fread((void*)(CPUmem), sizeof(T), n, fp);
            if (read_cnt!=n){
                delete [] CPUmem;
                CPUmem = NULL;
                return NULL;
            }
        }

        return CPUmem;
    };

    void Malloc(int batch_size){
        size_t n = numel();
        std::cout<<"  ";        memorySizePrint(n*sizeof(T));   std::cout<<std::endl;

        if (batch_size==1 || dim[0]%batch_size ==0){
            CPUmem = new T [n];
        }else{
            int dim0 =  (dim[0]/batch_size + 1) * batch_size; // 现在dim0能被batch_size正好整除了
            size_t oversize = n/dim[0] * dim0;  // 新的大小，将会有多的位置是空数据
            CPUmem = new T [oversize];
            memset((void*)(CPUmem+n),0, (oversize-n)*sizeof(T));
        }
    };

    T* read(std::string filename,int batch_size=1){
        FILE* fp = fopen(filename.c_str(),"rb");
        while (fp==NULL) {
            std::cerr<<"Tensor:read: fail to open file "<<filename<<". Please provide it first. Will retry after 5 seconds."<<std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            fp = fopen(filename.c_str(),"rb");
        }
        read(fp,batch_size);
        fclose(fp);
        return CPUmem;
    };

    //write without header
    void writeHeader(FILE* fp, std::vector<int> dim2write){
        uint8_t myTypeid = typeID(typeid(T));
        fwrite((void*)(&myTypeid), sizeof(uint8_t), 1, fp);
        uint32_t typesizeof = uint32_t(sizeof(T));
        fwrite((void*)(&typesizeof), sizeof(uint32_t), 1, fp);
        int lenName = name.size();
        fwrite((void*)(&lenName), sizeof(int), 1, fp);
        if (lenName>0) fwrite((void*)(name.data()), sizeof(char), lenName, fp);
        int nbDims = dim2write.size();
        fwrite((void*)(&nbDims), sizeof(int), 1, fp);
        if (nbDims>0) fwrite((void*)(&dim2write[0]), sizeof(int), nbDims, fp);
        if (ferror (fp)){
            std::cerr << "disk writing failed"<<std::endl;
            FatalError();
        }
    };

    void writeData(FILE* fp, size_t max_size = 0){
        size_t n = numel();
        if (max_size !=0 ) n = min(n,max_size);
        if (n>0){
            fwrite((void*)(CPUmem), sizeof(T), n, fp);
            if (ferror (fp)){
                std::cerr << "disk writing failed" << std::endl;
                FatalError();
            }
        }
    };

    // 调用solver.saveWeights(solver.path + ".fecnn");时要用到，往外部文件中保存权重信息。
    void write(FILE* fp){
        writeHeader(fp,dim);
        writeData(fp);
    };

    void write(std::string filename){
        FILE* fp = fopen(filename.c_str(),"wb");
        while (fp==NULL) {
            std::cerr<<"Tensor::write: fail to open file "<<filename<<". Will retry after 5 seconds."<<std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            fp = fopen(filename.c_str(),"wb");
        }
        write(fp);
        fclose(fp);
        return;
    };

    Tensor(std::string filename, int batch_size=1): CPUmem(NULL){ read(filename,batch_size); };

    Tensor(FILE* fp): CPUmem(NULL){ read(fp); };

    Tensor(std::vector<int> dim_): dim(dim_){ CPUmem = new T [numel()]; };

    Tensor(std::vector<int> dim_, T initValue): dim(dim_){
        int n = numel();
        CPUmem = new T [n];
        if (initValue == T(0))
            memset(CPUmem, 0, n*sizeof(T));
        else
            for (int i=0;i<n;++i) CPUmem[i] = initValue;

    };

    Tensor(std::string name_, std::vector<int> dim_): name(name_),dim(dim_){
        CPUmem = new T [numel()];
    };

    void permute(std::vector<size_t> v){
        size_t nbItems = numofitems();
        size_t sizeofitem_ = sizeofitem();
        size_t nbBytes = sizeofitem_ * sizeof(T);
        T* CPUmemNew = new T[numel()];
        memcpy(CPUmemNew, CPUmem, nbItems * nbBytes);
        for (size_t i=0;i<nbItems;++i){
            memcpy(CPUmem+i*sizeofitem_, CPUmemNew+v[i]*sizeofitem_, nbBytes);
        }
        delete [] CPUmemNew;
    };

    // 打印CPUmem中的数据
    void print(){
        std::cout<<"  name:"<<name<<" dim"; veciPrint(dim); std::cout<<std::endl;
    };
};


// 在读取.fecnn格式数据时调用
template <class T>
std::vector<Tensor<T>*> readTensors(std::string filename, size_t max_count = SIZE_MAX){

    FILE* fp = fopen(filename.c_str(),"rb");

    while (fp==NULL) {
        std::cerr<<"readTensors: fail to open file "<<filename<<". Please provide it first. Will retry after 5 seconds."<<std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
        fp = fopen(filename.c_str(),"rb");
    }

    std::vector<Tensor<T>*> tensors;
    size_t count = 0;
    while (feof(fp)==0) {
        tensors.push_back(new Tensor<T>(fp));
        count++;
        if (count>=max_count) break;
        int c = getc(fp);   // 要读出来再放回去，以检查是否已经读到末尾，这样feof就能检测到错误，并退出循环。
        ungetc(c, fp);
    }
    fclose(fp);
    return tensors;
}

template <class T>
void writeTensors(std::string filename, std::vector<Tensor<T>*> tensors){
    FILE* fp = fopen(filename.c_str(),"wb");
    while (fp==NULL) {
        std::cerr<<"writeTensors: fail to open file "<<filename<<". Disk full? Will retry after 5 seconds."<<std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
        fp = fopen(filename.c_str(),"wb");
    }

    for(int i=0;i<tensors.size();++i){
        tensors[i]->write(fp);
    }
    fclose(fp);
}



}// namespace fecnn

#endif  // TENSOR_H