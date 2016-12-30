#ifndef MEMORY_DATA_LAYER_H_
#define MEMORY_DATA_LAYER_H_

#include "Layer.h"

using namespace fecnn;

namespace fecnn {

class MemoryDataLayer : public DataLayer {
    std::vector<Tensor<StorageT>*> dataCPU; // [images,labels]
    public:
    std::vector<std::string> file_data;
    int batch_size;

    int numofitems(){
        return dataCPU[0]->dim[0];
    };
    void init(){
        train_me = false;
        std::cout<<"MemoryDataLayer "<<name<<" loading data: "<<std::endl;
        dataCPU.resize(file_data.size());
        for (int i =0;i<file_data.size();i++){
            dataCPU[i] = new Tensor<StorageT> (file_data[i],batch_size); // 将tensor文件中的内容读到内存中。
            dataCPU[i]->print();
        }
        if (phase!=Testing) shuffle();
    }

    MemoryDataLayer(std::string name_, Phase phase_, std::vector<std::string> file_data_, int batch_size_): DataLayer(name_), batch_size(batch_size_), file_data(file_data_){
        phase = phase_;
        init();
    };
    MemoryDataLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,       Training)
        SetOrDie(json, file_data    )
        SetValue(json, batch_size,  64)
        SetValue(json, random,      true)
        init();
    };
    ~MemoryDataLayer(){
        for (int i =0; i<dataCPU.size();i++){
            delete dataCPU[i];
        }
    };
    size_t Malloc(Phase phase_){
        if (phase == Training && phase_==Testing) return 0;
        
        if (!in.empty()){   std::cout<<"MemoryDataLayer shouldn't have any in's"<<std::endl; FatalError(__LINE__); }
        if (out.empty()){   std::cout<<"MemoryDataLayer should have some out's"<<std::endl; FatalError(__LINE__); }
        if (out.size()!=file_data.size()){  std::cout<<"MemoryDataLayer: # of out's should match the # of in's"<<std::endl; FatalError(__LINE__); }

        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;
        for (int i = 0;i < file_data.size(); i++){
            out[i]->need_diff = false;
            std::vector<int> data_dim = dataCPU[i]->dim;
            data_dim[0] = batch_size;
            memoryBytes += out[i]->Malloc(data_dim);
        }
        return memoryBytes;
    }
    void shuffle(){
        if (!random) return;
        std::vector<size_t> v = randperm(dataCPU[0]->numofitems(), rng);
        for(int i =0; i <dataCPU.size();i++){
            dataCPU[i]->permute(v);  // images和labels都以相同的规律打乱顺序，所以之后的对应关系还是成立的。
        }
    };

    void forward(Phase phase_){
        if (counter + batch_size >= dataCPU[0]->numofitems() ){
            ++epoch;
            if(phase!=Testing){
                shuffle();
                counter = 0;
            }
        }
        for(int i =0; i <dataCPU.size();i++){
            checkCUDA(__FILE__,__LINE__, 
                cudaMemcpy(out[i]->dataGPU, 
                    dataCPU[i]->CPUmem +  (size_t(counter) * size_t( dataCPU[i]->sizeofitem())), 
                    batch_size * dataCPU[i]->sizeofitem() * sizeofStorageT, 
                    cudaMemcpyHostToDevice) );     
        }
        counter+=batch_size;
        if (counter >= dataCPU[0]->numofitems()) counter = 0;
    };
};


}// namespace fecnn

#endif  // MEMORY_DATA_LAYER_H_