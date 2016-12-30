#ifndef NET_H_
#define NET_H_

#include "layers/Layer.h"
#include "layers/DataLayer.h"
#include "layers/ActivationLayer.h"
#include "layers/ConvolutionLayer.h"
#include "layers/InnerProductLayer.h"
#include "layers/LossLayer.h"
#include "layers/MemoryDataLayer.h"
#include "layers/PoolingLayer.h"
#include "layers/SoftmaxLayer.h"

using namespace fecnn;

namespace fecnn {
//////////////////////////////////////////////////////////////////////////////////////////////////
// Net
//////////////////////////////////////////////////////////////////////////////////////////////////

class Net{
public:
    Phase phase;
    std::vector<Layer*> layers;
    std::vector<Response*> responses;
    std::vector<LossLayer*> loss_layers;
    
    int GPU;
    bool debug_mode;
    int train_iter; // 每步train向前forword几次。
    int test_iter; // 每步test向前forword几次。
    int display_iter;

    cublasHandle_t cublasHandle;

    void init(JSON* architecture_obj){
        checkCUDA(__FILE__,__LINE__,cudaSetDevice(GPU));

        checkCUBLAS(__FILE__,__LINE__, cublasCreate(&cublasHandle) );

        for (int l=0;l<architecture_obj->array.size();++l){

            JSON* p = (JSON*)(architecture_obj->array[l]);

            std::string type = p->member["type"]->returnString();

            Layer* pLayer;
            Response* pResponse;

                 if (0==type.compare("MemoryData"))     pLayer = new MemoryDataLayer(p);
            else if (0==type.compare("Convolution"))            pLayer = new ConvolutionLayer(p);
            else if (0==type.compare("InnerProduct"))           pLayer = new InnerProductLayer(p);
            else if (0==type.compare("Pooling"))                pLayer = new PoolingLayer(p);
            else if (0==type.compare("Activation"))             pLayer = new ActivationLayer(p);
            else if (0==type.compare("Softmax"))                pLayer = new SoftmaxLayer(p);
            else if (0==type.compare("Loss"))                  {pLayer = new LossLayer(p); loss_layers.push_back((LossLayer*)pLayer);   }
            else { std::cout<<"ERROR: recognizable layer in JSON file: "<<type<<std::endl; FatalError(__LINE__);};

            pLayer->cublasHandle = cublasHandle;
            pLayer->GPU = GPU;

            addLayer(pLayer);

            if (p->member.find("out") != p->member.end()){
                std::vector<std::string> out = p->member["out"]->returnStringVector();
                for (int i=0;i<out.size();i++){
                    pResponse = getResponse(out[i]);
                    if (pResponse==NULL){
                        pResponse = addResponse(new Response(out[i]));
                        pResponse->cublasHandle = cublasHandle;
                    }
                    pLayer->addOut(pResponse);
                }
            }

            if (p->member.find("in") != p->member.end()){
                std::vector<std::string> in = p->member["in"]->returnStringVector();
                for (int i=0;i<in.size();i++){
                    pResponse = getResponse(in[i]);
                    if (pResponse==NULL){
                        pResponse = addResponse(new Response(in[i]));
                        pResponse->cublasHandle = cublasHandle;
                    }
                    pLayer->addIn(pResponse);
                }
            }
        }

    };


    Net(std::string filename){
        JSON* test_obj = new JSON;
        JSON* architecture_obj = new JSON;
        parseNetworkJSON(filename, NULL, test_obj, architecture_obj);
        SetValue(test_obj, GPU,             0)
        SetValue(test_obj, debug_mode,      false)
        SetValue(test_obj, display_iter,    1)

        init(architecture_obj);

        delete test_obj;
        delete architecture_obj;
    };

    Net(JSON* architecture_obj, int GPU_ = 0): GPU(GPU_){
        init(architecture_obj);
    };

    ~Net(){
        checkCUDA(__FILE__,__LINE__,cudaSetDevice(GPU));

        for (int i=0;i<layers.size();++i){
            delete layers[i];
        }
        for (int i=0;i<responses.size();++i){
            delete responses[i];
        }
        checkCUBLAS(__FILE__,__LINE__, cublasDestroy(cublasHandle) );
    };

    Layer* getLayer(std::string name){
        for (int l=0; l<layers.size();++l){
            if (layers[l]->name == name){
                return layers[l];
            }
        }
        return NULL;
    };

    Response* getResponse(std::string name){
        for (int r=0; r<responses.size();++r){
            if (responses[r]->name == name){
                return responses[r];
            }
        }
        return NULL;
    };

    Layer* addLayer(Layer* pLayer){
        layers.push_back(pLayer);
        return pLayer;
    };

    Response* addResponse(Response* pResponse){
        responses.push_back(pResponse);
        return pResponse;
    };

    void randInit(){
        checkCUDA(__FILE__,__LINE__,cudaSetDevice(GPU));
        for (int l=0; l<layers.size();++l){
            layers[l]->randInit();
        }
    };

    void loadWeights(std::vector<Tensor<StorageT>*> weights, bool diff=false){
        checkCUDA(__FILE__,__LINE__,cudaSetDevice(GPU));
        // 每层都可以根据它的名字找到对应的weight
        for (int l=0; l<layers.size();++l){
            layers[l]->setWeights(weights);
            if (diff) layers[l]->setDiffs(weights);
        }
    };

    void loadWeights(std::string filename, bool diff=false){
        std::cout<< "====================================================================================================================================="<<std::endl;

        std::vector<Tensor<StorageT>*> weights = readTensors<StorageT>(filename);
        loadWeights(weights, diff);

        // release memory for the weights
        for (int i=0; i<weights.size();++i){
            delete weights[i];
        }
    };

    void saveWeights(std::string filename, bool diff=false){
        FILE* fp = fopen(filename.c_str(),"wb");
        while (fp==NULL) {
            std::cerr<<"Net::saveWeights: fail to open file "<<filename<<". Please provide it first. Will retry after 5 seconds."<<std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            fp = fopen(filename.c_str(),"wb");
        }

        for (int l=0; l<layers.size();++l){
            layers[l]->saveWeights(fp);
            if (diff) layers[l]->saveDiffs(fp);
        }
        fclose(fp);
    };

    size_t Malloc(Phase phase_ = Testing){
        checkCUDA(__FILE__,__LINE__,cudaSetDevice(GPU));

        phase = phase_;

        std::cout<< "====================================================================================================================================="<<std::endl;
        std::cout<< "  Layers:                                                                        Responses:                                          "<<std::endl;
        std::cout<< "====================================================================================================================================="<<std::endl;

        size_t memoryBytes = 0;

        for (int l=0;l<layers.size();++l){
            memoryBytes += layers[l]->Malloc(phase);
        }

        std::cout<< "====================================================================================================================================="<<std::endl;
        std::cout<< "[Net] GPU " << GPU << ": Total GPU memory: ";    memorySizePrint(memoryBytes);   std::cout<<std::endl;

        return memoryBytes;
    };

    void forward(){
        for (int l=0; l<layers.size();++l){
            if (layers[l]->phase == phase || layers[l]->phase == TrainingTesting){ // 这里就把用于train和用于test的数据层做了区分
                // if (debug_mode){
                //     std::cout<<"[Forward] Layer["<<l<<"] "<<layers[l]->name;
                //     ComputeT avg;
                //     avg = layers[l]->ameanWeightData(); if (avg!=-1) std::cout<<" weight.data: "<< avg;
                //     avg = layers[l]->ameanBiasData();   if (avg!=-1) std::cout<<" bias.data: "<< avg;
                //     tic();
                // }

                layers[l]->forward(phase);

                // if (debug_mode){
                //     checkCUDA(__FILE__,__LINE__,cudaDeviceSynchronize()); checkCUDA(__FILE__,__LINE__,cudaGetLastError());
                //     if (layers[l]->out.size()>0){
                //         for (size_t o=0; o<layers[l]->out.size();++o){
                //             ComputeT avg = layers[l]->out[o]->ameanData();
                //             if (avg!=-1) std::cout<<" out[" << o << "].data("<< layers[l]->out[o]->name << "): " << avg;
                //             layers[l]->out[o]->checkNaN();
                //         }
                //     }
                //     std::cout<<std::endl; toc();
                // }
            }
        }
    };

    void backward(){
        for (int r=0;r<responses.size();++r){
            responses[r]->clearDiff();
        }

        for (int l=layers.size()-1;l>=0; --l){
            if (layers[l]->phase == phase || layers[l]->phase == TrainingTesting){

                // if (debug_mode){
                //     std::cout<<"[Backward] Layer["<<l<<"] "<<layers[l]->name;
                //     tic();
                // }

                layers[l]->backward(phase);

                // if (debug_mode){
                //     checkCUDA(__FILE__,__LINE__,cudaDeviceSynchronize()); checkCUDA(__FILE__,__LINE__,cudaGetLastError());
                //     ComputeT avg;
                //     avg = layers[l]->ameanWeightDiff(); if (avg!=-1) std::cout<<" weight.diff: "<<avg;
                //     avg = layers[l]->ameanBiasDiff();   if (avg!=-1) std::cout<<" bias.diff: "<<avg;

                //     if (layers[l]->in.size()>0){
                //         for (size_t i=0;i<layers[l]->in.size();++i){
                //             avg = layers[l]->in[i]->ameanDiff(); if (avg!=-1) std::cout<<" in[" << i << "].diff(" << layers[l]->in[i]->name << "): "<<avg;
                //         }
                //     }
                //     std::cout<<std::endl; toc();
                // }
            }
        }
    };

    void update(){
        for (int l=0; l<layers.size();++l){
            layers[l]->update();
        }
    };

    void resetLoss(){
        for (int l=0; l<loss_layers.size();++l){
            loss_layers[l]->result = ComputeT(0);
            loss_layers[l]->loss   = ComputeT(0);
        }
    };

    void eval(bool sync){
        checkCUDA(__FILE__,__LINE__,cudaSetDevice(GPU));
        for (int l=0; l<loss_layers.size();++l){
            if (loss_layers[l]->phase == phase || loss_layers[l]->phase == TrainingTesting)
                loss_layers[l]->eval();
        }
        if(sync) checkCUDA(__FILE__,__LINE__,cudaDeviceSynchronize());
    };

    void stepTest(bool sync){
        checkCUDA(__FILE__,__LINE__,cudaSetDevice(GPU));

        resetLoss();
        for (int i=0; i < test_iter; ++i){
            forward();
            eval(false);
        }
        for (int l=0; l<loss_layers.size();++l){
            loss_layers[l]->result /= test_iter;
            loss_layers[l]->loss   /= test_iter;
        }

        if(sync) checkCUDA(__FILE__,__LINE__,cudaDeviceSynchronize());
    };


    void stepTrain(bool sync){ // 默认为false
        checkCUDA(__FILE__,__LINE__,cudaSetDevice(GPU));

        this->update(); // 在solve方法中已完成了斜率的设置，直接更新权重即可

        resetLoss();
        for (int l=0; l<layers.size();++l){
            layers[l]->clearDiff();
        }

        for (int i=0; i < train_iter; ++i){ //默认train_iter=1
            forward();
            backward();
        }

        for (int l=0; l<loss_layers.size();++l){ //默认loss_layers.size()=1
            loss_layers[l]->result /= train_iter;
            loss_layers[l]->loss   /= train_iter;
        }

        if(sync) checkCUDA(__FILE__,__LINE__,cudaDeviceSynchronize());
    };


    // for testing or extract feature, have to call after Malloc
    std::vector<ComputeT> test(){

        phase = Testing;

        std::vector<ComputeT> result(loss_layers.size(), 0);

        DataLayer* pDataLayer = NULL;
        for (int l=0; l<layers.size();++l){
            if (layers[l]->phase == phase || layers[l]->phase == TrainingTesting){
                if (layers[l]->isDataLayer()){
                    pDataLayer = (DataLayer*) layers[l];
                    break;
                }
            }
        }
        if (pDataLayer==NULL) { std::cerr<<"No data layer for Testing."<<std::endl; FatalError(__LINE__);};


        std::cout<< "====================================================================================================================================="<<std::endl;

        int iter = 0;
        if(debug_mode){
            tic(); // 开始计时
        }
        while(pDataLayer->epoch == 0){ // 当全部的测试数据完成了一次时代，则循环结束
            resetLoss();
            forward();
            eval(false);

            // display
            if (display_iter >0 && iter % display_iter ==0) std::cout << "Iteration " << iter << "  ";
            for (int i=0;i<loss_layers.size();++i){
                if (loss_layers[i]->phase == phase || loss_layers[i]->phase == TrainingTesting){
                    if (display_iter >0 && iter % display_iter ==0) loss_layers[i]->display();
                    result[i] += loss_layers[i]->result;
                }
            }
            if (display_iter >0 && iter % display_iter ==0) std::cout << std::endl;

            ++iter;
        }

        for (int i=0;i<result.size();++i){
            result[i] /= iter;
        }

        std::cout << "Average over " << iter << " iterations  ";
        for (int i=0;i<result.size();++i){
            if (loss_layers[i]->phase == phase || loss_layers[i]->phase == TrainingTesting){
                std::cout << " eval = " << result[i];
                std::cout << "  ";
            }
        }
        std::cout << std::endl;

        if(debug_mode){
            std::cout<<"After Test "<<iter<<" Iterations"<<" Needs Time: ";
            toc(); // 计时结束
        }

        return result;
    };

};


}// namespace fecnn

#endif  // NET_H_