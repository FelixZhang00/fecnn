#ifndef SOLVER_H_
#define SOLVER_H_

#include "net.h"

using namespace fecnn;

namespace fecnn {
//////////////////////////////////////////////////////////////////////////////////////////////////
// Solver
//////////////////////////////////////////////////////////////////////////////////////////////////

class Solver{
    bool singleGPU;
public:
    Phase phase;
    std::vector<Net* > nets;

    std::string path;
    int iter;
    int current_step;
    int train_iter; // 每步train向前forword几次。

    std::vector<int> GPU;
    int GPU_solver;

    // machine learning paramters
    SolverAlgorithm solver;
    Regularizer regularizer;
    ComputeT momentum;
    ComputeT momentum2;
    ComputeT delta;
    ComputeT rms_decay;

    ComputeT weight_decay;

    ComputeT base_lr;
    LRPolicy lr_policy;
    ComputeT lr_gamma;
    ComputeT lr_power;
    int lr_stepsize;
    std::vector<int> stepvalue;

    int max_iter;           // 最大的迭代数（每次迭代一个forward，处理一个batch_size的数据）
    int snapshot_iter;      // 多少次迭代产生一个快照
    int display_iter;       // 多少次迭代在控制台中显示一次计算结果
    int test_iter;          // 每步test向前forword几次。
    int test_interval;      // 每多少次forward执行一次test
    bool debug_mode;


    Solver(std::string filename=std::string()){

        // 通过json文件构造神经网络
        JSON* train_obj = new JSON;
        JSON* architecture_obj = new JSON;
        parseNetworkJSON(filename, train_obj, NULL, architecture_obj);

        // 如果json文件中没有设置，就使用第三次参数作为默认值
        SetValue(train_obj, solver,         SGD)
        SetValue(train_obj, regularizer,    L2)
        SetValue(train_obj, momentum,       0.9)
        SetValue(train_obj, momentum2,      0.999)
        SetValue(train_obj, delta,          0.00000001)
        SetValue(train_obj, rms_decay,      0.98)

        SetValue(train_obj, weight_decay,   0.0005)
        SetValue(train_obj, base_lr,        0.01)
        SetValue(train_obj, lr_policy,      LR_inv)
        SetValue(train_obj, lr_gamma,       0.0001)
        SetValue(train_obj, lr_power,       0.75)
        SetValue(train_obj, lr_stepsize,    100000)
        SetValue(train_obj, train_iter,     1)
        SetValue(train_obj, max_iter,       10000)
        SetValue(train_obj, snapshot_iter,  5000)
        SetValue(train_obj, display_iter,   100)
        SetValue(train_obj, test_iter,      100)
        SetValue(train_obj, test_interval,  500)
        SetValue(train_obj, debug_mode,     false)
        SetValue(train_obj, GPU,            veci(1,0))
        SetOrDie(train_obj, path            )
        SetValue(train_obj, GPU_solver,     -1)

        if (GPU_solver==-1) GPU_solver=GPU[0];

        singleGPU = GPU.size()==1 && GPU_solver==GPU[0];
        
        int nGPUs = 0;
        checkCUDA(__FILE__,__LINE__, cudaGetDeviceCount(&nGPUs));
        if (nGPUs==0){
            std::cerr<<"There is no NVIDIA GPU available in this machine."<<std::endl;
            FatalError(__LINE__);
        }else if (nGPUs==1){
            std::cout<<"There is 1 NVIDIA GPU available in this machine."<<std::endl;
        }else{
            std::cout<<"There are "<< nGPUs<< " NVIDIA GPUs available in this machine."<<std::endl;
        }

        nets.resize(GPU.size());
        for (int n=0;n<nets.size();++n){
            nets[n] = new Net(architecture_obj, GPU[n]);
            nets[n]->debug_mode = debug_mode;
            nets[n]->train_iter = train_iter;
            nets[n]->test_iter  = test_iter;
        }

        // JOSN对象的数据已经全部分配到相应的对象中了，可以放心地删除指针
        delete train_obj; 
        delete architecture_obj;

    };

    // 参见 http://stackoverflow.com/questions/30033096/what-is-lr-policy-in-caffe
    ComputeT learning_rate(){
        ComputeT rate;
        switch(lr_policy){
            case LR_fixed:
                rate = base_lr;
                break;
            case LR_step:
                current_step = iter / lr_stepsize;
                rate = base_lr * pow(lr_gamma, current_step);
                break;
            case LR_exp:
                rate = base_lr * pow(lr_gamma, iter);
                break;
            case LR_inv:
                rate = base_lr * pow(ComputeT(1) + lr_gamma * iter, - lr_power);
                break;
            case LR_multistep:
                if (current_step < stepvalue.size() && iter >= stepvalue[current_step] ) {
                    current_step++;
                    std::cout << "MultiStep Status: Iteration " << iter << ", step = " << current_step << std::endl;
                }
                rate = base_lr * pow(lr_gamma, current_step);
                break;
            case LR_poly:
                rate = base_lr * pow(ComputeT(1) - (ComputeT(iter) / ComputeT(max_iter)), lr_power);
                break;
            case LR_sigmoid:
                rate = base_lr * (ComputeT(1) /  (ComputeT(1) + exp(-lr_gamma * (ComputeT(iter) - ComputeT(lr_stepsize)))));
                break;
        }
        return rate;
    };

    size_t Malloc(Phase phase_ = Training){
        phase = phase_;

        int nGPUs = 0;
        checkCUDA(__FILE__,__LINE__, cudaGetDeviceCount(&nGPUs));
        std::vector<size_t> memoryBytes(nGPUs,0);

        for (int n=0;n<nets.size();++n){
            memoryBytes[GPU[n]] += nets[n]->Malloc(phase);
        }


        if (phase == Training || phase == TrainingTesting){
            checkCUDA(__FILE__,__LINE__,cudaSetDevice(GPU_solver));

            for (int l=0; l<nets[0]->layers.size(); ++l){
                if (nets[0]->layers[l]->train_me){ // 选择有权重需要训练的层。第一层没有权重，需要过滤掉。
                    size_t weight_numel = nets[0]->layers[l]->weight_numel;
                    size_t   bias_numel = nets[0]->layers[l]->bias_numel;
                    if (weight_numel>0){
                        size_t weight_bytes = (1 + nets.size()) * weight_numel * sizeofStorageT;
                        checkCUDA(__FILE__,__LINE__, cudaMalloc(&(nets[0]->layers[l]->weight_histGPU), weight_bytes));
                        checkCUDA(__FILE__,__LINE__, cudaMemset(nets[0]->layers[l]->weight_histGPU, 0, weight_bytes));
                        memoryBytes[GPU_solver] += weight_bytes;
                        for (int n=0;n<nets.size();++n){
                            nets[n]->layers[l]->weight_histGPU = nets[0]->layers[l]->weight_histGPU;
                            nets[n]->layers[l]->weight_diffGPU = nets[0]->layers[l]->weight_histGPU + weight_numel * (n+1);
                        }
                    }
                    if (bias_numel>0){
                        size_t bias_bytes = (1 + nets.size()) * bias_numel * sizeofStorageT;
                        checkCUDA(__FILE__,__LINE__, cudaMalloc(&(nets[0]->layers[l]->bias_histGPU), bias_bytes));
                        checkCUDA(__FILE__,__LINE__, cudaMemset(nets[0]->layers[l]->bias_histGPU, 0, bias_bytes));
                        memoryBytes[GPU_solver] += bias_bytes;
                        for (int n=0;n<nets.size();++n){
                            nets[n]->layers[l]->bias_histGPU = nets[0]->layers[l]->bias_histGPU;
                            nets[n]->layers[l]->bias_diffGPU = nets[0]->layers[l]->bias_histGPU + bias_numel * (n+1);
                        }
                    }

                }
            }
        }

        std::cout<< "====================================================================================================================================="<<std::endl;
        for (int n=0;n<nGPUs;++n){
            if (memoryBytes[n]>0){
                std::cout<< "[Solver] GPU " << n << ": Total GPU memory: ";  memorySizePrint(memoryBytes[n]);    std::cout<<std::endl;
            }
        }

        size_t totalMemory = memoryBytes[0];
        for (int n=1;n<nGPUs;++n){
            totalMemory += memoryBytes[n];
        }

        std::cout<< "All GPUs: Total GPU memory: "; memorySizePrint(totalMemory);   std::cout<<std::endl;

        return totalMemory;
    };

    ~Solver(){
        checkCUDA(__FILE__,__LINE__,cudaSetDevice(GPU_solver));
        for (int l=0; l<nets[0]->layers.size(); ++l){
            if (nets[0]->layers[l]->train_me){
                if (nets[0]->layers[l]->weight_numel>0){
                    if (nets[0]->layers[l]->weight_histGPU!=NULL) checkCUDA(__FILE__,__LINE__, cudaFree(nets[0]->layers[l]->weight_histGPU));
                }
                if (nets[0]->layers[l]->bias_numel>0){
                    if (nets[0]->layers[l]->bias_histGPU!=NULL) checkCUDA(__FILE__,__LINE__, cudaFree(nets[0]->layers[l]->bias_histGPU));
                }
            }
        }
    };

    void randInit(){
        nets[0]->randInit();
    };

    void solve(ComputeT learning_rate){
        checkCUDA(__FILE__,__LINE__,cudaSetDevice(GPU_solver));

        for (int l=0; l<nets[0]->layers.size(); ++l){
            if (nets[0]->layers[l]->train_me){
                if (nets[0]->layers[l]->weight_numel>0){
                    update_solver(solver, regularizer,
                    	nets[0]->layers[l]->weight_numel, 
                    	nets.size(), 
                    	weight_decay * nets[0]->layers[l]->weight_decay_mult, // 各层的weight_decay_mult默认为1
                    	momentum, 
                    	learning_rate * nets[0]->layers[l]->weight_lr_mult,  // 各层的weight_lr_mult默认为1
                    	nets[0]->layers[l]->weight_dataGPU, 
                    	nets[0]->layers[l]->weight_histGPU);
                }
                if (nets[0]->layers[l]->bias_numel>0){
                    update_solver(solver, regularizer,
                    	nets[0]->layers[l]->bias_numel, 
                    	nets.size(), 
                    	weight_decay * nets[0]->layers[l]->bias_decay_mult, 
                    	momentum, 
                    	learning_rate * nets[0]->layers[l]->bias_lr_mult, 
                    	nets[0]->layers[l]->bias_dataGPU, 
                    	nets[0]->layers[l]->bias_histGPU);
                }
            }
        }
    };

    void loadWeights(std::string filename, bool diff=false){

        std::cout<< "====================================================================================================================================="<<std::endl;

        std::vector<Tensor<StorageT>*> weights = readTensors<StorageT>(filename);

        for (int i=0;i<nets.size();++i){
            nets[i]->loadWeights(weights, diff);
        }

        for (int i=0; i<weights.size();++i){
            delete weights[i];
        }
    };

    void saveWeights(std::string filename, bool diff=false){
        nets[0]->saveWeights(filename, diff);
    };

    void train(int iter_begin = 0){

        checkCUDA(__FILE__,__LINE__,cudaSetDevice(GPU_solver));
        
        phase = Training;
        current_step = 0;

        // 找到网络中的用于Train的dataLayer，为了获取当前的迭代周期（epoch）
        DataLayer* pDataLayer = NULL;
        for (int l=0; l<nets[0]->layers.size();++l){
            if (nets[0]->layers[l]->phase == phase){
                if (nets[0]->layers[l]->isDataLayer()){
                    pDataLayer = (DataLayer*) nets[0]->layers[l];
                    break;
                }
            }
        }
        if (pDataLayer==NULL) { std::cerr<<"No data layer for Train."<<std::endl; FatalError(__LINE__);};
        int epoch = pDataLayer->epoch;

        std::cout<< "==========================================================================================================================================="<<std::endl;
        std::cout<< "  Training:                                                                            Testing:                                            "<<std::endl;
        std::cout<< "==========================================================================================================================================="<<std::endl;

        if(debug_mode){
            tic(); // 开始计时
        }
        for (iter=iter_begin;iter<=max_iter;++iter){
            epoch = pDataLayer->epoch;

            if (iter % test_interval==0 && test_iter > 0){
                std::cout<< "                                                                                        ";
                std::cout << "Epoch " << epoch << "  ";
                std::cout << "Iteration " << iter;

                if (singleGPU){
                    nets[0]->phase = Testing;
                    nets[0]->stepTest(false);
                    nets[0]->phase = Training;
                }

                for (int l=0; l<nets[0]->loss_layers.size();++l){
                    if (nets[0]->loss_layers[l]->phase == phase || nets[0]->loss_layers[l]->phase == TrainingTesting){
                        for (int t=1;t<nets.size(); ++t){
                            nets[0]->loss_layers[l]->result += nets[t]->loss_layers[l]->result;
                            nets[0]->loss_layers[l]->loss += nets[t]->loss_layers[l]->loss;
                        }
                        nets[0]->loss_layers[l]->result /= nets.size();
                        nets[0]->loss_layers[l]->loss   /= nets.size();
                        nets[0]->loss_layers[l]->display();
                    }
                }
                std::cout << std::endl;
            }

            if (singleGPU){
                nets[0]->stepTrain(false); // 每次训练一个batch_size的数据
            }

            ComputeT lrate = learning_rate();
            solve(lrate);
            checkCUDA(__FILE__,__LINE__,cudaDeviceSynchronize());

            if (iter!=iter_begin && iter % snapshot_iter==0){
                saveWeights(path+"_snapshot_"+std::to_string(iter)+".fecnn",false);
            }
            if (iter % display_iter==0){
                std::cout << "Epoch " << epoch << "  ";
                std::cout << "Iteration " << iter << "  ";
                std::cout << "learning_rate = "<< lrate;


                if (singleGPU){
                    nets[0]->eval(false);
                }

                for (int l=0; l<nets[0]->loss_layers.size();++l){
                    if (nets[0]->loss_layers[l]->phase == phase || nets[0]->loss_layers[l]->phase == TrainingTesting){
                        nets[0]->loss_layers[l]->display();
                    }
                }
                std::cout << std::endl;
            }
        }
        if(debug_mode){
            std::cout<<"After Train "<<iter-1<<" Iterations "<<"Needs Time: ";
            toc(); // 计时结束
        }
    };
};

}// namespace fecnn

#endif  // SOLVER_H_