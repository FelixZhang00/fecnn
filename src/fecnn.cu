// Please choose a data type to compile
#include "fecnn.hpp"

using namespace fecnn;
using namespace std;

int main(int argc, char **argv){

    if (argc < 3 || argc >10){
        cout<<"Usage:"<<endl;
        cout<<argv[0]<<" train network.json"<<endl;
        cout<<"       example: "<<argv[0]<<" train examples/mnist/lenet_1_conv.json"<<endl;
        cout<<argv[0]<<" test network.json model.fecnn"<<endl;
        cout<<"       example: "<<argv[0]<<" test examples/mnist/lenet_1_conv.json examples/mnist/lenet_1_conv.fecnn"<<endl;
        return 0;
    }

    if(0==strcmp(argv[1], "train")){

        Solver solver(argv[2]);
        solver.Malloc(Training);
        solver.randInit();

        if (argc==3){       
            solver.train();
        }else FatalError(__LINE__);
        
        solver.saveWeights(solver.path + ".fecnn");
        
    }else if(0==strcmp(argv[1], "test")){

        Net net(argv[2]);
        net.Malloc(Testing);

        vector<string> models = getStringVector(argv[3]);
        for (int m=0;m<models.size();++m)   net.loadWeights(models[m]); // size=1


        if (argc==4){  // 输入./fecnn test examples/mnist/lenet.json examples/mnist/lenet.fecnn
            net.test();
        }else FatalError(__LINE__);

    }

    return 0;
}
