#ifndef DATA_LAYER_H_
#define DATA_LAYER_H_

#include "Layer.h"

using namespace fecnn;

namespace fecnn {

class DataLayer : public Layer {
public:
    // parameters:
    bool random;
    int counter;
    int epoch;
    bool isDataLayer(){ return true; };
    DataLayer(): counter(0), epoch(0), random(false){};
    DataLayer(std::string name_): Layer(name_), counter(0), epoch(0), random(false){};
    virtual int numofitems() = 0;
    virtual void shuffle() = 0;
};

}// namespace fecnn

#endif  // DATA_LAYER_H_