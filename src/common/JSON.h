#ifndef JSON_H_
#define JSON_H_

#include "common.h"

using namespace fecnn;

namespace fecnn {
//////////////////////////////////////////////////////////////////////////////////////////////////
// JSON parser
//////////////////////////////////////////////////////////////////////////////////////////////////

enum JSONType { JSON_String, JSON_Bool, JSON_Null, JSON_Number, JSON_Object, JSON_ObjectArray};

// plain object
class JSON{
public:
    JSONType type;
    std::vector<void*> array;
    std::map<std::string, JSON*> member;

    ~JSON(){
        for (int i=0;i<array.size();++i){
            if (array[i]!=NULL){
                switch(type){
                    case JSON_String:
                        delete ((std::string*)(array[i]));
                    break;
                    case JSON_Bool:
                        delete ((bool*)(array[i]));
                    break;
                    case JSON_Null:
                    break;
                    case JSON_Number:
                        delete ((ComputeT*)(array[i]));
                    break;
                    case JSON_Object:
                    break;
                    case JSON_ObjectArray:
                        delete ((JSON*)(array[i]));
                    break;
                }
            }
        }
        for (std::map<std::string, JSON*>::iterator it = member.begin(); it != member.end(); it++ ){
            if (it->second != NULL)
                delete it->second;
        }
    };

    std::string returnString(){
        if (type!=JSON_String) FatalError(__LINE__);
        return *((std::string*)(array[0]));
    };

    bool returnBool(){
        if (type!=JSON_Bool) FatalError(__LINE__);
        return *((bool*)(array[0]));
    };

    ComputeT returnReal(){
        if (type!=JSON_Number) FatalError(__LINE__);
        return *((ComputeT*)(array[0]));
    };

    std::vector<int> returnIntVector(){
        if (type!=JSON_Number) FatalError(__LINE__);
        std::vector<int> v(array.size());
        for (int i=0;i<array.size();++i){
            v[i] = (int)(*((ComputeT*)(array[i])));
        }
        return v;
    };

    std::vector<ComputeT> returnRealVector(){
        if (type!=JSON_Number) FatalError(__LINE__);
        std::vector<ComputeT> v(array.size());
        for (int i=0;i<array.size();++i){
            v[i] = (ComputeT)(*((ComputeT*)(array[i])));
        }
        return v;
    };

    std::vector<std::string> returnStringVector(){
        if (type!=JSON_String) FatalError(__LINE__);
        std::vector<std::string> v(array.size());
        for (int i=0;i<array.size();++i){
            v[i] = *((std::string*)(array[i]));
        }
        return v;
    };

    void setOrDie(std::string name, unsigned int &variable){
        if (this->member.find(name) == this->member.end()){
            FatalError(__LINE__);
        }
        else variable = (unsigned int)this->member[name]->returnReal();
    };

    void setOrDie(std::string name, bool &variable){
        if (this->member.find(name) == this->member.end()){
            FatalError(__LINE__);
        }
        else variable = this->member[name]->returnBool();
    };

    void setOrDie(std::string name, std::vector<ComputeT> &variable){
        if (this->member.find(name) == this->member.end())
            FatalError(__LINE__);
        else variable = this->member[name]->returnRealVector();
    };

    void set(std::string name, bool &variable, bool default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = this->member[name]->returnBool();
    };

    void set(std::string name, ComputeT &variable, ComputeT default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = (ComputeT)(this->member[name]->returnReal());
    };

    void setOrDie(std::string name, ComputeT &variable){
        if (this->member.find(name) == this->member.end())                              FatalError(__LINE__);
        else variable = (ComputeT)(this->member[name]->returnReal());
    };

    void set(std::string name, int &variable, int default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = (int)(this->member[name]->returnReal());
    };

    void set(std::string name, double &variable, double default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = (double)(this->member[name]->returnReal());
    };

    void set(std::string name, unsigned int &variable, unsigned int default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = (unsigned int)(this->member[name]->returnReal());
    };

    void setOrDie(std::string name, int &variable){
        if (this->member.find(name) == this->member.end())                              FatalError(__LINE__);
        else variable = (int)(this->member[name]->returnReal());
    };

    void set(std::string name, std::vector<int> &variable, std::vector<int> default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = this->member[name]->returnIntVector();
    };

    void set(std::string name, std::vector<ComputeT> &variable, std::vector<ComputeT> default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = this->member[name]->returnRealVector();
    };

    void set(std::string name, std::vector<std::string> &variable, std::vector<std::string> default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = this->member[name]->returnStringVector();
    };

    void setOrDie(std::string name, std::vector<std::string> &variable){
        if (this->member.find(name) == this->member.end())                              FatalError(__LINE__);
        else variable = this->member[name]->returnStringVector();
    };

    void setOrDie(std::string name, std::vector<int> &variable){
        if (this->member.find(name) == this->member.end())                              FatalError(__LINE__);
        else variable = this->member[name]->returnIntVector();
    };

    void set(std::string name, std::string &variable, std::string default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = this->member[name]->returnString();
    };

    void setOrDie(std::string name, std::string &variable){
        if (this->member.find(name) == this->member.end())                              FatalError(__LINE__);
        else variable = this->member[name]->returnString();
    };

    void set(std::string name, Filler &variable, Filler default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("Xavier"))             variable = Xavier;
        else if (0 == this->member[name]->returnString().compare("Gaussian"))           variable = Gaussian;
        else if (0 == this->member[name]->returnString().compare("Constant"))           variable = Constant;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, Pool &variable, Pool default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("Max"))                variable = Max;
        else if (0 == this->member[name]->returnString().compare("Average"))            variable = Average;
        else if (0 == this->member[name]->returnString().compare("Sum"))                variable = Sum;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, ActivateMode &variable, ActivateMode default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("ReLU"))               variable = ReLU;
        else if (0 == this->member[name]->returnString().compare("Sigmoid"))            variable = Sigmoid;
        else if (0 == this->member[name]->returnString().compare("Tanh"))               variable = Tanh;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void setOrDie(std::string name, LossObjective &variable){
        if (this->member.find(name) == this->member.end())                                                  FatalError(__LINE__);
        else if (0 == this->member[name]->returnString().compare("MultinomialLogistic_StableSoftmax"))      variable = MultinomialLogistic_StableSoftmax;
        else if (0 == this->member[name]->returnString().compare("MultinomialLogistic"))                    variable = MultinomialLogistic;
        else if (0 == this->member[name]->returnString().compare("SmoothL1"))                               variable = SmoothL1;
        else if (0 == this->member[name]->returnString().compare("Contrastive"))                            variable = Contrastive;
        else if (0 == this->member[name]->returnString().compare("EuclideanSSE"))                           variable = EuclideanSSE;
        else if (0 == this->member[name]->returnString().compare("HingeL1"))                                variable = HingeL1;
        else if (0 == this->member[name]->returnString().compare("HingeL2"))                                variable = HingeL2;
        else if (0 == this->member[name]->returnString().compare("SigmoidCrossEntropy"))                    variable = SigmoidCrossEntropy;
        else if (0 == this->member[name]->returnString().compare("Infogain"))                               variable = Infogain;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, Phase &variable, Phase default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("Training"))           variable = Training;
        else if (0 == this->member[name]->returnString().compare("Testing"))            variable = Testing;
        else if (0 == this->member[name]->returnString().compare("TrainingTesting"))    variable = TrainingTesting;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, LRPolicy &variable, LRPolicy default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("LR_fixed"))           variable = LR_fixed;
        else if (0 == this->member[name]->returnString().compare("LR_step"))            variable = LR_step;
        else if (0 == this->member[name]->returnString().compare("LR_exp"))             variable = LR_exp;
        else if (0 == this->member[name]->returnString().compare("LR_inv"))             variable = LR_inv;
        else if (0 == this->member[name]->returnString().compare("LR_multistep"))       variable = LR_multistep;
        else if (0 == this->member[name]->returnString().compare("LR_poly"))            variable = LR_poly;
        else if (0 == this->member[name]->returnString().compare("LR_sigmoid"))         variable = LR_sigmoid;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, SolverAlgorithm &variable, SolverAlgorithm default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("SGD"))                variable = SGD;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, Regularizer &variable, Regularizer default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("L2"))                 variable = L2;
        else if (0 == this->member[name]->returnString().compare("L1"))                 variable = L1;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void print(){
        switch(type){
            case JSON_String:
                if (array.size()>1) std::cout<<"[";
                for (int i=0;i<array.size();++i){
                    if (i>0) std::cout<< ",";
                    std::cout << "\"" << *((std::string*)(array[i])) << "\""  ;
                }
                if (array.size()>1) std::cout<<"]";
                std::cout<<std::endl;
            break;
            case JSON_Bool:
                if (array.size()>1) std::cout<<"[";
                for (int i=0;i<array.size();++i){
                    if (i>0) std::cout<< ",";
                    std::cout << ((*((bool*)(array[i])))? "true": "false");
                }
                if (array.size()>1) std::cout<<"]";
                std::cout<<std::endl;
            break;
            case JSON_Null:
                if (array.size()>1) std::cout<<"[";
                for (int i=0;i<array.size();++i){
                    if (i>0) std::cout<< ",";
                    std::cout << "null";
                }
                if (array.size()>1) std::cout<<"]";
                std::cout<<std::endl;
            break;
            case JSON_Number:
                if (array.size()>1) std::cout<<"[";
                for (int i=0;i<array.size();++i){
                    if (i>0) std::cout<< ",";
                    std::cout << *((ComputeT*)(array[i]));
                }
                if (array.size()>1) std::cout<<"]";
                std::cout<<std::endl;
            break;
            case JSON_Object:
                std::cout<<"{"<<std::endl;
                for (std::map<std::string, JSON*>::iterator it = member.begin(); it != member.end(); it++ ){
                    std::cout << "\t" << it->first << ": ";
                    it->second->print();
                }
                std::cout<<"}";
            break;
            case JSON_ObjectArray:
                std::cout<<"["<<std::endl;
                for (int i=0;i<array.size();++i){
                    JSON* p = (JSON*)(array[i]);
                    p->print();
                    if (i<array.size()-1) std::cout<<","<<std::endl;
                }
                std::cout<<"]"<<std::endl;
            break;
        }
    };

    void parseNumberOrTextArray(std::string input){
        while (input.size()>0){
            int e = input.find(",");
            if (e==std::string::npos){
                e = input.size();
            }
            std::string first = input.substr(0,e);
            if (first[0]=='\"'){
                type = JSON_String;
                std::string* p = new std::string(first.substr(1,first.size()-2));
                array.push_back((void*)p);
            }else if (first[0]=='t'){
                type = JSON_Bool;
                bool* p = new bool(true);
                array.push_back((void*)p);
            }else if (first[0]=='f'){
                type = JSON_Bool;
                bool* p = new bool(false);
                array.push_back((void*)p);
            }else if (first[0]=='n'){
                type = JSON_Null;
                void* p = NULL;
                array.push_back((void*)p);
            }else{
                type = JSON_Number;
                ComputeT* p = new ComputeT(stof(first));
                array.push_back((void*)p);
            }
            if(e+1<input.size())
                input=input.substr(e+1);
            else
                break;
        }
    };

    void parseObject(std::string input){
        type = JSON_Object;
        int b,m,e;
        JSON* p;
        b = input.find("{");
        e = input.find("}");
        input = input.substr(b+1,e-b-1);

        while (true){
            m= input.find(":");
            if (std::string::npos==m) break;

            std::string name = input.substr(0,m);
            name = name.substr(1,m-2);
            input = input.substr(m+1);
            if (input[0]=='\"'){
                e=input.find("\"",1);
                p = new JSON;
                p->parseNumberOrTextArray(input.substr(0,e+1));
                this->member[name] = p;

                if (e+2<input.size())
                    input = input.substr(e+2);
                else
                    break;
            }else if (input[0]=='['){
                // assume no nested array
                input = input.substr(1);
                e = input.find("]");
                p = new JSON;
                p->parseNumberOrTextArray(input.substr(0,e));
                this->member[name] = p;

                if (e+1<input.size())
                    input = input.substr(e+2);
                else
                    break;
            }else if (input[0]=='f' || input[0]=='t' || input[0]=='.' || input[0]=='-' || ('0'<=input[0] && input[0]<='9')){
                e=input.find(",");
                if (e==std::string::npos){
                    e = input.size();
                }
                p = new JSON;
                p->parseNumberOrTextArray(input.substr(0,e));
                this->member[name] = p;

                if (e+1<input.size())
                    input = input.substr(e+1);
                else
                    break;
            }else{
                FatalError(__LINE__);
            }
        }
    };
    void parseObjectArray(std::string input){
        type = JSON_ObjectArray;

        input = input.substr(1,input.size()-2);

        while (input.size()>0){
            int e = input.find("}")+1;
            if (e==std::string::npos){
                e = input.size();
            }
            std::string first = input.substr(0,e);
            JSON* pObj = new JSON;
            pObj->parseObject(first);
            array.push_back((void*)pObj);

            if(e+1<input.size())
                input=input.substr(e+1);
            else
                break;
        }
    };
};

// #表示字符串拼接
#define SetValue(obj,attribute,value) obj->set(#attribute,attribute,value);
#define SetOrDie(obj,attribute)       obj->setOrDie(#attribute,attribute);


void parseNetworkJSON(std::string filename, JSON* train_obj, JSON* test_obj, JSON* architecture_obj){
    std::ifstream t(filename);
    std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    str.erase(remove_if(str.begin(), str.end(), (int(*)(int))isspace), str.end()); // 去掉空格等
    std::string input = str;
    int b,e;

    b = input.find("\"train\"");
    std::string train_str = input.substr(b+7);
    b = train_str.find("{");
    e = train_str.find("}");
    train_str=train_str.substr(b,e-b+1);
    if (train_obj!=NULL) train_obj->parseObject(train_str);

    b = input.find("\"test\"");
    std::string test_str = input.substr(b+6);
    b = test_str.find("{");
    e = test_str.find("}");
    test_str=test_str.substr(b,e-b+1);
    if (test_obj!=NULL) test_obj->parseObject(test_str);

    b=input.find("\"layers\"");
    input = input.substr(b+9);
    e=input.find("}]");
    if (architecture_obj!=NULL) architecture_obj->parseObjectArray(input);

}

}// namespace fecnn

#endif  // JSON_H_