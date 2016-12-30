# fecnn
A fork from [Marvin](https://github.com/PrincetonVision/marvin), and simplify this project for learning Deep Learning.

This is a neural network framework using c++11 and CUDA.

## Dependences
Download [CUDA 7.5](https://developer.nvidia.com/cuda-downloads) and [cuDNN 5.1](https://developer.nvidia.com/
cudnn). 

Config you cuda path in compile.sh.


## Compilation

```shell
sh compile.sh
```

## Usage

Convolutional Neural Network on MNIST digits.

1. Prepare data: run examples/mnist/prepare_mnist.m in Matlab
2. Config your neural network model in /examples/mnist,just like this file lenet.json.
3. Visualize filters: config your json file path in /tools/viewer_matlab/plotNet.m,and run it in Matlab.
See you train result like this： 
![](https://raw.githubusercontent.com/FelixZhang00/fecnn/master/images/vis_model.jpg)

4. Train a model: run ./examples/mnist/train.sh in shell.
See you train result like this：
![](https://raw.githubusercontent.com/FelixZhang00/fecnn/master/images/train_result.jpg)

5. Visualize different model change the train result: use /tools/picture.py.
Before run it,you need config:
```
	file_list:the different models tained result output in shell,and save as files.
	dirname:the dir of file_list.
```
![](https://raw.githubusercontent.com/FelixZhang00/fecnn/master/images/diff_vis_result.jpg)


