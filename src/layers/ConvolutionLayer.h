#ifndef CONVOLUTION_LAYER_H_
#define CONVOLUTION_LAYER_H_

#include "Layer.h"

using namespace fecnn;

namespace fecnn {

void fecnn_gpu_gemm(cublasHandle_t cublasHandle,const cublasOperation_t TransA,
    const cublasOperation_t TransB, const int M, const int N, const int K,
    const StorageT alpha, const StorageT* A, const StorageT* B, const StorageT beta,
    StorageT* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CUBLAS_OP_N) ? K : M;
  int ldb = (TransB == CUBLAS_OP_N) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
  checkCUBLAS(__FILE__,__LINE__,
  	cublasSgemm(cublasHandle, cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

// @fixme 只适合计算float型
void fecnn_gpu_gemv(cublasHandle_t cublasHandle,const cublasOperation_t TransA, const int M,
    const int N, const StorageT alpha, const StorageT* A, const StorageT* x,
    const StorageT beta, StorageT* y) {
  cublasOperation_t cuTransA =
      (TransA == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N;
  checkCUBLAS(__FILE__,__LINE__,
        cublasSgemv(cublasHandle, cuTransA, N, M, &alpha,
         A, N, x, 1, &beta, y, 1) );
}

__global__ void im2col_gpu_kernel(const int n, const StorageT* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    StorageT* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;

    StorageT* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

    const StorageT* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void im2col_gpu(const StorageT* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    StorageT* data_col) {

  // std::cout<<"channels="<<channels<<" height="<<height<<" width="<<width<<std::endl; // conv1:[1,28,28] conv2:[20,12,12]
  // std::cout<<"dilation_h="<<dilation_h<<" dilation_w="<<dilation_w<<std::endl; // =1,1

  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;

  // std::cout<<"height_col="<<height_col<<" width_col="<<width_col<<" channels="<<channels<<std::endl; // conv1:[24,24,1] ; conv2:[8,8,20]

  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<<<FECNN_GET_BLOCKS(num_kernels),
                             FECNN_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
}

__global__ void col2im_gpu_kernel(const int n, const StorageT* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    StorageT* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    StorageT val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

void col2im_gpu(const StorageT* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    StorageT* data_im) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<<<FECNN_GET_BLOCKS(num_kernels),
                             FECNN_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      height_col, width_col, data_im);
}


class ConvolutionLayer : public Layer {
public:
    int num_output;
    std::vector<int> window;
    std::vector<int> stride;
    std::vector<int> padding;

    // forward中用到的M,N,K
    int conv_out_channels_;
    int conv_out_spatial_dim_;
    int kernel_dim_;

    int num_; // =batchsize

    // 除去batchsize外的count
    int in_dim_;
    int out_dim_;

    int conv_in_channels_;
    std::vector<int> conv_input_shape_;
    std::vector<int> kernel_shape_;

    Blob<StorageT> col_buffer_;

    StorageT* bias_multGPU;
    int out_spatial_dim_; // conv1:[24*24];conv2:[8*8]

    void init(){
        weight_dim.push_back(num_output);
        weight_dim.push_back(0);  // need the channel size from the input
        weight_dim.insert( weight_dim.end(), window.begin(), window.end() );

        bias_dim.resize(weight_dim.size(), 1);
        bias_dim[1] = num_output;
        // weight_dim = [20,1,5,5] [50,20,5,5]
        // bias_dim = [1,20,1,1]
    };

    ConvolutionLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,               TrainingTesting)
        SetValue(json, train_me,            true)
        SetOrDie(json, num_output           )
        SetOrDie(json, window               )
        SetValue(json, weight_lr_mult,      1.0)
        SetValue(json, weight_filler,       Xavier)
        SetValue(json, weight_filler_param, 0.0)
        SetValue(json, bias_lr_mult,        2.0)
        SetValue(json, bias_filler,         Constant)
        SetValue(json, bias_filler_param,   0.0)
        SetValue(json, weight_decay_mult,   1.0)
        SetValue(json, bias_decay_mult,     1.0)

        std::vector<int> ones  = std::vector<int>(window.size(),1);
        std::vector<int> zeros = std::vector<int>(window.size(),0);
        SetValue(json, padding,             zeros)
        SetValue(json, stride,              ones)

        init();
    };

    ConvolutionLayer(std::string name_,
                    int num_output_,
                    std::vector<int> window_,
                    std::vector<int> padding_, std::vector<int> stride_,
                    ComputeT weight_lr_mult_,   Filler weight_filler_, ComputeT weight_filler_param_,
                    ComputeT bias_lr_mult_,     Filler bias_filler_,   ComputeT  bias_filler_param_):
                    Layer(name_),
                    num_output(num_output_), window(window_), stride(stride_), padding(padding_){

        weight_lr_mult = weight_lr_mult_;
        weight_filler = weight_filler_;
        weight_filler_param = weight_filler_param_;

        bias_lr_mult = bias_lr_mult_;
        bias_filler = bias_filler_;
        bias_filler_param = bias_filler_param_;

        init();
    };
    size_t Malloc(Phase phase_){

        size_t memoryBytes = 0;
        train_me = train_me && phase_ != Testing;

        std::cout<< (train_me? "* " : "  ");
        std::cout<<name;

        if (in.size()==0) { std::cout<<std::endl<<"ConvolutionLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"ConvolutionLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        this->num_ = in[0]->dim[0];
        weight_dim[1] = in[0]->dim[1];
        this->in_dim_ = in[0]->sizeofitem();
        this->conv_in_channels_ = in[0]->dim[1];
        this->conv_input_shape_.resize(2);
        this->conv_input_shape_[0] = in[0]->dim[2];
        this->conv_input_shape_[1] = in[0]->dim[3];
        this->kernel_shape_.resize(2);
        this->kernel_shape_[0] = weight_dim[2]; 
        this->kernel_shape_[1] = weight_dim[3]; 
        // std::cout<<"kernel_shape_=";veciPrint(kernel_shape_);
        // std::cout<<"conv_input_shape_=";veciPrint(conv_input_shape_);
        // std::cout<<" conv_in_channels_="<<conv_in_channels_<<std::endl;
        // std::cout<<"in_dim_="<<in_dim_<<std::endl;

        weight_numel = numel(weight_dim);
        bias_numel   = numel(bias_dim);

        if (weight_numel>0){
            std::cout<<" weight"; veciPrint(weight_dim);
            checkCUDA(__FILE__,__LINE__, cudaMalloc( &weight_dataGPU, weight_numel * sizeofStorageT) );
            memoryBytes += weight_numel * sizeofStorageT;
        }
        if (bias_numel>0){
            std::cout<<" bias"; veciPrint(bias_dim);
            checkCUDA(__FILE__,__LINE__, cudaMalloc( &bias_dataGPU, bias_numel * sizeofStorageT) );
            memoryBytes += bias_numel * sizeofStorageT;
        }
        std::cout<<std::endl;


        for (int i=0;i<out.size();++i){
            out[i]->need_diff = train_me || in[i]->need_diff; // if one of them need the grad

            std::vector<int> dimOut;
            dimOut.resize(in[i]->dim.size());

            dimOut[0] = in[0]->dim[0];
            dimOut[1] = num_output;
            int w = (in[0]->dim[2] - window[0] + 2*padding[0])/stride[0]+1;
            int h = (in[0]->dim[3] - window[1] + 2*padding[1])/stride[1]+1;
            // std::cout<<" w="<<w<<" h="<<h<<std::endl;
            dimOut[2] = w;
            dimOut[3] = h;

            memoryBytes += out[i]->Malloc(dimOut);
        }
        this->out_dim_ = out[0]->sizeofitem();
        this->kernel_dim_ = weight_dim[1]*weight_dim[2]*weight_dim[3];
        this->conv_out_channels_ = num_output;
        this->conv_out_spatial_dim_ = out[0]->dim[2]*out[0]->dim[3];
        this->out_spatial_dim_ = conv_out_spatial_dim_;
        // std::cout<<"out_dim_="<<out_dim_<<" kernel_dim_="<<kernel_dim_<<" conv_out_channels_="<<conv_out_channels_
        // <<" conv_out_spatial_dim_="<<conv_out_spatial_dim_<<std::endl;	

        if(bias_numel>0){
        	checkCUDA(__FILE__,__LINE__, cudaMalloc(&bias_multGPU, out_spatial_dim_ * sizeofStorageT) );
        	Kernel_set_value<<<FECNN_GET_BLOCKS(out_spatial_dim_),FECNN_CUDA_NUM_THREADS>>>(out_spatial_dim_, bias_multGPU, CPUCompute2StorageT(1));
        	memoryBytes += out_spatial_dim_ * sizeofStorageT;
        }

        
        std::vector<int> shape = {kernel_dim_,out[0]->dim[2],out[0]->dim[3]};
        // std::cout<<" col_buffer_.shape"; veciPrint(shape);
        // @TODO 需要加上内存统计	
        col_buffer_.Reshape(shape);

        return memoryBytes;
    };

    void forward(Phase phase_){

        for (int i=0;i<in.size();++i){
        		const StorageT* in_data = in[i]->dataGPU;
        		StorageT* out_data = out[i]->dataGPU;
        		const StorageT* weight = weight_dataGPU;
        		const StorageT* bias =bias_dataGPU;

        		for (int n = 0; n < num_; ++n){// num_ = batchsize
        			this->forward_gpu_gemm(in_data + n * this->in_dim_, weight,
          					out_data + n * this->out_dim_);

        			this->forward_gpu_bias(out_data + n * this->out_dim_, bias);
        		}
        }
    };

    void backward(Phase phase_){
    		const StorageT* weight = weight_dataGPU;
    		StorageT* weight_diff = weight_diffGPU;

    		for (int i=0;i<in.size();++i){
    			if(train_me){
    				const StorageT* out_diff = out[i]->diffGPU;

    				// Bias gradient, if necessary.
    				if (bias_numel>0) {
    				  StorageT* bias_diff = bias_diffGPU;
    				  for (int n = 0; n < this->num_; ++n) {
    				    this->backward_gpu_bias(bias_diff, out_diff + n * this->out_dim_);
    				  }
    				}

    				const StorageT* in_data = in[i]->dataGPU;
    				StorageT* in_diff = in[i]->diffGPU;
    				for (int n = 0; n < this->num_; ++n) {
    				    // gradient w.r.t. weight. Note that we will accumulate diffs.
                        this->weight_gpu_gemm(in_data + n * this->in_dim_,
                            out_diff + n * this->out_dim_, weight_diff);

    				    // gradient w.r.t. bottom data, if necessary.
    				    if (in[i]->need_diff) {
    				        this->backward_gpu_gemm(out_diff + n * this->out_dim_, weight,
    				            in_diff + n * this->in_dim_);
    				  }
    				}
    			}
    	}

    };
    ~ConvolutionLayer(){
    };

    // 将图像卷积运算的input转化为矩阵形式
    void conv_im2col_gpu(const StorageT* data, StorageT* col_buff){
    	im2col_gpu(data, conv_in_channels_,
    	    conv_input_shape_[0], conv_input_shape_[1],
    	    kernel_shape_[0], kernel_shape_[1],
    	    padding[0], padding[1],
    	    stride[0], stride[1],
    	    1, 1, col_buff);
    }

    void conv_col2im_gpu(const StorageT* col_buff, StorageT* data) {
        col2im_gpu(col_buff, conv_in_channels_,
            conv_input_shape_[0], conv_input_shape_[1],
            kernel_shape_[0], kernel_shape_[1],
            padding[0], padding[1],
            stride[0], stride[1],
            1, 1, data);
    }

    void forward_gpu_gemm(const StorageT* input,
        const StorageT* weights, StorageT* output){

    	const StorageT* col_buff = input;
    	conv_im2col_gpu(input, col_buffer_.dataGPU);
    	col_buff = col_buffer_.dataGPU;

    	fecnn_gpu_gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
    			conv_out_channels_,conv_out_spatial_dim_,kernel_dim_,
    			(StorageT)1.,weights,
    			col_buff,
    			(StorageT)0.,output);
    } 

    void forward_gpu_bias(StorageT* output,const StorageT* bias){

    	fecnn_gpu_gemm(cublasHandle,CUBLAS_OP_N, CUBLAS_OP_N, 
    	    num_output,out_spatial_dim_, 1, 
    	    (StorageT)1., bias, bias_multGPU,
    	    (StorageT)1., output);
    }

    // BP
    void backward_gpu_bias(StorageT* bias,const StorageT* input) {
        fecnn_gpu_gemv(cublasHandle,CUBLAS_OP_N, 
            num_output, out_spatial_dim_, 1.,
            input, bias_multGPU, 1., bias);
    }

    void weight_gpu_gemm(const StorageT* input,
        const StorageT* output, StorageT* weights) {

        const StorageT* col_buff = input;
        conv_im2col_gpu(input, col_buffer_.dataGPU);
        col_buff = col_buffer_.dataGPU;
        
        fecnn_gpu_gemm(cublasHandle,CUBLAS_OP_N, CUBLAS_OP_T, 
            conv_out_channels_,kernel_dim_, conv_out_spatial_dim_,
            (StorageT)1., output, 
            col_buff,
            (StorageT)1., weights);
        
    }

    void backward_gpu_gemm(const StorageT* output,
        const StorageT* weights, StorageT* input) {

        StorageT* col_buff = col_buffer_.dataGPU;

        fecnn_gpu_gemm(cublasHandle,CUBLAS_OP_T, CUBLAS_OP_N, 
            kernel_dim_,conv_out_spatial_dim_, conv_out_channels_,
            (StorageT)1., weights, 
            output,
            (StorageT)0., col_buff);

        // conv_col2im_gpu(col_buff, input);
    }

};


}// namespace fecnn

#endif  // CONVOLUTION_LAYER_H_