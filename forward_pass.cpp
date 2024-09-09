#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include "cnpy.h"
#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

namespace fs = std::filesystem;

using namespace sycl;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

enum class ActivationType {
    None,
    ReLU,
    Sigmoid
};

void convolution(queue& q, float* input, float* weights, float* bias, float* output,
                 int num_input_channels, int num_output_channels,
                 int input_size, int kernel_size, int output_size, 
                 int stride, int padding,
                 ActivationType activation_type) {
    constexpr int REG_SIZE = 14; 
    
    q.submit([&](handler& h) {
        h.single_task([=]() {
            int total_output_elements = num_output_channels * output_size * output_size;
            int total_iterations = num_input_channels * kernel_size * kernel_size;

            #pragma unroll 24
            [[intel::ivdep]]
            for (int index = 0; index < total_output_elements; ++index) {
                int k = index / (output_size * output_size);
                int xy = index % (output_size * output_size);
                int x = xy / output_size;
                int y = xy % output_size;
                float acc = bias[k];

                float shift_reg[REG_SIZE] = {0};

                #pragma unroll 3
                for (int inner_index = 0; inner_index < total_iterations; ++inner_index) {
                    int c = inner_index / (kernel_size * kernel_size);
                    int i = (inner_index / kernel_size) % kernel_size;
                    int j = inner_index % kernel_size;

                    int xi = x * stride + i - padding;
                    int yj = y * stride + j - padding;
                    if (xi >= 0 && xi < input_size && yj >= 0 && yj < input_size) {
                        int weight_index = k * (num_input_channels * kernel_size * kernel_size) +
                                           c * (kernel_size * kernel_size) +
                                           i * kernel_size + j;

                        // Use shift register to accumulate results
                        shift_reg[REG_SIZE-1] = shift_reg[0] + input[(c * input_size + xi) * input_size + yj] * weights[weight_index];

                        // Shift every element of the shift register
                        #pragma unroll
                        for (int j = 0; j < REG_SIZE-1; ++j) {
                            shift_reg[j] = shift_reg[j + 1];
                        }
                    }
                }

                // Final accumulation to acc
                #pragma unroll
                for (int i = 0; i < REG_SIZE-1; ++i) {
                    acc += shift_reg[i];
                }

                // Apply activation function
                if (activation_type == ActivationType::ReLU) {
                    acc = std::max(0.0f, acc);
                } else if (activation_type == ActivationType::Sigmoid) {
                    acc = 1.0f / (1.0f + std::exp(-acc));
                }

                output[(k * output_size + x) * output_size + y] = acc;

            }
        });
    });

    q.wait();
}

void max_pooling(queue& q, float* input, float* output,
                 int input_channels, int input_size, int output_size,
                 int kernel_size, int stride) {
    q.submit([&](handler& h) {
        h.parallel_for(range<3>(input_channels, output_size, output_size), [=](id<3> idx) {
            int channel = idx[0];
            int out_row = idx[1];
            int out_col = idx[2];
            float max_value = std::numeric_limits<float>::lowest();

            // Calculate the starting point for the window
            int row_start = out_row * stride;
            int col_start = out_col * stride;

            // Traverse the window
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    int row = row_start + i;
                    int col = col_start + j;

                    // Check boundaries
                    if (row < input_size && col < input_size) {
                        float value = input[(channel * input_size + row) * input_size + col];
                        if (value > max_value) {
                            max_value = value;
                        }
                    }
                }
            }

            // Store the max value in the output
            output[(channel * output_size + out_row) * output_size + out_col] = max_value;
        });
    });

    q.wait();
}


int main() {

#if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#elif FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else
  auto selector = default_selector_v;
#endif

    queue q(selector, exception_handler);

    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
              
    // Setup for vgg layers
    const int num_input_channels = 3;
    const int num_output_0_channels = 64;
    const int input_size = 224;
    const int kernel_size = 3;
    const int output_layer_1_channels = 64;
    const int output_layer_2_channels = 128;
    const int output_layer_3_channels = 256;
    const int output_layer_4_5_channels = 512;
    const int output_add_channels = 128;
    const int stride = 1;
    const int padding = 1;

    // Setup for add-on layers
    const int kernel_size_add = 1;
    const int padding_add = 0;

    // Setup for max pooling
    const int pool_kernel_size = 2;
    const int pool_stride = 2;
    const int pooled_output_4_size = input_size/2;
    const int pooled_output_9_size = pooled_output_4_size/2;
    const int pooled_output_18_size = pooled_output_9_size/2;
    const int pooled_output_27_size = pooled_output_18_size/2;
    const int pooled_output_36_size = pooled_output_27_size/2;

    // Load data from .npy files
    cnpy::NpyArray weights_0 = cnpy::npy_load("./model_npy/features_features_0_weight.npy");
    float* weights_0_device = malloc_device<float>(num_output_0_channels * num_input_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_0_device, weights_0.data<float>(), weights_0.num_bytes()).wait();
    cnpy::NpyArray bias_0 = cnpy::npy_load("./model_npy/features_features_0_bias.npy");
    float* bias_0_device = malloc_device<float>(num_output_0_channels, q);
    q.memcpy(bias_0_device, bias_0.data<float>(), bias_0.num_bytes()).wait();
    cnpy::NpyArray weights_2 = cnpy::npy_load("./model_npy/features_features_2_weight.npy");
    float* weights_2_device = malloc_device<float>(output_layer_1_channels * num_output_0_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_2_device, weights_2.data<float>(), weights_2.num_bytes()).wait();
    cnpy::NpyArray bias_2 = cnpy::npy_load("./model_npy/features_features_2_bias.npy");   
    float* bias_2_device = malloc_device<float>(output_layer_1_channels, q);
    q.memcpy(bias_2_device, bias_2.data<float>(), bias_2.num_bytes()).wait();
    cnpy::NpyArray weights_5 = cnpy::npy_load("./model_npy/features_features_5_weight.npy");
    float* weights_5_device = malloc_device<float>(output_layer_2_channels * output_layer_1_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_5_device, weights_5.data<float>(), weights_5.num_bytes()).wait();
    cnpy::NpyArray bias_5 = cnpy::npy_load("./model_npy/features_features_5_bias.npy");   
    float* bias_5_device = malloc_device<float>(output_layer_2_channels, q);
    q.memcpy(bias_5_device, bias_5.data<float>(), bias_5.num_bytes()).wait();
    cnpy::NpyArray weights_7 = cnpy::npy_load("./model_npy/features_features_7_weight.npy");
    float* weights_7_device = malloc_device<float>(output_layer_2_channels * output_layer_2_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_7_device, weights_7.data<float>(), weights_7.num_bytes()).wait();
    cnpy::NpyArray bias_7 = cnpy::npy_load("./model_npy/features_features_7_bias.npy");  
    float* bias_7_device = malloc_device<float>(output_layer_2_channels, q);
    q.memcpy(bias_7_device, bias_7.data<float>(), bias_7.num_bytes()).wait();
    cnpy::NpyArray weights_10 = cnpy::npy_load("./model_npy/features_features_10_weight.npy");
    float* weights_10_device = malloc_device<float>(output_layer_3_channels * output_layer_2_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_10_device, weights_10.data<float>(), weights_10.num_bytes()).wait();
    cnpy::NpyArray bias_10 = cnpy::npy_load("./model_npy/features_features_10_bias.npy");    
    float* bias_10_device = malloc_device<float>(output_layer_3_channels, q);
    q.memcpy(bias_10_device, bias_10.data<float>(), bias_10.num_bytes()).wait();
    cnpy::NpyArray weights_12 = cnpy::npy_load("./model_npy/features_features_12_weight.npy");
    float* weights_12_device = malloc_device<float>(output_layer_3_channels * output_layer_3_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_12_device, weights_12.data<float>(), weights_12.num_bytes()).wait();
    cnpy::NpyArray bias_12 = cnpy::npy_load("./model_npy/features_features_12_bias.npy");    
    float* bias_12_device = malloc_device<float>(output_layer_3_channels, q);
    q.memcpy(bias_12_device, bias_12.data<float>(), bias_12.num_bytes()).wait();
    cnpy::NpyArray weights_14 = cnpy::npy_load("./model_npy/features_features_14_weight.npy");
    float* weights_14_device = malloc_device<float>(output_layer_3_channels * output_layer_3_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_14_device, weights_14.data<float>(), weights_14.num_bytes()).wait();
    cnpy::NpyArray bias_14 = cnpy::npy_load("./model_npy/features_features_14_bias.npy");    
    float* bias_14_device = malloc_device<float>(output_layer_3_channels, q);
    q.memcpy(bias_14_device, bias_14.data<float>(), bias_14.num_bytes()).wait();
    cnpy::NpyArray weights_16 = cnpy::npy_load("./model_npy/features_features_16_weight.npy");
    float* weights_16_device = malloc_device<float>(output_layer_3_channels * output_layer_3_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_16_device, weights_16.data<float>(), weights_16.num_bytes()).wait();
    cnpy::NpyArray bias_16 = cnpy::npy_load("./model_npy/features_features_16_bias.npy");    
    float* bias_16_device = malloc_device<float>(output_layer_3_channels, q);
    q.memcpy(bias_16_device, bias_16.data<float>(), bias_16.num_bytes()).wait();
    cnpy::NpyArray weights_19 = cnpy::npy_load("./model_npy/features_features_19_weight.npy");
    float* weights_19_device = malloc_device<float>(output_layer_4_5_channels * output_layer_3_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_19_device, weights_19.data<float>(), weights_19.num_bytes()).wait();
    cnpy::NpyArray bias_19 = cnpy::npy_load("./model_npy/features_features_19_bias.npy");    
    float* bias_19_device = malloc_device<float>(output_layer_4_5_channels, q);
    q.memcpy(bias_19_device, bias_19.data<float>(), bias_19.num_bytes()).wait();
    cnpy::NpyArray weights_21 = cnpy::npy_load("./model_npy/features_features_21_weight.npy");
    float* weights_21_device = malloc_device<float>(output_layer_4_5_channels * output_layer_4_5_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_21_device, weights_21.data<float>(), weights_21.num_bytes()).wait();
    cnpy::NpyArray bias_21 = cnpy::npy_load("./model_npy/features_features_21_bias.npy");    
    float* bias_21_device = malloc_device<float>(output_layer_4_5_channels, q);
    q.memcpy(bias_21_device, bias_21.data<float>(), bias_21.num_bytes()).wait();
    cnpy::NpyArray weights_23 = cnpy::npy_load("./model_npy/features_features_23_weight.npy");
    float* weights_23_device = malloc_device<float>(output_layer_4_5_channels * output_layer_4_5_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_23_device, weights_23.data<float>(), weights_23.num_bytes()).wait();
    cnpy::NpyArray bias_23 = cnpy::npy_load("./model_npy/features_features_23_bias.npy");    
    float* bias_23_device = malloc_device<float>(output_layer_4_5_channels, q);
    q.memcpy(bias_23_device, bias_23.data<float>(), bias_23.num_bytes()).wait();
    cnpy::NpyArray weights_25 = cnpy::npy_load("./model_npy/features_features_25_weight.npy");
    float* weights_25_device = malloc_device<float>(output_layer_4_5_channels * output_layer_4_5_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_25_device, weights_25.data<float>(), weights_25.num_bytes()).wait();
    cnpy::NpyArray bias_25 = cnpy::npy_load("./model_npy/features_features_25_bias.npy");    
    float* bias_25_device = malloc_device<float>(output_layer_4_5_channels, q);
    q.memcpy(bias_25_device, bias_25.data<float>(), bias_25.num_bytes()).wait();
    cnpy::NpyArray weights_28 = cnpy::npy_load("./model_npy/features_features_28_weight.npy");
    float* weights_28_device = malloc_device<float>(output_layer_4_5_channels * output_layer_4_5_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_28_device, weights_28.data<float>(), weights_28.num_bytes()).wait();
    cnpy::NpyArray bias_28 = cnpy::npy_load("./model_npy/features_features_28_bias.npy");    
    float* bias_28_device = malloc_device<float>(output_layer_4_5_channels, q);
    q.memcpy(bias_28_device, bias_28.data<float>(), bias_28.num_bytes()).wait();
    cnpy::NpyArray weights_30 = cnpy::npy_load("./model_npy/features_features_30_weight.npy");
    float* weights_30_device = malloc_device<float>(output_layer_4_5_channels * output_layer_4_5_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_30_device, weights_30.data<float>(), weights_30.num_bytes()).wait();
    cnpy::NpyArray bias_30 = cnpy::npy_load("./model_npy/features_features_30_bias.npy");    
    float* bias_30_device = malloc_device<float>(output_layer_4_5_channels, q);
    q.memcpy(bias_30_device, bias_30.data<float>(), bias_30.num_bytes()).wait();
    cnpy::NpyArray weights_32 = cnpy::npy_load("./model_npy/features_features_32_weight.npy");
    float* weights_32_device = malloc_device<float>(output_layer_4_5_channels * output_layer_4_5_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_32_device, weights_32.data<float>(), weights_32.num_bytes()).wait();
    cnpy::NpyArray bias_32 = cnpy::npy_load("./model_npy/features_features_32_bias.npy");    
    float* bias_32_device = malloc_device<float>(output_layer_4_5_channels, q);
    q.memcpy(bias_32_device, bias_32.data<float>(), bias_32.num_bytes()).wait();
    cnpy::NpyArray weights_34 = cnpy::npy_load("./model_npy/features_features_34_weight.npy");
    float* weights_34_device = malloc_device<float>(output_layer_4_5_channels * output_layer_4_5_channels * kernel_size * kernel_size, q);
    q.memcpy(weights_34_device, weights_34.data<float>(), weights_34.num_bytes()).wait();
    cnpy::NpyArray bias_34 = cnpy::npy_load("./model_npy/features_features_34_bias.npy");    
    float* bias_34_device = malloc_device<float>(output_layer_4_5_channels, q);
    q.memcpy(bias_34_device, bias_34.data<float>(), bias_34.num_bytes()).wait();
    cnpy::NpyArray weights_add_0 = cnpy::npy_load("./model_npy/add_on_layers_0_weight.npy");
    float* weights_add_0_device = malloc_device<float>(output_add_channels * output_layer_4_5_channels * kernel_size_add * kernel_size_add, q);
    q.memcpy(weights_add_0_device, weights_add_0.data<float>(), weights_add_0.num_bytes()).wait();
    cnpy::NpyArray bias_add_0 = cnpy::npy_load("./model_npy/add_on_layers_0_bias.npy");    
    float* bias_add_0_device = malloc_device<float>(output_add_channels, q);
    q.memcpy(bias_add_0_device, bias_add_0.data<float>(), bias_add_0.num_bytes()).wait();
    cnpy::NpyArray weights_add_2 = cnpy::npy_load("./model_npy/add_on_layers_2_weight.npy");
    float* weights_add_2_device = malloc_device<float>(output_add_channels * output_add_channels * kernel_size_add * kernel_size_add, q);
    q.memcpy(weights_add_2_device, weights_add_2.data<float>(), weights_add_2.num_bytes()).wait();
    cnpy::NpyArray bias_add_2 = cnpy::npy_load("./model_npy/add_on_layers_2_bias.npy");    
    float* bias_add_2_device = malloc_device<float>(output_add_channels, q);
    q.memcpy(bias_add_2_device, bias_add_2.data<float>(), bias_add_2.num_bytes()).wait();


    int image_set_size = 0;
    int image_folder_num = 0;
    // Iterate over each class directory within test_images_npy
    for (const auto& class_dir : fs::directory_iterator("./test_images_npy")) {
        // Ensure the entry is a directory
        if (fs::is_directory(class_dir)) {
            // Iterate over each npy file within the subdirectory
            image_folder_num += 1;
            std::cout << "Folder : " << image_folder_num << std::endl;
            std::cout << "Processing: " << class_dir << std::endl;
            for (const auto& file : fs::directory_iterator(class_dir.path())) {
                if (file.path().extension() == ".npy") {
                        std::string file_path = file.path().string();

                        cnpy::NpyArray img_test = cnpy::npy_load(file_path);
                        float* img_test_device = malloc_device<float>(num_input_channels * input_size * input_size, q);
                        q.memcpy(img_test_device, img_test.data<float>(), img_test.num_bytes()).wait();

                        float* conv_output_0_device = malloc_device<float>(num_output_0_channels * input_size * input_size, q);
                        convolution(q, img_test_device, weights_0_device, bias_0_device, conv_output_0_device, num_input_channels, num_output_0_channels, input_size, kernel_size, input_size, stride, padding, ActivationType::ReLU);

                        float* conv_output_2_device = malloc_device<float>(output_layer_1_channels * input_size * input_size, q);
                        convolution(q, conv_output_0_device, weights_2_device, bias_2_device, conv_output_2_device, num_output_0_channels, output_layer_1_channels, input_size, kernel_size, input_size, stride, padding, ActivationType::ReLU);

                        float* pool_output_4_device = malloc_device<float>(output_layer_1_channels * pooled_output_4_size * pooled_output_4_size, q);
                        max_pooling(q, conv_output_2_device, pool_output_4_device, output_layer_1_channels, input_size, pooled_output_4_size, pool_kernel_size, pool_stride);

                        //std::cout << "Finish layer 1";

                        float* conv_output_5_device = malloc_device<float>(output_layer_2_channels * pooled_output_4_size * pooled_output_4_size, q);
                        convolution(q, pool_output_4_device, weights_5_device, bias_5_device, conv_output_5_device, output_layer_1_channels, output_layer_2_channels, pooled_output_4_size, kernel_size, pooled_output_4_size, stride, padding, ActivationType::ReLU);

                        float* conv_output_7_device = malloc_device<float>(output_layer_2_channels * pooled_output_4_size * pooled_output_4_size, q);
                        convolution(q, conv_output_5_device, weights_7_device, bias_7_device, conv_output_7_device, output_layer_2_channels, output_layer_2_channels, pooled_output_4_size, kernel_size, pooled_output_4_size, stride, padding, ActivationType::ReLU);

                        float* pool_output_9_device = malloc_device<float>(output_layer_2_channels * pooled_output_9_size * pooled_output_9_size, q);
                        max_pooling(q, conv_output_7_device, pool_output_9_device, output_layer_2_channels, pooled_output_4_size, pooled_output_9_size, pool_kernel_size, pool_stride);

                        //std::cout << "Finish layer 2";

                        float* conv_output_10_device = malloc_device<float>(output_layer_3_channels * pooled_output_9_size * pooled_output_9_size, q);
                        convolution(q, pool_output_9_device, weights_10_device, bias_10_device, conv_output_10_device, output_layer_2_channels, output_layer_3_channels, pooled_output_9_size, kernel_size, pooled_output_9_size, stride, padding, ActivationType::ReLU);

                        float* conv_output_12_device = malloc_device<float>(output_layer_3_channels * pooled_output_9_size * pooled_output_9_size, q);
                        convolution(q, conv_output_10_device, weights_12_device, bias_12_device, conv_output_12_device, output_layer_3_channels, output_layer_3_channels, pooled_output_9_size, kernel_size, pooled_output_9_size, stride, padding, ActivationType::ReLU);

                        float* conv_output_14_device = malloc_device<float>(output_layer_3_channels * pooled_output_9_size * pooled_output_9_size, q);
                        convolution(q, conv_output_12_device, weights_14_device, bias_14_device, conv_output_14_device, output_layer_3_channels, output_layer_3_channels, pooled_output_9_size, kernel_size, pooled_output_9_size, stride, padding, ActivationType::ReLU);

                        float* conv_output_16_device = malloc_device<float>(output_layer_3_channels * pooled_output_9_size * pooled_output_9_size, q);
                        convolution(q, conv_output_14_device, weights_16_device, bias_16_device, conv_output_16_device, output_layer_3_channels, output_layer_3_channels, pooled_output_9_size, kernel_size, pooled_output_9_size, stride, padding, ActivationType::ReLU);

                        float* pool_output_18_device = malloc_device<float>(output_layer_3_channels * pooled_output_18_size * pooled_output_18_size, q);
                        max_pooling(q, conv_output_16_device, pool_output_18_device, output_layer_3_channels, pooled_output_9_size, pooled_output_18_size, pool_kernel_size, pool_stride);

                        //std::cout << "Finish layer 3";

                        float* conv_output_19_device = malloc_device<float>(output_layer_4_5_channels * pooled_output_18_size * pooled_output_18_size, q);
                        convolution(q, pool_output_18_device, weights_19_device, bias_19_device, conv_output_19_device, output_layer_3_channels, output_layer_4_5_channels, pooled_output_18_size, kernel_size, pooled_output_18_size, stride, padding, ActivationType::ReLU);

                        float* conv_output_21_device = malloc_device<float>(output_layer_4_5_channels * pooled_output_18_size * pooled_output_18_size, q);
                        convolution(q, conv_output_19_device, weights_21_device, bias_21_device, conv_output_21_device, output_layer_4_5_channels, output_layer_4_5_channels, pooled_output_18_size, kernel_size, pooled_output_18_size, stride, padding, ActivationType::ReLU);

                        float* conv_output_23_device = malloc_device<float>(output_layer_4_5_channels * pooled_output_18_size * pooled_output_18_size, q);
                        convolution(q, conv_output_21_device, weights_23_device, bias_23_device, conv_output_23_device, output_layer_4_5_channels, output_layer_4_5_channels, pooled_output_18_size, kernel_size, pooled_output_18_size, stride, padding, ActivationType::ReLU);

                        float* conv_output_25_device = malloc_device<float>(output_layer_4_5_channels * pooled_output_18_size * pooled_output_18_size, q);
                        convolution(q, conv_output_23_device, weights_25_device, bias_25_device, conv_output_25_device, output_layer_4_5_channels, output_layer_4_5_channels, pooled_output_18_size, kernel_size, pooled_output_18_size, stride, padding, ActivationType::ReLU);

                        float* pool_output_27_device = malloc_device<float>(output_layer_4_5_channels * pooled_output_27_size * pooled_output_27_size, q);
                        max_pooling(q, conv_output_25_device, pool_output_27_device, output_layer_4_5_channels, pooled_output_18_size, pooled_output_27_size, pool_kernel_size, pool_stride);

                        //std::cout << "Finish layer 4";

                        float* conv_output_28_device = malloc_device<float>(output_layer_4_5_channels * pooled_output_27_size * pooled_output_27_size, q);
                        convolution(q, pool_output_27_device, weights_28_device, bias_28_device, conv_output_28_device, output_layer_4_5_channels, output_layer_4_5_channels, pooled_output_27_size, kernel_size, pooled_output_27_size, stride, padding, ActivationType::ReLU);

                        float* conv_output_30_device = malloc_device<float>(output_layer_4_5_channels * pooled_output_27_size * pooled_output_27_size, q);
                        convolution(q, conv_output_28_device, weights_30_device, bias_30_device, conv_output_30_device, output_layer_4_5_channels, output_layer_4_5_channels, pooled_output_27_size, kernel_size, pooled_output_27_size, stride, padding, ActivationType::ReLU);

                        float* conv_output_32_device = malloc_device<float>(output_layer_4_5_channels * pooled_output_27_size * pooled_output_27_size, q);
                        convolution(q, conv_output_30_device, weights_32_device, bias_32_device, conv_output_32_device, output_layer_4_5_channels, output_layer_4_5_channels, pooled_output_27_size, kernel_size, pooled_output_27_size, stride, padding, ActivationType::ReLU);

                        float* conv_output_34_device = malloc_device<float>(output_layer_4_5_channels * pooled_output_27_size * pooled_output_27_size, q);
                        convolution(q, conv_output_32_device, weights_34_device, bias_34_device, conv_output_34_device, output_layer_4_5_channels, output_layer_4_5_channels, pooled_output_27_size, kernel_size, pooled_output_27_size, stride, padding, ActivationType::ReLU);

                        float* pool_output_36_device = malloc_device<float>(output_layer_4_5_channels * pooled_output_36_size * pooled_output_36_size, q);
                        max_pooling(q, conv_output_34_device, pool_output_36_device, output_layer_4_5_channels, pooled_output_27_size, pooled_output_36_size, pool_kernel_size, pool_stride);

                        //std::cout << "Finish layer 5";

                        float* conv_output_add_0_device = malloc_device<float>(output_add_channels * pooled_output_36_size * pooled_output_36_size, q);
                        convolution(q, pool_output_36_device, weights_add_0_device, bias_add_0_device, conv_output_add_0_device, output_layer_4_5_channels, output_add_channels, pooled_output_36_size, kernel_size_add, pooled_output_36_size, stride, padding_add, ActivationType::ReLU);

                        float* conv_output_add_2_device = malloc_device<float>(output_add_channels * pooled_output_36_size * pooled_output_36_size, q);
                        convolution(q, conv_output_add_0_device, weights_add_2_device, bias_add_2_device, conv_output_add_2_device, output_add_channels, output_add_channels, pooled_output_36_size, kernel_size_add, pooled_output_36_size, stride, padding_add, ActivationType::Sigmoid);

                        //std::cout << "Finish All";

                        std::vector<float> host_output_add_2(output_add_channels * pooled_output_36_size * pooled_output_36_size);
                        q.memcpy(host_output_add_2.data(), conv_output_add_2_device, sizeof(float) * output_add_channels * pooled_output_36_size * pooled_output_36_size).wait();
                        
                        // Saving the output for this image
                        // Change the directory from 'test_images_npy' to 'test_images_conv'
                        size_t npy_idx = file_path.rfind("test_images_npy");
                        if (npy_idx != std::string::npos) {
                                file_path.replace(npy_idx, std::string("test_images_npy").length(), "test_images_conv");
                        }
                        std::string output_path = file_path.substr(0, file_path.size() - 4) + "_output.npy"; 
                        cnpy::npy_save(output_path, host_output_add_2.data(), {output_add_channels, pooled_output_36_size, pooled_output_36_size}, "w");
                        image_set_size += 1;

                        // Free USM memory
                        sycl::free(img_test_device, q);
                        sycl::free(conv_output_0_device, q);
                        sycl::free(conv_output_2_device, q);
                        sycl::free(pool_output_4_device, q);
                        sycl::free(conv_output_5_device, q);
                        sycl::free(conv_output_7_device, q);
                        sycl::free(pool_output_9_device, q);
                        sycl::free(conv_output_10_device, q);
                        sycl::free(conv_output_12_device, q);
                        sycl::free(conv_output_14_device, q);
                        sycl::free(conv_output_16_device, q);
                        sycl::free(pool_output_18_device, q);
                        sycl::free(conv_output_19_device, q);
                        sycl::free(conv_output_21_device, q);
                        sycl::free(conv_output_23_device, q);
                        sycl::free(conv_output_25_device, q);
                        sycl::free(pool_output_27_device, q);
                        sycl::free(conv_output_28_device, q);
                        sycl::free(conv_output_30_device, q);
                        sycl::free(conv_output_32_device, q);
                        sycl::free(conv_output_34_device, q);
                        sycl::free(pool_output_36_device, q);
                        sycl::free(conv_output_add_0_device, q);
                        sycl::free(conv_output_add_2_device, q);
                }
            }
        }
    }

    sycl::free(weights_0_device, q);
    sycl::free(bias_0_device, q);
    sycl::free(weights_2_device, q);
    sycl::free(bias_2_device, q);
    sycl::free(weights_5_device, q);
    sycl::free(bias_5_device, q);
    sycl::free(weights_7_device, q);
    sycl::free(bias_7_device, q);
    sycl::free(weights_10_device, q);
    sycl::free(bias_10_device, q);
    sycl::free(weights_12_device, q);
    sycl::free(bias_12_device, q);
    sycl::free(weights_14_device, q);
    sycl::free(bias_14_device, q);
    sycl::free(weights_16_device, q);
    sycl::free(bias_16_device, q);
    sycl::free(weights_19_device, q);
    sycl::free(bias_19_device, q);
    sycl::free(weights_21_device, q);
    sycl::free(bias_21_device, q);
    sycl::free(weights_23_device, q);
    sycl::free(bias_23_device, q);
    sycl::free(weights_25_device, q);
    sycl::free(bias_25_device, q);
    sycl::free(weights_28_device, q);
    sycl::free(bias_28_device, q);
    sycl::free(weights_30_device, q);
    sycl::free(bias_30_device, q);
    sycl::free(weights_32_device, q);
    sycl::free(bias_32_device, q);
    sycl::free(weights_34_device, q);
    sycl::free(bias_34_device, q);
    sycl::free(weights_add_0_device, q);
    sycl::free(bias_add_0_device, q);
    sycl::free(weights_add_2_device, q);
    sycl::free(bias_add_2_device, q);
  
    std::cout << "Finished processing all files."<< std::endl;
    std::cout << "Dataset Size: " << image_set_size << std::endl;

    return 0;
}