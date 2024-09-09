# Accelerating Interpretable Deep Neural Networks using FPGAs
## Acknowledgments
The code in the `ProtoPNet` folder is cloned from https://github.com/cfchen-duke/ProtoPNet 
- Some modifications have been made in `setting.py` and `img_aug.py` within the `ProtoPNet` folder.
- The code in the `ProtoPNet` folder is only used for training the model.
- The files `cpu_execution.py` and `check_fpga_acc.py` are adapted from the code within the `ProtoPNet` folder.


---

## Training the Model
- Follow the instructions provided in the ProtoPNet repository for training the model.
- Ensure to use the dataset in accordance with the ProtoPNet repository.
- After training the model:
     - Place the trained model `.pth` file and its corresponding prototype files in the `saved_models` folder.
     - Place the cropped test images in the `datasets` folder.
---

## Measure Inference Duration on CPU
- Run `cpu_execution.py`.
---
## For FPGA Implementation

### 1. **Preparation**:
- **Save Model and Test Images in `.npy` Format**:
  - Run `save_model_npy.py` to save the model’s weights and biases in `.npy` format.
  - Run `save_test_img_npy.py` to save the test images in `.npy` format.

- **Install Required Libraries**:
  - Install the `cnpy` library by cloning it from https://github.com/rogersce/cnpy

- **Setup Intel oneAPI and Quartus Prime**:
  - Ensure that Intel's oneAPI environment and Quartus Prime are installed and set up.
  - Set environment variables:
    ```bash
    export QSYS_ROOTDIR="/intel/quartus/qsys/bin"
    export QUARTUS_ROOTDIR="/intel/quartus/quartus"
    ```
### 2. **Compiling SYCL C++ Code for FPGA**:
The SYCL C++ code for FPGA-optimized convolutional layers is located in `forward_pass.cpp`. Below are the instructions for compiling this code in various configurations:

#### a. **FPGA Emulation**:
   - Use the following command to compile for FPGA emulation:
     ```bash
     icpx -fsycl -fintelfpga -DFPGA_EMULATOR -I/home/cnpy/ forward_pass.cpp -o forward_pass.fpga_emu -L/home/cnpy/build -lcnpy -lz
     ```

#### b. **FPGA Optimization Report**:
   - Use the following command to generate an optimization report:
     ```bash
     icpx -fsycl -fintelfpga -Xshardware -Xstarget=Arria10 -fsycl-link=early -I/home/cnpy/ forward_pass.cpp -o forward_pass.a -L/home/cnpy/build -lcnpy -lz
     ```

#### c. **FPGA Bitstream**:
   - Use the following command to generate the FPGA bitstream:
     ```bash
     icpx -fsycl -fintelfpga -Xshardware -Xstarget=Arria10 -I/home/cnpy/ forward_pass.cpp -o forward_pass.fpga -L/home/cnpy/build -lcnpy -lz
     ```

   - **Note**:
     - `/home/cnpy` refers to the location of the `cnpy` folder.
     - `Xstarget` specifies the targeted FPGA board.

### 3. **Check FPGA Implementation Accuracy**:
   - Compile the `forward_pass.cpp` for FPGA emulation.
   - Set the library path and run the emulation:
       ```bash
       export LD_LIBRARY_PATH=/home/cnpy/build:$LD_LIBRARY_PATH
       ./forward_pass.fpga_emu
       ```
   - Run `check_fpga_acc.py` to measure the accuracy.

### 4. **Estimate FPGA Execution Time**:
   - Compile the `forward_pass.cpp` for  optimization report or bitstream.
   - Use the information from the optimization report to fill in the variables in `estimate_fpga_runtime.py`.
   - Run `estimate_fpga_runtime.py` to estimate the execution time.
    
---
