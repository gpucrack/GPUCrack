#!/bin/bash

# This script will install GPUCrack on your system.

printf "GPUCrack v0.1.4\n\nThis script will install GPUCrack on your system based on your choices.\n\n"
printf "Do you want to install the GPUCrack NTLM table generator? (CUDA needed) (y/n): "
read install_gpucrack_ntlm
if [ "$install_gpucrack_ntlm" = "y" ]; then
    printf "What is the length of the passwords you want to generate tables for? (1-8): "
    read pwd_length
    modified_file=$(sed "s/^#define PASSWORD_LENGTH.*/#define PASSWORD_LENGTH ${pwd_length}/" ./src/gpu/constants.cuh)
    echo "$modified_file" > ./src/gpu/constants.cuh
    printf "Installing the GPUCrack NTLM table generator...  "
fi
if [ "$install_gpucrack_ntlm" = "n" ]; then
    printf "Installing the GPUCrack attack program only...  "
    rm ./CMakeLists.txt
    printf "add_executable(online src/common/online.c)\ntarget_link_libraries(online PRIVATE m)\n" >> ./CMakeLists.txt
fi

# get output of command 'cmake .'
error_check=$(cmake .)
if [[ "$error_check" == *"The CUDA compiler identification is unknown"* || "$error_check" == *"Failed to detect a default CUDA architecture."* ]]; then
    printf "\n\nCUDA not found. Trying to locate NVCC...\n\n"
    nvcc_path=$(which nvcc)
    echo "nvcc_path: $nvcc_path"
    if [ -z "$nvcc_path" ]; then
        printf "NVCC not found. Please install CUDA and try again.\nIf you are sure CUDA is installed on your system, enter the full path to the nvcc executable :"
        read nvcc_path
    fi
    rm -rf ./CMakeFiles
    rm ./CMakeCache.txt
    wait
    error_check=$(cmake -DCMAKE_CUDA_COMPILER:PATH=$nvcc_path .)
fi

if [[ "$error_check" == *"-- Build files have been written"* ]]; then
    error_check=$(cmake --build .)
    if [[ "$error_check" == *"error"* ]]; then
        printf "\n\nBuild failed.\n\n"
        exit 1
    fi
    printf "Done.\n\nGPUCrack is now installed on your system.\n\nUse the following command to get started:\n\n"
    printf "  ./online -h\n ./generateTable\n"
    exit 0
fi

printf "\n\nError encountered while trying to install.\n\n"
exit 1