################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/BlackScholes_gold.cpp 

CU_SRCS += \
../src/BlackScholes.cu 

CU_DEPS += \
./src/BlackScholes.d 

OBJS += \
./src/BlackScholes.o \
./src/BlackScholes_gold.o 

CPP_DEPS += \
./src/BlackScholes_gold.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I"/usr/local/cuda-7.5/samples/4_Finance" -I"/usr/local/cuda-7.5/samples/common/inc" -I"/home/piotr/cuda-workspace/BSprofile_dummy" -G -g -O0 -gencode arch=compute_20,code=sm_20  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I"/usr/local/cuda-7.5/samples/4_Finance" -I"/usr/local/cuda-7.5/samples/common/inc" -I"/home/piotr/cuda-workspace/BSprofile_dummy" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I"/usr/local/cuda-7.5/samples/4_Finance" -I"/usr/local/cuda-7.5/samples/common/inc" -I"/home/piotr/cuda-workspace/BSprofile_dummy" -G -g -O0 -gencode arch=compute_20,code=sm_20  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I"/usr/local/cuda-7.5/samples/4_Finance" -I"/usr/local/cuda-7.5/samples/common/inc" -I"/home/piotr/cuda-workspace/BSprofile_dummy" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


