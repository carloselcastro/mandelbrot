OUTPUT=mandelbrot

IMAGE=.ppm

CC=gcc
MCC=mpicc -Wall
CC_OPT=-std=c11 -Wall 
CUDA=nvcc

CC_OMP=-fopenmp 
CC_PTH=-pthread 
CUDA_FLAGS=-lmpi -lmpi_cxx

.PHONY: all
all: $(OUTPUT)_omp $(OUTPUT)_pth $(OUTPUT)_seq $(OUTPUT)_cuda $(OUTPUT)_mpi $(OUTPUT)_mpi_omp $(OUTPUT)_cuda_mpi

$(OUTPUT)_omp: $(OUTPUT)_omp.c
	$(CC) -o $(OUTPUT)_omp $(CC_OPT) $(CC_OMP) $(OUTPUT)_omp.c -lm

$(OUTPUT)_pth: $(OUTPUT)_pth.c
	$(CC) -o $(OUTPUT)_pth $(CC_OPT) $(CC_PTH) $(OUTPUT)_pth.c -lm

$(OUTPUT)_seq: $(OUTPUT)_seq.c
	$(CC) -o $(OUTPUT)_seq $(CC_OPT) $(OUTPUT)_seq.c

$(OUTPUT)_cuda: $(OUTPUT)_cuda.cu
	$(CUDA) -o $(OUTPUT)_cuda $(OUTPUT)_cuda.cu

$(OUTPUT)_mpi: $(OUTPUT)_mpi.c
	$(MCC) -o $(OUTPUT)_mpi $(OUTPUT)_mpi.c

$(OUTPUT)_mpi_omp: $(OUTPUT)_mpi_omp.c
	$(MCC) -o $(OUTPUT)_mpi_omp $(CC_OPT) $(CC_OMP) $(OUTPUT)_mpi_omp.c -lm

$(OUTPUT)_cuda_mpi: $(OUTPUT)_cuda_mpi.cu
	$(CUDA) -o $(OUTPUT)_cuda_mpi $(CUDA_FLAGS) $(OUTPUT)_cuda_mpi.cu

.PHONY: clean
clean:
	rm $(OUTPUT)_omp $(OUTPUT)_pth $(OUTPUT)_seq $(OUTPUT)_cuda $(OUTPUT)_mpi $(OUTPUT)_mpi_omp *$(IMAGE) $(OUTPUT)_cuda_mpi

