# Mandelbrot Set

This repository contains some of programs-exercises of class MAC5742 - Introduction to Concurrent, Parallel and Distributed Programming (2020) - Institute of Mathematics and Statistics, São Paulo University (IME-USP).

## Members
Carlos Eduardo Leal de Castro - o/

[Felipe de Lima Peressim](https://github.com/feperessim)

[José Luiz Maciel Pimenta](https://github.com/JoseLuiz432)

[Luis Ricardo Manrique](https://github.com/lllmanriquelll)

[Rafael Fernandes Alencar](https://github.com/rafalencar1997)

### Goal
The main goal of this work was to generate the Mandelbrot Set using different approaches: Sequential, Pthreads, OpenMP, OpenMPI and Cuda.

The Mandelbrot Set is the set of all complex points <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;$\mathbb{C}$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;$\mathbb{C}$" title="$\mathbb{C}$" /></a> such that the sequence <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\{z_n\}_{n&space;\in&space;(\mathbb{N}&space;\cup&space;\{0\})}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\{z_n\}_{n&space;\in&space;(\mathbb{N}&space;\cup&space;\{0\})}" title="\{z_n\}_{n \in (\mathbb{N} \cup \{0\})}" /></a>, given by <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;z_{n&plus;1}&space;=&space;z_n^2&space;&plus;&space;c," target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;z_{n&plus;1}&space;=&space;z_n^2&space;&plus;&space;c," title="z_{n+1} = z_n^2 + c," /></a> where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;z_0&space;=&space;c&space;\in&space;\mathbb{C}\text{&space;and&space;}\mathbb{N}&space;\cup&space;\{0\}&space;=&space;\{0,1,2,3,4,\dots\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;z_0&space;=&space;c&space;\in&space;\mathbb{C}\text{&space;and&space;}\mathbb{N}&space;\cup&space;\{0\}&space;=&space;\{0,1,2,3,4,\dots\}" title="z_0 = c \in \mathbb{C}\text{ and }\mathbb{N} \cup \{0\} = \{0,1,2,3,4,\dots\}" /></a> remains bounded.That is, for any <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;$z&space;\in&space;\mathbb{C}$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;$z&space;\in&space;\mathbb{C}$" title="$z \in \mathbb{C}$" /></a> that remains on this set, there exists a <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;R_{max}\text{&space;such&space;that&space;}\vert&space;z&space;\vert&space;\leq&space;R_{max}." target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;R_{max}\text{&space;such&space;that&space;}\vert&space;z&space;\vert&space;\leq&space;R_{max}." title="R_{max}\text{ such that }\vert z \vert \leq R_{max}." /></a>

![Mandelbrot full set](https://github.com/carloselcastro/mandelbrot/blob/master/image/download.png)

## Performance tests

We performed some performance tests to compare all the methods used here: sequential, pthreads, openmp, openmpi, cuda, openmpi+openmp and openmpi+cuda. In these experiments, we measure the average time of 15 executions on each method, varying 3 different parameters:
* Dimensions (x,y) of grid and blocks (CUDA);
* Number of Threads (Pthreads and OMP);
* Number of processes (OMPI).

Algorithm | Executions
--------- | ----------
Sequential | ---
OpenMP | 2^0 to 2^10 threads
Pthreads | 2^0 to 2^10 threads
CUDA | 2^2 a 2^6 clock size
CUDA+OpenMPI (one node) | n = (2,4,...,32), for each n, 2^2 to 2^6 clock size
CUDA+OpenMPI (two nodes) | n = (2,4,...,64) in two nodes, each one with n/2 processes, for each n, 2^2 to 2^6 block size
OpenMPI (two nodes) | n = (2,4,...,32)
OpenMPI (two nodes) | n = (2,4,...,64) in two nodes, with n/2 processes each
OpenMPI com OpenMP (one node) | n = (2,4,...,32), for each n, 2^2 to 2^6 threads
OpenMPI com OpenMP (two nodes) | n = (2,4,...,64) in two nodes, with n/2 processes each and, for each n we have 2^2 to 2^6 threads

## Some results
### Best Peformance

![Best Peformance](https://github.com/carloselcastro/mandelbrot/blob/master/image/1.png)

### Worst Peformance

![Worst Performance](https://github.com/carloselcastro/mandelbrot/blob/master/image/2.png)

### Peformance by varyng the number of threads

![Peformance by varyng the number of threads](https://github.com/carloselcastro/mandelbrot/blob/master/image/3.png)

### Peformance by varyng the number of processes (one node)

![Peformance by varyng the number of processes (one node)](https://github.com/carloselcastro/mandelbrot/blob/master/image/4.png)

### Peformance by varyng the number of processes (two nodes)

![Peformance by varyng the number of processes (two node)](https://github.com/carloselcastro/mandelbrot/blob/master/image/5.png)

### Peformance by varyng the dimensions of blocks and with 8 processes (Cuda and Cuda+OpenMPI)

![Peformance by varyng the number of blocks with 8 processes (Cuda and Cuda+OpenMPI)](https://github.com/carloselcastro/mandelbrot/blob/master/image/6.png)

### Peformance by varyng the dimensions of blocks and with 32 processes (Cuda and Cuda+OpenMPI)

![Peformance by varyng the number of blocks with 32 processes (Cuda and Cuda+OpenMPI)](https://github.com/carloselcastro/mandelbrot/blob/master/image/7.png)

