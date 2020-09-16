#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

int BLOCK_SIZE = 16;

double c_x_min_host; // CPU
double c_x_max_host;
double c_y_min_host;
double c_y_max_host;
__device__ double c_x_min_device; // GPU
__device__ double c_x_max_device;
__device__ double c_y_min_device;
__device__ double c_y_max_device;


double pixel_width_host;
double pixel_height_host;
__device__ double pixel_width_device;
__device__ double pixel_height_device;

__device__ int iteration_max = 200;

int image_size_host;
__device__ int image_size_device;

unsigned char *image_buffer_host;


int i_x_max_host;
int i_y_max_host;
__device__ int i_x_max_device;
__device__ int i_y_max_device;

int image_buffer_size_host;
__device__ int image_buffer_size_device;


__device__ int gradient_size = 16;
__device__ int colors[17][3] = {
                        {66, 30, 15},
                        {25, 7, 26},
                        {9, 1, 47},
                        {4, 4, 73},
                        {0, 7, 100},
                        {12, 44, 138},
                        {24, 82, 177},
                        {57, 125, 209},
                        {134, 181, 229},
                        {211, 236, 248},
                        {241, 233, 191},
                        {248, 201, 95},
                        {255, 170, 0},
                        {204, 128, 0},
                        {153, 87, 0},
                        {106, 52, 3},
                        {16, 16, 16},
                    };

void allocate_image_buffer_host() {
    int rgb_size = 3;
    image_buffer_host = (unsigned char *) malloc(sizeof(unsigned char ) * image_buffer_size_host * rgb_size);
}

// Aloca espaço de forma linear - row-major order
void allocate_image_buffer_device(unsigned char **image_buffer_device) {
    int rgb_size = 3;
    cudaError_t code;
    
    code = cudaMalloc((void**)&(*image_buffer_device), sizeof(unsigned char) * image_buffer_size_host * rgb_size);
    if (code != cudaSuccess) {
	printf("Could not allocate space\n");
	printf("CUDA malloc A: %s", cudaGetErrorString(code));
	exit(-1);
    }
}

void free_image_buffer_host() {
  free(image_buffer_host);
}

void free_image_buffer_device(unsigned char *image_buffer_device) {
  cudaFree(image_buffer_device);
}

void init(int argc, char *argv[]){
    if(argc < 6){
        printf("usage: ./mandelbrot_cuda c_x_min c_x_max c_y_min c_y_max image_size\n");
        printf("examples with image_size = 11500:\n");
        printf("    Full Picture:         ./mandelbrot_cuda -2.5 1.5 -2.0 2.0 11500\n");
        printf("    Seahorse Valley:      ./mandelbrot_cuda -0.8 -0.7 0.05 0.15 11500\n");
        printf("    Elephant Valley:      ./mandelbrot_cuda 0.175 0.375 -0.1 0.1 11500\n");
        printf("    Triple Spiral Valley: ./mandelbrot_cuda -0.188 -0.012 0.554 0.754 11500\n");
        exit(0);
    }
    else{
        sscanf(argv[1], "%lf", &c_x_min_host);
        sscanf(argv[2], "%lf", &c_x_max_host);
        sscanf(argv[3], "%lf", &c_y_min_host);
        sscanf(argv[4], "%lf", &c_y_max_host);
        sscanf(argv[5], "%d", &image_size_host);
	sscanf(argv[6], "%d", &BLOCK_SIZE);

        i_x_max_host           = image_size_host;
        i_y_max_host           = image_size_host;
        image_buffer_size_host = image_size_host * image_size_host;

        pixel_width_host       = (c_x_max_host - c_x_min_host) / i_x_max_host;
        pixel_height_host      = (c_y_max_host - c_y_min_host) / i_y_max_host;
    };
};

// Para chamar uma função dentro de um kernel lançado
// é necessário usar a keyword __device__
__device__ void update_rgb_buffer(int iteration, int x, int y, unsigned char *image_buffer_device){
    int color;
    int rgb_size = 3;

    if(iteration == iteration_max){
        image_buffer_device[((i_y_max_device * y) + x) * rgb_size + 0] = colors[gradient_size][0];
        image_buffer_device[((i_y_max_device * y) + x) * rgb_size + 1] = colors[gradient_size][1];
        image_buffer_device[((i_y_max_device * y) + x) * rgb_size + 2] = colors[gradient_size][2];
    }
    else{
        color = iteration % gradient_size;

        image_buffer_device[((i_y_max_device * y) + x) * rgb_size + 0] = colors[color][0];
        image_buffer_device[((i_y_max_device * y) + x) * rgb_size + 1] = colors[color][1];
        image_buffer_device[((i_y_max_device * y) + x) * rgb_size + 2] = colors[color][2];
    };
};

void write_to_file(){
    FILE * file;
    const char * filename               = "output.ppm";
    const char * comment                = "# ";

    int max_color_component_value = 255;
    int rgb_size = 3;

    file = fopen(filename,"wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
            i_x_max_host, i_y_max_host, max_color_component_value);

    for(int i = 0; i < image_buffer_size_host * rgb_size; i+=1){  
        fwrite(image_buffer_host + i, 1 , 1, file);
    };

    fclose(file);
};

__global__ void compute_mandelbrot(unsigned char *image_buffer_device){
    double z_x;
    double z_y;
    double z_x_squared;
    double z_y_squared;
    double escape_radius_squared = 4;

    int iteration;
    int i_x = blockIdx.x * blockDim.x + threadIdx.x;
    int i_y = blockIdx.y * blockDim.y + threadIdx.y;

    double c_x;
    double c_y;

    // Quando a imagem não for
    // múltiplo da quantidade
    // de threads.
    if (i_x > image_size_device || i_y > image_size_device) {
      return;
    }
    
    c_y = c_y_min_device + i_y * pixel_height_device;

    if(fabs(c_y) < pixel_height_device / 2){
	c_y = 0.0;
    };

    c_x         = c_x_min_device + i_x * pixel_width_device;

    z_x         = 0.0;
    z_y         = 0.0;

    z_x_squared = 0.0;
    z_y_squared = 0.0;

    for(iteration = 0;
	iteration < iteration_max && \
	    ((z_x_squared + z_y_squared) < escape_radius_squared);
	iteration++){
	z_y         = 2 * z_x * z_y + c_y;
	z_x         = z_x_squared - z_y_squared + c_x;

	z_x_squared = z_x * z_x;
	z_y_squared = z_y * z_y;
    };

    update_rgb_buffer(iteration, i_x, i_y, image_buffer_device);
};

__global__ void init_device_variables(double c_x_min_host, double c_x_max_host,
				      double c_y_min_host, double c_y_max_host,
				      double pixel_width_host, double pixel_height_host,
				      int image_size_host, int i_x_max_host,
				      int i_y_max_host, int image_buffer_size_host) {
    c_x_min_device = c_x_min_host;
    c_x_max_device = c_x_max_host;
    c_y_min_device = c_y_min_host;
    c_y_max_device = c_y_max_host;

    pixel_width_device = pixel_width_host;
    pixel_height_device = pixel_height_host;

    image_size_device = image_size_host;
    i_x_max_device = i_x_max_host;
    i_y_max_device = i_y_max_host;
    image_buffer_size_device = image_buffer_size_host;
}

int main(int argc, char *argv[]){
    int rgb_size = 3;
    
    init(argc, argv);

    init_device_variables<<<1, 1>>>(c_x_min_host, c_x_max_host,
				    c_y_min_host,  c_y_max_host,
				    pixel_width_host,  pixel_height_host,
				    image_size_host,  i_x_max_host,
				    i_y_max_host, image_buffer_size_host);
    
    allocate_image_buffer_host();

    unsigned char *image_buffer_device = NULL;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(image_size_host / dimBlock.x),
		 ceil(image_size_host / dimBlock.y));

    allocate_image_buffer_device(&image_buffer_device);   
    
    compute_mandelbrot<<<dimGrid, dimBlock>>>(image_buffer_device);
    
    cudaDeviceSynchronize();

    cudaMemcpy(image_buffer_host,
	       image_buffer_device,
	       sizeof(unsigned char) * image_buffer_size_host * rgb_size,
	       cudaMemcpyDeviceToHost);

    write_to_file();

    free_image_buffer_host();

    free_image_buffer_device(image_buffer_device);
    
    return 0;
};
