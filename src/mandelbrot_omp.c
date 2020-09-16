#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Localhost instructions:
* After 
*
*                                 $ make
*
* you have to run in your terminal 
*
*                   $ ./mandelbrot_omp NUMTHREADS
*
*where NUMTHREADS is the number of threads.*/ 

double c_x_min = -0.188;
double c_x_max = -0.012;
double c_y_min = 0.554;
double c_y_max = 0.754;

double pixel_width;
double pixel_height;

int iteration_max = 200;

int image_size = 4096;
unsigned char **image_buffer;

int i_x_max;
int i_y_max;
int image_buffer_size;
int n_threads;

int gradient_size = 16;
int colors[17][3] = {
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


void compute_mandelbrot();
void allocate_image_buffer();
void init(int argc, char *argv[]);
void update_rgb_buffer(int iteration, int x, int y);
void write_to_file();
void compute_mandelbrot();
void compute_mandelbrot_omp(int x, int y, int max);
void calcule(int i_x, int i_y);

int main(int argc, char *argv[]){
    init(argc, argv);

    
 
    allocate_image_buffer();

    compute_mandelbrot();

    write_to_file();

    return 0;
};


void allocate_image_buffer(){
    int rgb_size = 3;
    image_buffer = (unsigned char **) malloc(sizeof(unsigned char *) * image_buffer_size);

    for(int i = 0; i < image_buffer_size; i++){
        image_buffer[i] = (unsigned char *) malloc(sizeof(unsigned char) * rgb_size);
    };
};

void init(int argc, char *argv[]){
    sscanf(argv[1], "%d", &n_threads);

    i_x_max           = image_size;
    i_y_max           = image_size;
    image_buffer_size = image_size * image_size;
    pixel_width       = (c_x_max - c_x_min) / i_x_max;
    pixel_height      = (c_y_max - c_y_min) / i_y_max;
};

void update_rgb_buffer(int iteration, int x, int y){
    int color;
    // if(y >= 2000)
    //     printf("%d \n", x);
    if(iteration == iteration_max){
        image_buffer[(i_y_max * y) + x][0] = colors[gradient_size][0];
        image_buffer[(i_y_max * y) + x][1] = colors[gradient_size][1];
        image_buffer[(i_y_max * y) + x][2] = colors[gradient_size][2];
    }
    else{
        color = iteration % gradient_size;

        image_buffer[(i_y_max * y) + x][0] = colors[color][0];
        image_buffer[(i_y_max * y) + x][1] = colors[color][1];
        image_buffer[(i_y_max * y) + x][2] = colors[color][2];
    };
};

void write_to_file(){
    FILE * file;
    char * filename               = "output.ppm";
    char * comment                = "# ";

    int max_color_component_value = 255;

    file = fopen(filename,"wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
            i_x_max, i_y_max, max_color_component_value);

    for(int i = 0; i < image_buffer_size; i++){
        fwrite(image_buffer[i], 1 , 3, file);
    };

    fclose(file);
};

void compute_mandelbrot(){
    int x = 0; int y = -1; int max, i;
    max = ceil(i_x_max / ((float) n_threads));

    #pragma omp parallel for private(x, y) shared(max) num_threads(n_threads)
    for(i = 0; i < (n_threads * n_threads); i++){
        x =  i % n_threads;
        y = i / n_threads;
        compute_mandelbrot_omp(x * max, y * max , max);
    };    
};

void calcule(int i_x, int i_y){
    double z_x;
    double z_y;
    double z_x_squared;
    double z_y_squared;
    double escape_radius_squared = 4;

    int iteration;

    double c_x;
    double c_y;

    c_y = c_y_min + i_y * pixel_height;

    if(fabs(c_y) < pixel_height / 2){
        c_y = 0.0;
    }; // ENDIF
    c_x         = c_x_min + i_x * pixel_width;

    z_x         = 0.0;
    z_y         = 0.0;

    z_x_squared = 0.0;
    z_y_squared = 0.0;
    for(iteration = 0; iteration < iteration_max && \
            ((z_x_squared + z_y_squared) < escape_radius_squared);
            iteration++){
        z_y         = 2 * z_x * z_y + c_y;
        z_x         = z_x_squared - z_y_squared + c_x;

        z_x_squared = z_x * z_x;
        z_y_squared = z_y * z_y;
    }; // ENDFOR

    update_rgb_buffer(iteration, i_x, i_y);
}
void compute_mandelbrot_omp(int x, int y, int max){

    for(int i_y = y; (i_y < max + y && i_y < i_y_max); i_y++){
        for(int i_x = x; (i_x < max +  x && i_x < i_x_max); i_x++){
           calcule(i_x, i_y);
        }; //ENDFOR
    };
}
