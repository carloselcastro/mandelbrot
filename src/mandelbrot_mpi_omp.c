#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "mpi.h"
#define  MASTER     0
#define  BEGIN      1
#define  DONE       4
#define  NONE       0

/* Localhost instructions:
 * After 
 *
 *                                 $ make
 *
 * you have to run in your terminal 
 *
 *                   $ mpirun --host localhost:NUMTASKS mandelbrot_mpi
 *
 *where NUMTASKS is the number of tasks.*/ 

int image_size = 4096;
double c_x_min = -0.188;
double c_x_max = -0.012;
double c_y_min = 0.554;
double c_y_max = 0.754;
int i_x_max = 4096;
int i_y_max = 4096;


int iteration_max = 200;


unsigned char *image_buffer;


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

void allocate_image_buffer(){
    int image_buffer_size = image_size * image_size;
    int rgb_size = 3;
    
    image_buffer = (unsigned char *) malloc(sizeof(unsigned char ) * image_buffer_size * rgb_size);
};

void free_image_buffer() {
    free(image_buffer);
}


void update_rgb_buffer(int iteration, int x, int y){
    int color;
    int rgb_size = 3;

    if(iteration == iteration_max){
        image_buffer[((i_y_max * y) + x) * rgb_size + 0] = colors[gradient_size][0];
        image_buffer[((i_y_max * y) + x) * rgb_size + 1] = colors[gradient_size][1];
        image_buffer[((i_y_max * y) + x) * rgb_size + 2] = colors[gradient_size][2];
    }
    else{
        color = iteration % gradient_size;

        image_buffer[((i_y_max * y) + x) * rgb_size + 0] = colors[color][0];
        image_buffer[((i_y_max * y) + x) * rgb_size + 1] = colors[color][1];
        image_buffer[((i_y_max * y) + x) * rgb_size + 2] = colors[color][2];
    };
};

void write_to_file(){
    FILE * file;
    char * filename               = "output.ppm";
    char * comment                = "# ";
    int image_buffer_size = image_size * image_size;
    int max_color_component_value = 255;
    int rgb_size = 3;
    
    file = fopen(filename,"wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
            i_x_max, i_y_max, max_color_component_value);

    for(int i = 0; i < image_buffer_size * rgb_size; i+=1){  
        fwrite(image_buffer + i, 1 , 1, file);
    };

    fclose(file);
};

void compute_mandelbrot(int start, int end, uint8_t *vec_results){
    double z_x;
    double z_y;
    double z_x_squared;
    double z_y_squared;
    double escape_radius_squared = 4;

    uint8_t iteration;
    int i_x;
    int i_y;

    double c_x;
    double c_y;
    double pixel_width = (c_x_max - c_x_min) / i_x_max;
    double pixel_height = (c_y_max - c_y_min) / i_y_max;
    for(i_y = start; i_y < end; i_y++){
        c_y = c_y_min + i_y * pixel_height;

        if(fabs(c_y) < pixel_height / 2){
            c_y = 0.0;
        };

        for(i_x = 0; i_x < i_x_max; i_x++){
            c_x         = c_x_min + i_x * pixel_width;

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
            vec_results[(i_y - start)*i_y_max+i_x] = iteration;
            //update_rgb_buffer(iteration, i_x, i_y);
        };
    };
};

void update_mpi_rgb_buffer(uint8_t *results, int start, int tamanho){
    int j = start-1;
    for (int i=0; i< tamanho; i++){
        if (i % i_y_max == 0){
            j++;
        }
        update_rgb_buffer(results[i], i % i_x_max, j);
    }
}

int main(int argc, char *argv[]){
    
    int taskid,                     /* this task's unique id */
        numtasks,                   /* number of tasks */
    	numworkers,
        averow,rows,offset,extra,   /* for sending rows of data */
        dest, source,               /* to - from for message send-receive */
        msgtype,                    /* for message types */
        start,end,               /* misc */
        i;              /* loop variables */

    MPI_Status status;
    
    // void compute_mandelbrot(int start, int end);
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    numworkers = numtasks-1;
    
    averow = image_size/numworkers;
    extra = image_size % numworkers;
    
    offset = 0;

    if (taskid == MASTER){
	rows = averow;
	for (i=1; i <=numworkers; i++)
	    {
		/*  Now send startup information to each worker  */
		dest = i;
		MPI_Send(&offset, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
		MPI_Send(&rows, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
            
	    
		offset = offset + rows;
	    }
	allocate_image_buffer();
	
	uint8_t *vec_results;
	vec_results = (uint8_t *)malloc(rows*i_x_max*sizeof(uint8_t));
      
	/* Now wait for results from all worker tasks */
	for (i=1; i<=numworkers; i++)
	    {
		source = i;
		msgtype = DONE;
		MPI_Recv(&offset, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD,&status);
		MPI_Recv(&rows, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
		MPI_Recv(vec_results, averow*i_y_max, MPI_BYTE, source, msgtype, MPI_COMM_WORLD, &status);
		update_mpi_rgb_buffer(vec_results, offset, i_x_max*rows);
	  
	    }
	MPI_Finalize();

	if (extra != 0){
            uint8_t *vec_results;
            vec_results = (uint8_t *)malloc(extra*i_x_max*sizeof(uint8_t));
            compute_mandelbrot(image_size - extra,image_size, vec_results);
            update_mpi_rgb_buffer(vec_results, image_size - extra, i_x_max*extra);
        }
	
	write_to_file();
	free_image_buffer();
    }else{
	source = MASTER;
	msgtype = BEGIN;
	MPI_Recv(&offset, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
	MPI_Recv(&rows, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
	start=offset;
	end=offset+rows; 
      
	uint8_t *vec_results;
	vec_results = (uint8_t *)malloc(rows*i_x_max*sizeof(uint8_t));
      
      
	compute_mandelbrot(start,end, vec_results);
      
      
	MPI_Send(&offset, 1, MPI_INT, MASTER, DONE, MPI_COMM_WORLD);
	MPI_Send(&rows, 1, MPI_INT, MASTER, DONE, MPI_COMM_WORLD);        
	MPI_Send(vec_results, rows*i_y_max, MPI_BYTE, MASTER, DONE, MPI_COMM_WORLD);
	MPI_Finalize();
    }//End workers
    
    return 0;
};
