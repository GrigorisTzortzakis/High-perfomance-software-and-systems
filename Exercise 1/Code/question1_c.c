#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>
#include <math.h>

// Set 8 threads in case later the teams decide to do it with more than 4
#define MAX_THREADS 8

// Size of the matrix 
#define N 10

// Initialize variables
int thread_vals[MAX_THREADS];    // Stores initial values of threads as set in the report
int thread_exscan[MAX_THREADS];  // Stores the result of running exscan on each process on the threads
int sum_per_process = 0;         // Sum of all the thread initial values each time (they are all 10)
int base_offset = 0;             // Sum of threads sum for previous processes (they go 0,10,20,30)

void MPI_Exscan_omp(int my_val, int thread_id, int num_threads, MPI_Comm comm, int* result) {
    // Store initial values of each thread as specified
    thread_vals[thread_id] = my_val;

    // Ensure all threads have stored their values
    #pragma omp barrier

    // One thread computes the exclusive scan and the sum_per_process
    #pragma omp single
    {
        // Calculating exclusive scan for threads within the process
        thread_exscan[0] = 0; // First thread has no preceding threads
        for(int t = 1; t < num_threads; t++) {
            thread_exscan[t] = thread_exscan[t-1] + thread_vals[t-1];
        }

        // Compute the total sum of initial thread values in this process
        sum_per_process = 0;
        for(int t = 0; t < num_threads; t++) {
            sum_per_process += thread_vals[t];
        }
    }

    // Ensure that the exclusive scan and sum_per_process are computed
    #pragma omp barrier

    // One thread performs the MPI exclusive scan to get the base_offset
    #pragma omp single
    {
        MPI_Exscan(&sum_per_process, &base_offset, 1, MPI_INT, MPI_SUM, comm);

        // For process 0, MPI_Exscan does not set the base_offset; it should be 0
        int rank;
        MPI_Comm_rank(comm, &rank);
        if(rank == 0) {
            base_offset = 0;
        }
    }

    // Ensure that the base_offset is computed
    #pragma omp barrier

    // Each thread computes its final result by adding the base_offset
    *result = base_offset + thread_exscan[thread_id];
}

int main(int argc, char *argv[]) {
    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if(provided < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "Error: MPI does not provide needed threading level (MPI_THREAD_MULTIPLE).\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define the number of threads per MPI process
    const int num_threads = 4; // Adjust as needed, <= MAX_THREADS

    if(num_threads > MAX_THREADS) {
        if(rank == 0) {
            fprintf(stderr, "Error: num_threads (%d) exceeds MAX_THREADS (%d).\n", num_threads, MAX_THREADS);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Define initial values for each thread
    // Set initial_values to [1, 2, 3, 4] for threads 0,1,2,3 respectively
    int initial_values[MAX_THREADS];
    for(int i = 0; i < num_threads; i++) {
        initial_values[i] = i + 1; // Thread 0 = 1, Thread 1 = 2, etc.
    }

    // Array to store the results from each thread
    int results[MAX_THREADS];
    memset(results, 0, sizeof(results));

    // Matrix size and data size in bytes
    const int matrix_size = N * N * N;
    const int data_size = matrix_size * sizeof(double);

    // Name of the binary file
    const char *filename = "matrix_data.bin";

    // Open the binary file for writing and reading (create or overwrite)
    MPI_File fh;
    MPI_Status status;
    int mpi_error;

    mpi_error = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    if(mpi_error != MPI_SUCCESS) {
        fprintf(stderr, "Process %d: Error opening file for writing and reading.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, mpi_error);
    }

    // Start parallel region with OpenMP
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int my_val = initial_values[thread_id];
        int thread_result = 0;

        // Each thread calls the MPI_Exscan_omp function to get its write position index
        MPI_Exscan_omp(my_val, thread_id, num_threads, MPI_COMM_WORLD, &thread_result);

        // Store the result in the shared results array
        results[thread_id] = thread_result;

        // Calculate the write offset in bytes
        // Each thread writes its matrix at (base_offset + exscan_result) * data_size
        MPI_Offset write_offset = ((MPI_Offset)(base_offset + thread_exscan[thread_id])) * data_size;

        // Allocate and initialize the matrix with a unique random seed
        double *matrix = (double*) malloc(data_size);
        if(matrix == NULL) {
            fprintf(stderr, "Process %d, Thread %d: Error allocating memory for matrix.\n", rank, thread_id);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Initialize the matrix with random numbers using a unique seed
        unsigned int seed = (unsigned int)(rank * num_threads + thread_id + 1); // Unique seed per thread
        for(int i = 0; i < matrix_size; i++) {
            matrix[i] = ((double)rand_r(&seed)) / RAND_MAX; // Random double between 0 and 1
        }

        // Write the matrix to the binary file at the calculated offset
        mpi_error = MPI_File_write_at(fh, write_offset, matrix, matrix_size, MPI_DOUBLE, &status);
        if(mpi_error != MPI_SUCCESS) {
            fprintf(stderr, "Process %d, Thread %d: Error writing to file.\n", rank, thread_id);
            free(matrix);
            MPI_Abort(MPI_COMM_WORLD, mpi_error);
        }

        // Synchronize after writing
        #pragma omp barrier

        // Verification Phase: Read back the data and compare
        // Allocate buffer for reading
        double *read_buffer = (double*) malloc(data_size);
        if(read_buffer == NULL) {
            fprintf(stderr, "Process %d, Thread %d: Error allocating memory for read buffer.\n", rank, thread_id);
            free(matrix);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read the data back from the file at the same offset
        mpi_error = MPI_File_read_at(fh, write_offset, read_buffer, matrix_size, MPI_DOUBLE, &status);
        if(mpi_error != MPI_SUCCESS) {
            fprintf(stderr, "Process %d, Thread %d: Error reading from file.\n", rank, thread_id);
            free(matrix);
            free(read_buffer);
            MPI_Abort(MPI_COMM_WORLD, mpi_error);
        }

        // Compare the read data with the original matrix
        int verification_passed = 1;
        for(int i = 0; i < matrix_size; i++) {
            // Allow a small epsilon for floating-point comparison
            if(fabs(matrix[i] - read_buffer[i]) > 1e-9) {
                verification_passed = 0;
                break;
            }
        }

        if(verification_passed) {
            printf("Process %d, Thread %d: Verification PASSED.\n", rank, thread_id);
        }
        else {
            printf("Process %d, Thread %d: Verification FAILED.\n", rank, thread_id);
        }

        // Clean up
        free(matrix);
        free(read_buffer);
    }

    // Close the file after all threads have finished writing and verifying
    mpi_error = MPI_File_close(&fh);
    if(mpi_error != MPI_SUCCESS) {
        fprintf(stderr, "Process %d: Error closing the file.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, mpi_error);
    }

    // Synchronize all processes before finalizing
    MPI_Barrier(MPI_COMM_WORLD);

    // Only one thread per process prints the summary
    #pragma omp parallel num_threads(1)
    {
        #pragma omp single
        {
            printf("Process %d has completed writing and verification.\n", rank);
            fflush(stdout);
        }
    }

    MPI_Finalize();
    return 0;
}

