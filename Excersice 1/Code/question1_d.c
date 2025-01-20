#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <zlib.h> // Include ZLIB for compression and decompression

// Set 8 threads in case later the teams decide to do it with more than 4
#define MAX_THREADS 8

// Size of the matrix 
#define N 10

// Initialize variables
int thread_vals[MAX_THREADS];    // Stores initial values of threads as set in the report
int thread_exscan[MAX_THREADS];  // Stores the result of running exscan on each process on the threads
int sum_per_process = 0;         // Sum of all the thread initial values each time (they are all 10)
int base_offset = 0;             // Sum of threads sum for previous processes (they go 0,10,20,30)

/**
 * MPI_Exscan_omp performs an exclusive scan across threads within each MPI process.
 * Each thread receives the sum of the compressed sizes of all preceding threads in its process,
 * plus the sum of all compressed sizes in all preceding MPI processes.
 *
 * @param my_val      The compressed size of the calling thread.
 * @param thread_id   The ID of the calling thread within its MPI process.
 * @param num_threads The total number of threads per MPI process.
 * @param comm        The MPI communicator (typically MPI_COMM_WORLD).
 * @param result      Pointer to store the exclusive scan result for the calling thread.
 */
void MPI_Exscan_omp(int my_val, int thread_id, int num_threads, MPI_Comm comm, int* result) {
    // Store compressed sizes of each thread
    thread_vals[thread_id] = my_val;

    // Ensure all threads have stored their compressed sizes
    #pragma omp barrier

    // One thread computes the exclusive scan and the sum_per_process
    #pragma omp single
    {
        // Calculating exclusive scan for threads within the process
        thread_exscan[0] = 0; // First thread has no preceding threads
        for(int t = 1; t < num_threads; t++) {
            thread_exscan[t] = thread_exscan[t-1] + thread_vals[t-1];
        }

        // Compute the total sum of compressed sizes in this process
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

    // Each thread computes its final write offset by adding the base_offset and its exclusive scan result
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
    // These will be overwritten with compressed sizes after compression
    int initial_values[MAX_THREADS];
    for(int i = 0; i < num_threads; i++) {
        initial_values[i] = i + 1; // Thread 0 = 1, Thread 1 = 2, etc.
    }

    // Array to store the results from each thread (final write offsets)
    int results[MAX_THREADS];
    memset(results, 0, sizeof(results));

    // Matrix size and data size in bytes
    const int matrix_size = N * N * N;
    const int data_size = matrix_size * sizeof(double);

    // Name of the binary file
    const char *filename = "matrix_data_compressed.bin";

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

        // Compress the matrix using ZLIB
        uLongf compressed_bound = compressBound(data_size);
        Bytef *compressed_data = (Bytef*) malloc(compressed_bound);
        if(compressed_data == NULL) {
            fprintf(stderr, "Process %d, Thread %d: Error allocating memory for compressed data.\n", rank, thread_id);
            free(matrix);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int zlib_status = compress(compressed_data, &compressed_bound, (Bytef*)matrix, data_size);
        if(zlib_status != Z_OK) {
            fprintf(stderr, "Process %d, Thread %d: Error during compression. ZLIB status: %d\n", rank, thread_id, zlib_status);
            free(matrix);
            free(compressed_data);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Overwrite initial_values with compressed sizes for exclusive scan
        initial_values[thread_id] = compressed_bound; // Compressed size in bytes

        // Free the original matrix as it's no longer needed
        free(matrix);

        // Ensure all threads have compressed their data and updated initial_values
        #pragma omp barrier

        // Each thread calls the MPI_Exscan_omp function to get its write position index
        MPI_Exscan_omp(initial_values[thread_id], thread_id, num_threads, MPI_COMM_WORLD, &results[thread_id]);

        // Calculate the write offset in bytes
        // Each thread writes its compressed data at (base_offset + thread_exscan[thread_id]) bytes
        MPI_Offset write_offset = (MPI_Offset)(base_offset + thread_exscan[thread_id]);

        // Write the compressed data to the binary file at the calculated offset
        mpi_error = MPI_File_write_at(fh, write_offset, compressed_data, compressed_bound, MPI_BYTE, &status);
        if(mpi_error != MPI_SUCCESS) {
            fprintf(stderr, "Process %d, Thread %d: Error writing compressed data to file.\n", rank, thread_id);
            free(compressed_data);
            MPI_Abort(MPI_COMM_WORLD, mpi_error);
        }

        // Synchronize after writing
        #pragma omp barrier

        // Verification Phase: Read back the compressed data and compare
        // Allocate buffer for reading compressed data
        Bytef *read_compressed_data = (Bytef*) malloc(compressed_bound);
        if(read_compressed_data == NULL) {
            fprintf(stderr, "Process %d, Thread %d: Error allocating memory for read compressed data.\n", rank, thread_id);
            free(compressed_data);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read the compressed data back from the file at the same offset
        mpi_error = MPI_File_read_at(fh, write_offset, read_compressed_data, compressed_bound, MPI_BYTE, &status);
        if(mpi_error != MPI_SUCCESS) {
            fprintf(stderr, "Process %d, Thread %d: Error reading compressed data from file.\n", rank, thread_id);
            free(compressed_data);
            free(read_compressed_data);
            MPI_Abort(MPI_COMM_WORLD, mpi_error);
        }

        // Decompress the read data
        Bytef *decompressed_data = (Bytef*) malloc(data_size);
        if(decompressed_data == NULL) {
            fprintf(stderr, "Process %d, Thread %d: Error allocating memory for decompressed data.\n", rank, thread_id);
            free(compressed_data);
            free(read_compressed_data);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        uLongf decompressed_size = data_size;
        zlib_status = uncompress(decompressed_data, &decompressed_size, read_compressed_data, compressed_bound);
        if(zlib_status != Z_OK) {
            fprintf(stderr, "Process %d, Thread %d: Error during decompression. ZLIB status: %d\n", rank, thread_id, zlib_status);
            free(compressed_data);
            free(read_compressed_data);
            free(decompressed_data);
            MPI_Abort(MPI_COMM_WORLD, mpi_error);
        }

        // Re-initialize the original matrix for comparison
        // Since the original matrix was freed after compression, re-initialize it
        double *original_matrix = (double*) malloc(data_size);
        if(original_matrix == NULL) {
            fprintf(stderr, "Process %d, Thread %d: Error allocating memory for original matrix.\n", rank, thread_id);
            free(compressed_data);
            free(read_compressed_data);
            free(decompressed_data);
            MPI_Abort(MPI_COMM_WORLD, mpi_error);
        }

        // Re-initialize the matrix with the same seed used before compression
        seed = (unsigned int)(rank * num_threads + thread_id + 1); // Same unique seed per thread
        for(int i = 0; i < matrix_size; i++) {
            original_matrix[i] = ((double)rand_r(&seed)) / RAND_MAX; // Random double between 0 and 1
        }

        // Compare the decompressed data with the original matrix
        int verification_passed = 1;
        for(int i = 0; i < matrix_size; i++) {
            // Allow a small epsilon for floating-point comparison
            if(fabs(original_matrix[i] - ((double*)decompressed_data)[i]) > 1e-9) {
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
        free(compressed_data);
        free(read_compressed_data);
        free(decompressed_data);
        free(original_matrix);
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
            printf("Process %d has completed writing and verification with compression.\n", rank);
            fflush(stdout);
        }
    }

    MPI_Finalize();
    return 0;
}
