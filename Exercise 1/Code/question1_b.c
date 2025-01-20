#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

// Define the maximum number of threads (adjust as needed)
#define MAX_THREADS 8

// Global shared variables
int thread_vals[MAX_THREADS];    // Stores initial values of threads
int thread_exscan[MAX_THREADS];  // Stores exclusive scan results per thread
int sum_per_process = 0;         // Sum of all initial thread values in the process
int base_offset = 0;             // Base offset from MPI exclusive scan

/**
 * MPI_Exscan_omp performs an exclusive scan across threads within each MPI process.
 * Each thread receives the sum of the initial values of all preceding threads in its process,
 * plus the sum of all initial thread values in all preceding MPI processes.
 *
 * @param my_val      The initial integer value of the calling thread.
 * @param thread_id   The ID of the calling thread within its MPI process.
 * @param num_threads The total number of threads per MPI process.
 * @param comm        The MPI communicator (typically MPI_COMM_WORLD).
 * @param result      Pointer to store the exclusive scan result for the calling thread.
 */
void MPI_Exscan_omp(int my_val, int thread_id, int num_threads, MPI_Comm comm, int* result) {
    // Each thread stores its initial value in the global array
    thread_vals[thread_id] = my_val;

    // Ensure all threads have stored their values
    #pragma omp barrier

    // One thread computes the exclusive scan and the sum_per_process
    #pragma omp single
    {
        // Compute the exclusive scan for threads within the process
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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    if(provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Error: MPI does not provide needed threading level\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define the number of threads per MPI process
    const int num_threads = 4;

    // Define initial values for each thread
    // All processes have threads with initial values: 1, 2, 3, 4
    int initial_values[4] = {1, 2, 3, 4};

    // Array to store the results from each thread
    int results[4] = {0};

    // Start parallel region with OpenMP
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int my_val = initial_values[thread_id];
        int thread_result = 0;

        // Each thread calls the MPI_Exscan_omp function
        MPI_Exscan_omp(my_val, thread_id, num_threads, MPI_COMM_WORLD, &thread_result);

        // Store the result in the shared results array
        results[thread_id] = thread_result;
    }

    // Synchronize all processes before printing
    MPI_Barrier(MPI_COMM_WORLD);

    // Only one thread per process prints the results to avoid jumbled output
    #pragma omp parallel num_threads(1)
    {
        #pragma omp single
        {
            printf("Process %d results:\n", rank);
            for(int t = 0; t < num_threads; t++) {
                printf("  Thread %d: %d\n", t, results[t]);
            }
            fflush(stdout);
        }
    }

    MPI_Finalize();
    return 0;
}

