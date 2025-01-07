#include <mpi.h>
#include <stdio.h>

int MPI_Exscan_pt2pt(int *sendbuf, int *recvbuf, int count, MPI_Comm comm) {
    int rank, size;
    int result = 0;
    int temp;
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // Process 0 receives nothing and sets output to 0
    if (rank == 0) {
        *recvbuf = 0;
        // Send its value to next process
        if (size > 1) {
            MPI_Send(sendbuf, 1, MPI_INT, 1, 0, comm);
        }
    } 
    else {
        // Receive from previous rank
        MPI_Recv(&temp, 1, MPI_INT, rank-1, 0, comm, MPI_STATUS_IGNORE);
        
        if (rank < size - 1) {
            // Calculate running sum and send to next process
            result = temp + *sendbuf;
            MPI_Send(&result, 1, MPI_INT, rank+1, 0, comm);
        }
        
        // Store received value as result
        *recvbuf = temp;
    }
    
    return MPI_SUCCESS;
}

int main(int argc, char **argv) {
    int rank, size;
    int sendval, recvval;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Each process has its rank as input value
    sendval = rank + 1;
    
    // Call our implementation
    MPI_Exscan_pt2pt(&sendval, &recvval, 1, MPI_COMM_WORLD);
    
    printf("Process %d: Input = %d, Prefix sum = %d\n", 
           rank, sendval, recvval);
    
    MPI_Finalize();
    return 0;

//mpicc github1.c -o github1
//mpirun -np 4 ./github1
}