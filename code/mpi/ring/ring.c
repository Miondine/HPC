#include <mpi.h>
#include <stdio.h>

static void ring_reduce(const int *sendbuf, int *recvbuf, MPI_Comm comm)
{
  /* Add my local contribution */
  recvbuf[0] = sendbuf[0];
  /* TODO implement the reduction */
  /* Hint, you can compute the left and right neighbours with modular
   * arithmetic:
   *
   * (x + y) % N;
   *
   * Produces is (x + y) mod N
   *
   * For subtraction, you should add the mod:
   *
   * (x - y + N) % N;
   */
  int rank;
  int size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  int right_rank = (rank + 1) % size;
  int left_rank = (rank - 1 + size) % size;
  MPI_Sendrecv(sendbuf,1,MPI_INT,right_rank,0,recvbuf,1,MPI_INT,left_rank,0,comm,MPI_STATUS_IGNORE);

  return;
}


int main(int argc, char **argv)
{
  int rank;
  int size;
  MPI_Comm comm;
  MPI_Init(&argc, &argv);

  double start = MPI_Wtime();
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int send_value = rank;

  int received = 0;

  int summed_value = 0;

  for (int i = 0; i < size; i++) {
    printf("[%d] Before reduction: send value is %d; summed value is %d\n",
            rank, send_value, summed_value);
    ring_reduce(&send_value, &received, comm);
    summed_value += received;
    send_value = received;
    printf("[%d] After reduction: send value is %d; summed value is %d\n",
            rank, send_value, summed_value);
  }

  printf("[%d] Final values: send value is %d; summed value is %d\n",
          rank, send_value, summed_value);
  double end = MPI_Wtime();
  double total = end - start;
  MPI_Finalize();
  return 0;
}
