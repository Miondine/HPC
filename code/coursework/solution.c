#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "mat.h"
#include "vec.h"
#include "utils.h"
#include <stdlib.h>

/* y <- Ax
 * - A: matrix
 * - x: input vector
 * - y: output vector
 */
int MatMult(Mat A, Vec x, Vec y)
{
  int ierr;
  if (A->N != x->N || A->N != y->N || x->n != A->n/A->np || x->n != y->n) {
    fprintf(stderr, "Mismatching sizes in MatMult %d %d %d\n", A->N, x->N, y->N);
    return MPI_Abort(A->comm, MPI_ERR_ARG);
  }
  //init of world comm and rank of processes in world
  MPI_Comm comm;
  comm = MPI_COMM_WORLD;

  int rank;
  int rank_row;
  int rank_col;
  int row;
  int col;

  MPI_Comm_rank(comm, &rank);
  // calc of row to which each process belongs
  row = rank / A->np;
  // calc column to which each process belongs
  col = rank % A->np;

  // init of row comm and rank of processes in row
  MPI_Comm comm_row;
  MPI_Comm_split(comm,row,rank,&comm_row);
  MPI_Comm_rank(comm_row, &rank_row);

  // init of column comm and rank of processes in column
  MPI_Comm comm_col;
  MPI_Comm_split(comm,col,rank,&comm_col);
  MPI_Comm_rank(comm_col, &rank_col);

  // gather in each column in diagonal process all values of x from row; saved in xpartial
  double *xpartial = malloc(A->n*sizeof(*xpartial));
  
  for (int i = 0; i < A->np; i++){

    if (row == i){

      MPI_Gather(x->data,x->n,MPI_DOUBLE,xpartial,x->n,MPI_DOUBLE,i,comm_row);

    }

  }

  //send xpartial from diagonal processes to each process in column
  for (int i = 0; i < A->np; i++) {

    if(col == i){
    MPI_Bcast(xpartial,A->n,MPI_DOUBLE,i,comm_col);
    }

  }

  MPI_Comm_free(&comm_col);

  //make local matrix vector mult, save solution in y_partial
  double *ypartial = malloc(A->n*sizeof(*ypartial));

  MatMultLocal(A->n,A->data,xpartial,ypartial);

  free(xpartial);

  //sum y_partial in each row and then scatter to each process
  MPI_Reduce_scatter_block(ypartial, y->data, y->n, MPI_DOUBLE, MPI_SUM, comm_row);

  MPI_Comm_free(&comm_row);
  free(ypartial);

  return 0;
}

/* C <- AB + C using the SUMMA algorithm.
 *
 * - A: input matrix
 * - B: input matrix
 * - C: output matrix
 */
int MatMatMultSumma(Mat A, Mat B, Mat C)
{
  int ierr;

  //init of world comm and rank of processes in world
  MPI_Comm comm;
  comm = MPI_COMM_WORLD;

  int rank;
  int rank_row;
  int rank_col;
  int row;
  int col;

  MPI_Comm_rank(comm, &rank);
  // calc of row to which each process belongs
  row = rank / A->np;
  // calc column to which each process belongs
  col = rank % A->np;

  // init of row comm and rank of processes in row
  MPI_Comm comm_row;
  MPI_Comm_split(comm,row,rank,&comm_row);
  MPI_Comm_rank(comm_row, &rank_row);

  // init of column comm and rank of processes in column
  MPI_Comm comm_col;
  MPI_Comm_split(comm,col,rank,&comm_col);
  MPI_Comm_rank(comm_col, &rank_col);

  int send_rank;
  double *vec_hor = malloc(A->n*sizeof(*vec_hor)); // horizontal vec for outer product
  double *vec_ver = malloc(A->n*sizeof(*vec_ver)); // vertical vec for outer product

  //loop over all process columns/rows
  for(int k = 0; k < A->np; k++){
    send_rank = k;
    //loop over all columns/rows per process
    for (int l = 0; l < A->n; l++){
      //loop over all process rows/columns
      for (int m = 0; m < A->np; m++){
        //save vec_hor in sending process
        if (col == k && row == m){
          //save data to broadcast in vec_ver
          for (int i = 0; i < A->n; i++){
            vec_hor[i] = A->data[i*A->n+l];
          }
        }
        //save vec_ver in sending process
        if (row == k && col == m){
          //save data to broadcast in vec_ver
          for (int i = 0; i < A->n; i++){
            vec_ver[i] = B->data[l*A->n+i];
          }
        }
      }
      //send vec_hor/vec_ver to all other processes of column
      MPI_Bcast(vec_hor,A->n,MPI_DOUBLE,send_rank,comm_row);
      MPI_Bcast(vec_ver,A->n,MPI_DOUBLE,send_rank,comm_col);
      //calc of outer product and sum to C
      for (int i = 0; i < A->n; i++){
        for(int j = 0; j < A->n; j++){
          C->data[i*A->n+j] += vec_ver[i] * vec_hor[j];
        }
      }
    }
  }

  free(vec_hor);
  free(vec_ver);
  MPI_Comm_free(&comm_row);
  MPI_Comm_free(&comm_col);
  return 0;
}

/* C <- AB + C using Cannon's algorithm.
 *
 * - A: input matrix
 * - B: input matrix
 * - C: output matrix
 */
int MatMatMultCannon(Mat A, Mat B, Mat C)
{
  int ierr;
  
  //init of world comm and rank of processes in world
  MPI_Comm comm;
  comm = MPI_COMM_WORLD;

  int rank;
  int rank_row;
  int rank_col;
  int row;
  int col;

  MPI_Comm_rank(comm, &rank);
  // calc of row to which each process belongs
  row = rank / A->np;
  // calc column to which each process belongs
  col = rank % A->np;

  // init of row comm and rank of processes in row
  MPI_Comm comm_row;
  MPI_Comm_split(comm,row,rank,&comm_row);
  MPI_Comm_rank(comm_row, &rank_row);

  // init of column comm and rank of processes in column
  MPI_Comm comm_col;
  MPI_Comm_split(comm,col,rank,&comm_col);
  MPI_Comm_rank(comm_col, &rank_col);

  double *sendbufA = calloc(A->n* A->n, sizeof(*sendbufA)); 
  double *sendbufB = calloc(A->n* A->n, sizeof(*sendbufB)); 

  int destination;
  int source;
  int left_neighbour;
  int right_neighbour; 
  int up_neighbour;
  int down_neighbour;

  //do initial alignment of matrices A:
  for (int i = 1; i < A->np + 1; i++){

    if (row==i) {

      for(int j=0; j<A->n * A->n; j++){

         sendbufA[j] = A->data[j];

      }

      destination = (rank_row - i + A->np) % A->np;
      source = (rank_row + i) % A->np;

      MPI_Sendrecv(sendbufA,A->n* A->n,MPI_DOUBLE,destination,0,A->data,A->n* A->n,MPI_DOUBLE,source,0,comm_row,MPI_STATUS_IGNORE);
       
    }

  }
  
 // do initial alignment of matrices B:
  for (int i = 1; i < A->np + 1; i++){

    if (col==i) {

      for(int j=0; j<A->n * A->n; j++){

         sendbufB[j] = B->data[j];

      }

    destination = (rank_col - i + A->np) % A->np;
    source = (rank_col + i) % A->np;

    MPI_Sendrecv(sendbufB,A->n* A->n,MPI_DOUBLE,destination,0,B->data,A->n* A->n,MPI_DOUBLE,source,0,comm_col,MPI_STATUS_IGNORE);

    }

  }

  // do first local Mult:
  ierr = MatMatMultLocal(A->n, A->data, B->data, C->data);CHKERR(ierr);

  // do shifts and local Mult:

  for (int i = 1; i < A->np; i++){

    left_neighbour = (rank_row - 1 + A->np) % A->np;
    right_neighbour = (rank_row + 1) % A->np;
    up_neighbour = (rank_col - 1 + A->np) % A->np;
    down_neighbour = (rank_col + 1) % A->np;

    for(int j=0; j<A->n * A->n; j++){

      sendbufA[j] = A->data[j];
      sendbufB[j] = B->data[j];
    }

    MPI_Sendrecv(sendbufB,A->n* A->n,MPI_DOUBLE,up_neighbour,0,B->data,A->n* A->n,MPI_DOUBLE,down_neighbour,0,comm_col,MPI_STATUS_IGNORE);
    MPI_Sendrecv(sendbufA,A->n* A->n,MPI_DOUBLE,left_neighbour,0,A->data,A->n* A->n,MPI_DOUBLE,right_neighbour,0,comm_row,MPI_STATUS_IGNORE);

    ierr = MatMatMultLocal(A->n, A->data, B->data, C->data);CHKERR(ierr);

  }
  //free allocated memory
  free(sendbufA);
  free(sendbufB);
  MPI_Comm_free(&comm_row);
  MPI_Comm_free(&comm_col);

  return 0;
}
