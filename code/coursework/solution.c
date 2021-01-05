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
  MPI_Comm_rank(comm_row, &rank_col);

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
  fprintf(stderr, "[MatMatMultSumma]: TODO, please implement me.\n");
  /* Do local part of multiplication. Only correct in serial. */
  ierr = MatMatMultLocal(A->n, A->data, B->data, C->data);CHKERR(ierr);
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
  fprintf(stderr, "[MatMatMultCannon]: TODO, please implement me.\n");
  /* Do local part of multiplication. Only correct in serial. */
  ierr = MatMatMultLocal(A->n, A->data, B->data, C->data);CHKERR(ierr);
  return 0;
}
