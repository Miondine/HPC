// This file is part of the HPC workshop 2019 at Durham University
// Author: Christian Arnold

#include "proto.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

/* This function the numbers and prints the result. */
void add_numbers(int n_numbers, float *numbers) {
  printf("Adding %i numbers ...\n", n_numbers);

  float result = 0;
  clock_t start = clock();

  /* do the actual calculation */
  for (int i = 0; i < n_numbers; i++) {
    double result_i = fabs(log(pow(fabs(numbers[i]), 2.1))) +
                      log(pow(fabs(numbers[i]), 1.9)) +
                      log(pow(fabs(numbers[i]), -1.97)) +
                      fabs(log(pow(fabs(numbers[i]), -1.005)));
    if (result_i > 0) {
      result += result_i;
    }
    if (result > 1e7){
      break;
    }
  }

  /* time the calculation */
  clock_t end = clock();
  double time = ((double)end - start) / CLOCKS_PER_SEC;

  printf("The result is: %g\n", result);
  printf("Doing %i calculations took %g s\n", n_numbers, time);
}
