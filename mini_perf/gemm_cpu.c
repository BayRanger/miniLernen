#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 500

void initialize_matrix(int matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = rand() % 100;
        }
    }
}

void initialize_vector(int vector[SIZE]) {
    for (int i = 0; i < SIZE; i++) 
    {
        vector[i] = rand() % 100;
    }
}


void multiply_matrices(int a[SIZE][SIZE], int b[SIZE][SIZE], int result[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void vectorAdd(int vec1[SIZE], int vec2[SIZE], int vec[SIZE])
{
    printf("perform vectoAdd\n");
    for (int i = 0; i < SIZE; i++)
    for (int i = 0; i < SIZE; i++)
    for (int i = 0; i < SIZE; i++)
    {
        vec[i] = vec1[i] + vec2[i];
    }
}



int main() {
    int a[SIZE][SIZE], b[SIZE][SIZE], result[SIZE][SIZE];
    int vec_a[SIZE], vec_b[SIZE], vec_result[SIZE];
    // Initialize random number generator
    srand(time(NULL));

    // Initialize matrices
    initialize_matrix(a);
    initialize_matrix(b);

    // Multiply matrices
    multiply_matrices(a, b, result);


    initialize_vector(vec_a);
    initialize_vector(vec_b);
    vectorAdd(vec_a, vec_b, vec_result);



#ifdef SHOWREULST
    // Print a part of the result matrix to verify the operation
    printf("Result matrix:\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%d ", result[i][j]);
        }
        printf("\n");
    }

#else
    printf("############### Finish caculation #################\n");
#endif

    return 0;
}

