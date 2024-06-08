#include <stdio.h>

long TotalSize = 0;
#define N 1024 

void stack_overflow() {
    char array[N]; // Declaring a large array on the stack	
	TotalSize += 1;
	printf("allocate %ld kiloBytes\n",TotalSize);	
    stack_overflow();  // Recursive call without a base casse
	
}

int main() {
    stack_overflow();
    return 0;
}
