#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <cuda_runtime.h>
#include <math.h>


__device__ void
vectorAdd1(int* d_A, int* d_B, int* d_C, int size, int* mapBlk, int blockDim){
	int vId = threadIdx.x + mapBlk[blockIdx.x]*blockDim;
	if(vId < size){
		d_C[vId] = d_A[vId] + d_B[vId];
	}

}

__device__ void dummy(int n){
	for(int i = 0; i < n; i++);
}
__global__ void
scheduler(int* d_A1, int* d_B1, int* d_C1, int size1, int dummyLoopIterations, int* mapBlk, int* mapKernel, int gridDim_A, int blockDim_A){

	if(mapKernel[blockIdx.x] == 1)
		vectorAdd1(d_A1, d_B1, d_C1, size1, mapBlk, blockDim_A);
	else
		dummy(dummyLoopIterations);


}

int main(){

	/*Ilość elementów wektora*/
	const int numElements = 1000000;
	size_t size = numElements*sizeof(int);

	/* Ilość obrotów pętli for w dummy - wysterowana metodą prób
	 * i błędów tak, aby czas wykonania dummy był taki sam
	 * jak kernela.
	 */
	int dummyIterations = 4;

	/*Ilość bloków wątków kernela vectorAdd1*/
	const int TBi = 1000;
	const int M = 15;

	int tests = 8;

	int m[8] = {1, 2, 4, 6, 8, 10, 12, 14};
	int* dummies = new int[tests];
	int* mods = new int[tests];

	/*
		 * dummies[i] = [(TBi*(M-m))/m] ( [] - oznaczają sufit)
		 * dummies[i] -> liczba bloków wątków dummy
		 * m[i] -> liczba SM'ów, które chcemy zaalokować dla kernela
		 * M -> liczba SM'ów w naszym procesorze GPU, czyli 15 (info z benchmarka deviceQuery)
		 */
	for(int i = 0; i < tests; i++)
	{
		dummies[i] = ceil((TBi*(M-m[i]))/m[i]);
		if(dummies[i] >= TBi)
			mods[i] = floor((TBi + dummies[i])/TBi);
		else
			mods[i] = 2;

	}

	for(int i = 0; i < tests; i++)
	{
			int di = dummies[i];
			int mod = mods[i];
			int blocks = TBi+di;;
			int threads = 1000;
			int* mapKernel = new int[blocks];
			int* mapBlk = new int[blocks];

			int dummy = 0; // identyfikator dummy
			int vector = 1; //identyfikator vector
			int id1;
			int id2;

			int minTB; // min(TBi, di)
			if(TBi + di < 2*TBi)
			{
				minTB = di;
				id1 = dummy;
				id2 = vector;
			}
			else
			{
				minTB = TBi;
				id1 = vector;
				id2 = dummy;
			}




				for(int blkA =0, blkB = 0, i = 0; i < blocks; i++)
				{

					if(i%mod == 0)
					{
						if(blkA < minTB) //if(blkA < TBi)
						{
						mapKernel[i] = id1;
						mapBlk[i] =blkA++;
						}
						else
						{
							mapKernel[i] = id2;
							mapBlk[i] = blkB++;
						}
					}
					else
					{
						mapKernel[i] = id2;
						mapBlk[i] = blkB++;
					}

				}

			/*Alokowanie pamięci na wektory na hoście*/

			int* h_A1 = (int*)malloc(size);
			int* h_B1 = (int*)malloc(size);
			int* h_C1 = (int*)malloc(size);

			/*Wylosowanie liczb do wektorów*/
			srand( time(NULL));
			for(int i = 0; i < numElements; i++){
				h_A1[i] = (int)rand()%10000;
				h_B1[i] = (int)rand()%10000;
			}

			int* d_A1 = NULL;
			int* d_B1 = NULL;
			int* d_C1 = NULL;

			cudaError_t err = cudaSuccess;


			err = cudaMalloc((void**)&d_A1, size);
			if(err != cudaSuccess)
			{
				printf("Nie udalo sie zaalokowac pamieci na urzadzeniu ( kod bledu %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);

			}
			err = cudaMalloc((void**)&d_B1, size);
			if(err != cudaSuccess)
			{
				printf("Nie udalo sie zaalokowac pamieci na urzadzeniu ( kod bledu %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);

			}
			err = cudaMalloc((void**)&d_C1, size);
			if(err != cudaSuccess)
			{
				printf("Nie udalo sie zaalokowac pamieci na urzadzeniu ( kod bledu %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);

			}

			/*Memcpy host to device*/
			printf("Kopiowanie wektorów z hosta do urzadzenia...\n");

			err = cudaMemcpy(d_A1, h_A1, size, cudaMemcpyHostToDevice);
			if(err != cudaSuccess)
			{
				printf("Nie udalo sie skopiowac danych do urzadzenia( kod bledu %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
			err = cudaMemcpy(d_B1, h_B1, size, cudaMemcpyHostToDevice);
				if(err != cudaSuccess)
				{
					printf("Nie udalo sie skopiowac danych do urzadzenia( kod bledu %s)!\n", cudaGetErrorString(err));
					exit(EXIT_FAILURE);
				}

			/*Skopiowanie tablic mapujacych do urzadzenia*/
			size_t mapSize = blocks*sizeof(int);
			int* d_mapKernel = NULL;
			int* d_mapBlk = NULL;

			printf("Kopiowanie tablic mapujacych do urzadzenia...\n");
			err = cudaMalloc((void**)&d_mapKernel,mapSize);
			if(err != cudaSuccess)
				{
					printf("Nie udalo sie zaalokowac pamieci na urzadzeniu ( kod bledu %s)!\n", cudaGetErrorString(err));
					exit(EXIT_FAILURE);

				}
			err = cudaMalloc((void**)&d_mapBlk, mapSize);
			if(err != cudaSuccess)
				{
					printf("Nie udalo sie zaalokowac pamieci na urzadzeniu ( kod bledu %s)!\n", cudaGetErrorString(err));
					exit(EXIT_FAILURE);

				}

			err = cudaMemcpy(d_mapKernel, mapKernel, mapSize, cudaMemcpyHostToDevice);
			if(err != cudaSuccess)
				{
					printf("Nie udalo sie skopiowac map do urzadzenia( kod bledu %s)!\n", cudaGetErrorString(err));
					exit(EXIT_FAILURE);

				}
			err = cudaMemcpy(d_mapBlk, mapBlk, mapSize, cudaMemcpyHostToDevice);
			if(err != cudaSuccess)
				{
					printf("Nie udalo sie skopiowac map do urzadzenia ( kod bledu %s)!\n", cudaGetErrorString(err));
					exit(EXIT_FAILURE);

				}

			/*Launching kernel!!!*/
			scheduler<<<blocks,threads>>>(d_A1, d_B1, d_C1, size, dummyIterations, d_mapBlk, d_mapKernel, blocks, threads);

			/*Memcpy device to host*/
			err = cudaMemcpy(h_C1, d_C1, size, cudaMemcpyDeviceToHost);
			if(err != cudaSuccess){

				printf("Nie udalo sie skopiowac danych z urzadzenia do hosta ( kod bledu %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
			/*Checking the resultes*/
			for(int i = 0; i < numElements; i++)
			{
				int w;
				w = h_A1[i]+h_B1[i];
				if( w != h_C1[i])
				{
					printf("nr %d Wynik 1 niepoprawny %d + %d != %d!!\n",i, h_A1[i], h_B1[i], h_C1[i] );
					return 1;
				}
			}

			printf("Wynik poprawny!!\n");
	}
	return 0;
}
