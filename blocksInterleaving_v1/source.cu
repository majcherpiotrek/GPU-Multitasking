#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <cuda_runtime.h>


__device__ void
vectorAdd1(int* d_A, int* d_B, int* d_C, int size, int* mapBlk, int blockDim){
	int vId = threadIdx.x + mapBlk[blockIdx.x]*blockDim;
	if(vId < size){
		d_C[vId] = d_A[vId] + d_B[vId];
	}
	for(int i =0; i<100000;i++);
}

__device__ void
vectorAdd2(int* d_A, int* d_B, int* d_C, int size, int* mapBlk, int blockDim){
	int vId = threadIdx.x + mapBlk[blockIdx.x]*blockDim;
		if(vId < size){
			d_C[vId] = d_A[vId] + d_B[vId];
		}
		for(int i =0; i<100000;i++);
}

__global__ void
scheduler(int* d_A1, int* d_B1, int* d_C1, int size1, int* d_A2, int* d_B2, int* d_C2, int size2, int* mapBlk, int* mapKernel, int gridDim_A, int blockDim_A, int gridDim_B, int blockDim_B ){

	if(mapKernel[blockIdx.x] == 0)
		vectorAdd1(d_A1, d_B1, d_C1, size1, mapBlk, blockDim_A);
	else
		vectorAdd2(d_A2, d_B2, d_C2, size2, mapBlk, blockDim_A);
}

int main(){

	const int numElements = 1000000;
	size_t size = numElements*sizeof(int);

	int blocks = 2000;
	int threads = 1000;
	int* mapKernel = new int[blocks];
	int* mapBlk = new int[blocks];

		/*Będą wykonywały się na zmianę po jednym bloku*/
		for(int blkA =0, blkB = 0, i = 0; i < blocks; i++)
		{
			mapKernel[i] = i%2;
			if(mapKernel[i] == 0)
				mapBlk[i] = blkA++;
			else
				mapBlk[i] = blkB++;

		}

	/*Alokowanie pamięci na wektory na hoście*/

	int* h_A1 = (int*)malloc(size);
	int* h_B1 = (int*)malloc(size);
	int* h_C1 = (int*)malloc(size);

	int* h_A2 = (int*)malloc(size);
	int* h_B2 = (int*)malloc(size);
	int* h_C2 = (int*)malloc(size);

	/*Wylosowanie liczb do wektorów*/
	srand( time(NULL));
	for(int i = 0; i < numElements; i++){
		h_A1[i] = (int)rand()%10000;
		h_B1[i] = (int)rand()%10000;

		h_A2[i] = (int)rand()%10000;
		h_B2[i] = (int)rand()%10000;
	}

	int* d_A1 = NULL;
	int* d_B1 = NULL;
	int* d_C1 = NULL;

	int* d_A2 = NULL;
	int* d_B2 = NULL;
	int* d_C2 = NULL;

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
	err = cudaMalloc((void**)&d_A2, size);
	if(err != cudaSuccess)
	{
		printf("Nie udalo sie zaalokowac pamieci na urzadzeniu ( kod bledu %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);

	}
	err = cudaMalloc((void**)&d_B2, size);
	if(err != cudaSuccess)
	{
		printf("Nie udalo sie zaalokowac pamieci na urzadzeniu ( kod bledu %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);

	}
	err = cudaMalloc((void**)&d_C2, size);
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
	err = cudaMemcpy(d_A2, h_A2, size, cudaMemcpyHostToDevice);
		if(err != cudaSuccess)
		{
			printf("Nie udalo sie skopiowac danych do urzadzenia( kod bledu %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	err = cudaMemcpy(d_B2, h_B2, size, cudaMemcpyHostToDevice);
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
	scheduler<<<blocks,threads>>>(d_A1, d_B1, d_C1, size, d_A2, d_B2, d_C2, size, d_mapBlk, d_mapKernel, blocks, threads, blocks, threads );

	/*Memcpy device to host*/
	err = cudaMemcpy(h_C1, d_C1, size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){

		printf("Nie udalo sie skopiowac danych z urzadzenia do hosta ( kod bledu %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(h_C2, d_C2, size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){

		printf("Nie udalo sie skopiowac danych z urzadzenia do hosta ( kod bledu %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	/*Checking the resultes*/
	for(int i = 0; i < numElements; i++)
	{
		int w1,w2;
		w1 = h_A1[i]+h_B1[i];
		w2 = h_A2[i]+h_B2[i];
		if( w1 != h_C1[i])
		{
			printf("nr %d Wynik 1 niepoprawny %d + %d != %d!!\n",i, h_A1[i], h_B1[i], h_C1[i] );
			return 1;
		}
		if( w2 != h_C2[i])
				{
					printf("nr %d Wynik 2 niepoprawny %d + %d != %d!!\n",i,h_A2[i], h_B2[i], h_C2[i] );
					return 1;
				}

	}

	printf("Wynik poprawny!!\n");
	return 0;
}
