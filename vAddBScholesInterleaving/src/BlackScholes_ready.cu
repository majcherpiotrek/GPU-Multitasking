/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */

#include <math.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization
////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
    K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = __fdividef(1.0F, rsqrtf(T));
    d1 = __fdividef(__logf(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    //Calculate Call and Put simultaneously
    expRT = __expf(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////

// Dodane parametry ze schedulera
__device__ void BlackScholesGPU(
    float2 * __restrict d_CallResult,
    float2 * __restrict d_PutResult,
    float2 * __restrict d_StockPrice,
    float2 * __restrict d_OptionStrike,
    float2 * __restrict d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
    int* mapBlk,
    int blkDim,
    int gridDim
)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;
	int bid = mapBlk[blockIdx.x];

	if(bid < gridDim && threadIdx.x < blkDim){


    const int opt = blkDim * bid + threadIdx.x;

     // Calculating 2 options per thread to increase ILP (instruction level parallelism)
    if (opt < (optN / 2))
    {
        float callResult1, callResult2;
        float putResult1, putResult2;
        BlackScholesBodyGPU(
            callResult1,
            putResult1,
            d_StockPrice[opt].x,
            d_OptionStrike[opt].x,
            d_OptionYears[opt].x,
            Riskfree,
            Volatility
        );
        BlackScholesBodyGPU(
            callResult2,
            putResult2,
            d_StockPrice[opt].y,
            d_OptionStrike[opt].y,
            d_OptionYears[opt].y,
            Riskfree,
            Volatility
        );
        d_CallResult[opt] = make_float2(callResult1, callResult2);
        d_PutResult[opt] = make_float2(putResult1, putResult2);
	 }
	}
	__syncthreads();
}


////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 4000000;
const int  NUM_ITERATIONS = 512;


const int          OPT_SZ = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

////////////////////////////////////////////////////////////////////////////////
// Inicjalizacja danych do vectorAdd
////////////////////////////////////////////////////////////////////////////////
void initVectors(int* h_A, int* h_B, int* d_A, int* d_B, int* d_C, int size, int val_range){
return;


}
////////////////////////////////////////////////////////////////////////////////
// vectorAdd
///////////////////////////////////////////////////////////////////////////////
__device__ void vectorAdd(int* d_A, int* d_B, int* d_C, int size, int* mapBlk, int blkDim, int gridDim){

	int bid = mapBlk[blockIdx.x];
	if(threadIdx.x < blkDim){

	int vId = blkDim*bid + threadIdx.x;

	if(vId < size)
		d_C[vId] = d_A[vId] + d_B[vId];
	}
	//__syncthreads();
}
////////////////////////////////////////////////////////////////////////////////
// SCHEDULER KERNEL
///////////////////////////////////////////////////////////////////////////////

__global__ void scheduler(//Parametry BlackScholes
	float2 * __restrict d_CallResult,
	float2 * __restrict d_PutResult,
    float2 * __restrict d_StockPrice,
    float2 * __restrict d_OptionStrike,
    float2 * __restrict d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN,
    int gridDim_BS,
    int blkDim_BS,
    //Parametry vectorAdd
    int* A, int* B, int* C, int size,
    int gridDim_vA, int blkDim_vA,
    //Parametry wywołania
    int* mapBlk,int* mapKernel ){

	int bid = blockIdx.x;

	if(mapKernel[bid] == 0)
		vectorAdd(A, B, C, size, mapBlk, blkDim_vA, gridDim_vA);
	else
		BlackScholesGPU(
		            (float2 *)d_CallResult,
		            (float2 *)d_PutResult,
		            (float2 *)d_StockPrice,
		            (float2 *)d_OptionStrike,
		            (float2 *)d_OptionYears,
		            Riskfree,
		            Volatility,
		            optN,
		            mapBlk,
		            blkDim_BS,
		            gridDim_BS
		        );
	__syncthreads();

}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{


	///////////////////////////////////////////////////////
	/******************Nasz kod programu******************/
	///////////////////////////////////////////////////////

    int numElements = 1000000;
    int range  = 1000;


	/********POCZĄTEK WAŻNEGO KODU***********/
    /*Ilość bloków wątków kernela vectorAdd1*/
	 const int TB_BS = DIV_UP((OPT_N/2), 128);
	 const int blkDim_BS = 128;
	 const int M = 15;

	 int tests = 6;

	 /*m[] -> liczba SM'ów, które chcemy zaalokować dla kernela*/
	 int m[8] = {4, 6, 8, 10, 12, 14};

	 /* TV_vA[] - liczby bloków wątków dla vectorAdd*/
	 int* TB_vA = new int[tests];

	 /*mods[] - liczby, przez które dzielimy
	  *modulo w naszym algorytmi LeakyBucket*/
	 int* mods = new int[tests];

	  /*Pętla, w której wyliczamy liczby bloków wątków, które zostaną przydzielone dla vectorAdd.
	   *Liczymy to z wzoru podanego w artykule:
	   * TB_vA[i] = ceil( (TB_BS*(M-m[i])) / m[i] );
	   *Na podstawie ilości SM'ów jakie chcemy przydzielić dla kernela k1 oraz całkowitej liczby
	   *SM'ów w procesorze i liczby bloków wątków, z jaką musi być uruchomiony k1, można z niego
	   *policzyć liczbę bloków wątków dla k2, taką, aby zostały zajęte wszystkie SM'y*/
	 for(int i = 0; i < tests; i++)
	 {
		 	/*Obliczenie liczby bloków wątków dla vectorAdd*/
	   		TB_vA[i] = ceil((TB_BS*(M-m[i]))/m[i]);

	   		/*Jeśli bloków wątków vectorAdd będzie więcej niż BlackScholes to chcemy się dowiedzieć,
	   		 *co który blok wątków będziemy mapowali do BlackScholes. Dlatego liczymy stosunek sumy
	   		 *bloków wątków obu kerneli przez liczbę bloków wątków BlackScholes.
	   		 *
	   		 *np.
	   		 *	TB_vA = 2000;
	   		 *	TB_BS = 1000;
	   		 *	mods = (2000+1000)/1000 = 3 -> czyli co trzeci blok wątków dla BlackScholes.
	   		 */
	   		if(TB_vA[i] >= TB_BS)
	   			mods[i] = floor((TB_BS + TB_vA[i])/TB_BS);
	   		else
	   			mods[i] = 2;
	 }

	 	/*Pętla przechodząca przez wszystkie względne alokacje SM'ów dla vectorAdd i BlackScholes*/
    	for(int t = 0; t < tests; t++)
    	{
    		/*Lokalne zmienne na liczbę bloków wątków vA oraz dzielną modulo*/
    		int TB_vAdd = TB_vA[t];
    		int mod = mods[t];

    		/*Liczę liczbę wątków na blok dla vectorAdd. W sumie jak pomnożymy blkDim_vA * TB_vAdd to musimy
    		 * otrzymać liczbę >= numElements (długości wektora). W przeciwnym wypadku nie będziemy mogli dodać
    		 * dwóch wektorów, ponieważ będzie za mało wątków. Dlatego jeśli liczba elementów wektora dzieli się
    		 * przez liczbę bloków wątków, to przypisujemy blkDim_vA ten iloraz. Jeśli nie, to dodajemy do ilorazu
    		 * jeden, ponieważ wtedy iloczyn blkDim_vA * TB_vAdd będzie na pewno większy od numElements
    		 * */
    		int blkDim_vA = (numElements % TB_vAdd == 0) ? (numElements/TB_vAdd) : (numElements/TB_vAdd+1);

    		/*Liczba bloków wątków oraz wątków na blok, z jakimi wywołany będzie scheduler.
    		 * blocks - suma bloków wątków, ponieważ wszystkie bloki wątków obu kerneli muszą się wykonać
    		 * threads - max(blkDim_vA, blkDim_BS), ponieważ wątków w bloku może być więcej niż potrzeba,
    		 * nie może natomiast być ich mniej
    		 */
    		int blocks = TB_BS+TB_vAdd;
    		int threads = (blkDim_vA > blkDim_BS) ? blkDim_vA : blkDim_BS;

    		/*Tablica mapująca, które bloki wątków schedulera mają być przeznaczone, dla którego kernela*/
    		int* mapKernel = new int[blocks];
    		/*Tablica mapująca numer bloku wątku schedulera, na numer bloku wątku danego kernela
    		 * np.
    		 * 		mapKernel = [A,A,B,A,A,B,A,A,B]
    		 * 		mapBlk = [ (A)0, (A)1, (B)0, (A)2, (A)3, (B)1, (A)4, (A)5, (B)2 ]
    		 */
    		int* mapBlk = new int[blocks];

    		/*Identyfikatory kerneli do schedulera*/
    		int vector = 0; // identyfikator vectorAdd
    		int blackscholes = 1; //identyfikator BlackScholes

    		/*W zależności od tego, który kernel ma mniej bloków wątków,
    		 * jego identyfikator zostanie przypisany do id1 lub id2.
    		 */
    		int id1; //kernel o mniejszej ilości bloków wątków
    		int id2;

    		int minTB;
    		if(TB_vAdd < TB_BS)
    		{
    			minTB = TB_vAdd;
    			id1 = vector;
    			id2 = blackscholes;
    		}
    		else
    		{
    			minTB = TB_BS;
    			id1 = blackscholes;
    			id2 = vector;
    		}

    	/*Pętla realizująca nasz pseudo Leaky-Bucket.
    	 * Na podstawie wcześniej wyliczonych danych
    	 * tworzy tablice mapujące.
    	 */
    	for(int blkA =0, blkB = 0, i = 0; i < blocks; i++)
    	{	//blkA - liczba wpisanych już bloków wątków kernela "mniejszego", blkB - "większego"

    		/*Co "mod" kernel wpisujemy do mapKernel kernela o mniejszej ilości bloków wątków*/
    		if(i%mod == 0)
    		{
    			/*Dopóki nie wpisaliśmy już wszystkich bloków wątków z puli*/
    			if(blkA < minTB)
    			{
    				mapKernel[i] = id1;
    				mapBlk[i] =blkA++;
    			}
    			/*Jeśli już wszystki wpisane do tabeli to wpisujemy bloki drugiego kernela ("większego")*/
    			else
    			{
    				mapKernel[i] = id2;
    				mapBlk[i] = blkB++;
    			}
    		}
    		/*Bloki o numerze niepodzielnym przez mod przypisujemy dla "większego" kernela*/
    		else
    		{
    			mapKernel[i] = id2;
    			mapBlk[i] = blkB++;
    		}

    	}

    	/*Zalokowanie pamięci urządzenia i skopiowanie tablic mapujących*/
    	int* d_mapKernel;
    	checkCudaErrors(cudaMalloc((void**)&d_mapKernel, blocks*sizeof(int)));
    	checkCudaErrors(cudaMemcpy(d_mapKernel, mapKernel, blocks*sizeof(int), cudaMemcpyHostToDevice));
    	int* d_mapBlk;
    	checkCudaErrors(cudaMalloc((void**)&d_mapBlk, blocks*sizeof(int)));
    	checkCudaErrors(cudaMemcpy(d_mapBlk, mapBlk, blocks*sizeof(int), cudaMemcpyHostToDevice));

    	/*Inicjalizacja danych dla vectorAdd*/
    	    int* h_A = NULL;
    	    int* h_B = NULL;
    	    int* h_C = NULL;

    	    int* d_A = NULL;
    	    int* d_B = NULL;
    	    int* d_C = NULL;

    	    h_A = (int*)malloc(numElements*sizeof(int));
    	    h_B = (int*)malloc(numElements*sizeof(int));
    	    h_C = (int*)malloc(numElements*sizeof(int));

    	    srand( time(NULL) );
    	    for(int k=0; k<numElements; k++){
    	        h_A[k] =(int)rand()%range;
    	        h_B[k] =(int)rand()%range;
    	    }

    	    cudaError_t err = cudaSuccess;

    	    /*Alokowanie pamięci urządzenia*/
    	    err = cudaMalloc((void**)&d_A, numElements*sizeof(int));

    	    	if(err != cudaSuccess){
    		      printf("vectorAdd - blad podczas alokowania pamieci urzadzenia (%s)", cudaGetErrorString(err));
    		      exit(EXIT_FAILURE);
    	        }
    		   err = cudaMalloc((void**)&d_B, numElements*sizeof(int));
    		   if(err != cudaSuccess){
    			   printf("vectorAdd - blad podczas alokowania pamieci urzadzenia (%s)", cudaGetErrorString(err));
    			   exit(EXIT_FAILURE);
    			}
    			err = cudaMalloc((void**)&d_C, numElements*sizeof(int));
    			if(err != cudaSuccess){
    				printf("vectorAdd - blad podczas alokowania pamieci urzadzenia (%s)", cudaGetErrorString(err));
    				exit(EXIT_FAILURE);
    			}

    		   /*Kopiowanie danych do urządzenia*/
    		   err = cudaMemcpy(d_A, h_A, numElements*sizeof(int), cudaMemcpyHostToDevice );
    			if(err != cudaSuccess){
    					printf("vectorAdd - blad podczas kopiowania danych do urzadzenia (%s)", cudaGetErrorString(err));
    					exit(EXIT_FAILURE);
    			}
    			err = cudaMemcpy(d_B, h_B, numElements*sizeof(int), cudaMemcpyHostToDevice );
    			if(err != cudaSuccess){
    					printf("vectorAdd - blad podczas kopiowania danych do urzadzenia (%s)", cudaGetErrorString(err));
    					exit(EXIT_FAILURE);
    			}
    								/************************************************************************/
    				    		    /*****************KOD ORYGINALNEGO KERNELA BLACKSCHOLES******************/
    				    		    /************************************************************************/
    				    		    printf("[%s] - Starting...\n", argv[0]);

    				    		    //'h_' prefix - CPU (host) memory space
    				    		    float
    				    		    //Results calculated by CPU for reference
    				    		    *h_CallResultCPU,
    				    		    *h_PutResultCPU,
    				    		    //CPU copy of GPU results
    				    		    *h_CallResultGPU,
    				    		    *h_PutResultGPU,
    				    		    //CPU instance of input data
    				    		    *h_StockPrice,
    				    		    *h_OptionStrike,
    				    		    *h_OptionYears;

    				    		    //'d_' prefix - GPU (device) memory space
    				    		    float
    				    		    //Results calculated by GPU
    				    		    *d_CallResult,
    				    		    *d_PutResult,
    				    		    //GPU instance of input data
    				    		    *d_StockPrice,
    				    		    *d_OptionStrike,
    				    		    *d_OptionYears;

    				    		    double
    				    		    delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;

    				    		    StopWatchInterface *hTimer = NULL;
    				    		    int k;

    				    		    findCudaDevice(argc, (const char **)argv);

    				    		    sdkCreateTimer(&hTimer);

    				    		    printf("Initializing data...\n");
    				    		    printf("...allocating CPU memory for options.\n");
    				    		    h_CallResultCPU = (float *)malloc(OPT_SZ);
    				    		    h_PutResultCPU  = (float *)malloc(OPT_SZ);
    				    		    h_CallResultGPU = (float *)malloc(OPT_SZ);
    				    		    h_PutResultGPU  = (float *)malloc(OPT_SZ);
    				    		    h_StockPrice    = (float *)malloc(OPT_SZ);
    				    		    h_OptionStrike  = (float *)malloc(OPT_SZ);
    				    		    h_OptionYears   = (float *)malloc(OPT_SZ);

    				    		    printf("...allocating GPU memory for options.\n");
    				    		    checkCudaErrors(cudaMalloc((void **)&d_CallResult,   OPT_SZ));
    				    		    checkCudaErrors(cudaMalloc((void **)&d_PutResult,    OPT_SZ));
    				    		    checkCudaErrors(cudaMalloc((void **)&d_StockPrice,   OPT_SZ));
    				    		    checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
    				    		    checkCudaErrors(cudaMalloc((void **)&d_OptionYears,  OPT_SZ));

    				    		    printf("...generating input data in CPU mem.\n");
    				    		    srand(5347);

    				    		    //Generate options set
    				    		    for (k = 0; k < OPT_N; k++)
    				    		    {
    				    		        h_CallResultCPU[k] = 0.0f;
    				    		        h_PutResultCPU[k]  = -1.0f;
    				    		        h_StockPrice[k]    = RandFloat(5.0f, 30.0f);
    				    		        h_OptionStrike[k]  = RandFloat(1.0f, 100.0f);
    				    		        h_OptionYears[k]   = RandFloat(0.25f, 10.0f);
    				    		    }

    				    		    printf("...copying input data to GPU mem.\n");
    				    		    //Copy options data to GPU memory for further processing
    				    		    checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    				    		    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    				    		    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));
    				    		    printf("Data init done.\n\n");


    				    		    printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    				    		    checkCudaErrors(cudaDeviceSynchronize());
    				    		    sdkResetTimer(&hTimer);
    				    		    sdkStartTimer(&hTimer);

    	/*URUCHOMIENIE KERNELA SCHEDULERA*/
    	scheduler<<<blocks, threads>>>(
    				/*Oryginalne parametry BlackScholes*/
    	            (float2 *)d_CallResult,
    	            (float2 *)d_PutResult,
    	            (float2 *)d_StockPrice,
    	            (float2 *)d_OptionStrike,
    	            (float2 *)d_OptionYears,
    	            RISKFREE,
    	            VOLATILITY,
    	            OPT_N,
    	            /*****/
    	            TB_BS, blkDim_BS, //liczba bloków wątków i wątków na blok BlackScholes
    	            /*Parametry vectorAdd*/
    	            d_A, d_B, d_C, numElements,
    	            /*************************/
    	            TB_vAdd, blkDim_vA, //liczba bloków wątków i wątków na blok vectorAdd
    	            /*Tablice mapujące*/
    	            d_mapBlk, d_mapKernel
    	 );

    	getLastCudaError("BlackScholesGPU() execution failed\n");


    	/*Skopiowanie wyników vectorAdd do hosta*/
		err = cudaSuccess;

		err = cudaMemcpy(h_C, d_C, numElements*sizeof(int), cudaMemcpyDeviceToHost);
		if(err != cudaSuccess){
			printf("Blad przy kopiowaniu danych z urzadzenia do hosta (%s)", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		/*Sprawdzenie wyników vectorAdd*/

		for(int i = 0; i < numElements; i++)
		{
			int w = h_A[i]+h_B[i];

			if( w != h_C[i])
			{
				printf("nr %d Wynik 1 niepoprawny %d + %d != %d!!\n",i, h_A[i], h_B[i], h_C[i] );
				return 1;
			}

		}

		printf("Wynik vectorAdd poprawny!!\n");

		/************************************************************************/
		/*****************KOD ORYGINALNEGO KERNELA BLACKSCHOLES******************/
		/************************************************************************/
		 sdkStopTimer(&hTimer);
		 gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;

		 //Both call and put is calculated
		 printf("Options count             : %i     \n", 2 * OPT_N);
		 printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
		 printf("Effective memory bandwidth: %f GB/s\n", ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
		 printf("Gigaoptions per second    : %f     \n\n", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

		 printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u options, NumDevsUsed = %u, Workgroup = %u\n",
				(((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * OPT_N), 1, 128);

		 printf("\nReading back GPU results...\n");
		 //Read back GPU results to compare them to CPU results
		 checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
		 checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));


		 printf("Checking the results...\n");
		 printf("...running CPU calculations.\n\n");
		 //Calculate options values on CPU
		 BlackScholesCPU(
			 h_CallResultCPU,
			 h_PutResultCPU,
			 h_StockPrice,
			 h_OptionStrike,
			 h_OptionYears,
			 RISKFREE,
			 VOLATILITY,
			 OPT_N
		 );

		 printf("Comparing the results...\n");
		 //Calculate max absolute difference and L1 distance
		 //between CPU and GPU results
		 sum_delta = 0;
		 sum_ref   = 0;
		 max_delta = 0;

		 for (int i = 0; i < OPT_N; i++)
		 {
			 ref   = h_CallResultCPU[i];
			 delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

			 if (delta > max_delta)
			 {
				 max_delta = delta;
			 }

			 sum_delta += delta;
			 sum_ref   += fabs(ref);
		 }

		 L1norm = sum_delta / sum_ref;
		 printf("L1 norm: %E\n", L1norm);
		 printf("Max absolute error: %E\n\n", max_delta);

		 printf("Shutting down...\n");
		 printf("...releasing GPU memory.\n");
		 checkCudaErrors(cudaFree(d_OptionYears));
		 checkCudaErrors(cudaFree(d_OptionStrike));
		 checkCudaErrors(cudaFree(d_StockPrice));
		 checkCudaErrors(cudaFree(d_PutResult));
		 checkCudaErrors(cudaFree(d_CallResult));

		 printf("...releasing CPU memory.\n");
		 free(h_OptionYears);
		 free(h_OptionStrike);
		 free(h_StockPrice);
		 free(h_PutResultGPU);
		 free(h_CallResultGPU);
		 free(h_PutResultCPU);
		 free(h_CallResultCPU);
		 sdkDeleteTimer(&hTimer);
		 printf("Shutdown done.\n");

		 printf("\n[BlackScholes] - Test Summary\n");

		 // cudaDeviceReset causes the driver to clean up all state. While
		 // not mandatory in normal operation, it is good practice.  It is also
		 // needed to ensure correct operation when the application is being
		 // profiled. Calling cudaDeviceReset causes all profile data to be
		 // flushed before the application exits
		 //cudaDeviceReset();

		 if (L1norm > 1e-6)
		 {
			 printf("Test failed!\n");
			 exit(EXIT_FAILURE);
		 }

		 printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");
		 printf("Test passed\n");



    	     delete[] mapKernel;
    	     delete[] mapBlk;
    	     free(h_A);
    	     free(h_B);
    	     free(h_C);

    	}

    	delete[] TB_vA;
    	delete[] mods;

    exit(EXIT_SUCCESS);
}
