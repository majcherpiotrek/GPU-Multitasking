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


#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization

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
    int gDim
)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;
	if(threadIdx.x < blkDim)
	{
		const int opt = blkDim * mapBlk[blockIdx.x] + threadIdx.x;

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

}
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

__device__ void dummy(int n){
	for(int i = 0; i < n; i++);
}

__global__ void
scheduler(float2 * __restrict d_CallResult,
	    float2 * __restrict d_PutResult,
	    float2 * __restrict d_StockPrice,
	    float2 * __restrict d_OptionStrike,
	    float2 * __restrict d_OptionYears,
	    float Riskfree,
	    float Volatility,
	    int optN, int dummyLoopIterations, int* mapBlk, int* mapKernel, int gridDim_A, int blockDim_A){

	if(mapKernel[blockIdx.x] == 1)
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
		            blockDim_A,
		            gridDim_A
		        );
	else
		dummy(dummyLoopIterations);


}
////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ////////// MAPPING KERNELS /////////////////
    int dummyIterations = 170;

    /*Ilość bloków wątków kernela vectorAdd1*/
    const int TBi = DIV_UP((OPT_N/2), 128);
    const int M = 15;

    int tests = 6;

    int m[6] = {4, 6, 8, 10, 12, 14};
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

    for(int j = 0; j < tests; j++)
    {
    	 // Start logs
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
    	    for ( k = 0; k < OPT_N; k++)
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

    	int di = dummies[j];
    	int mod = mods[j];
    	int blocks = TBi+di;;
    	int threads = 128;
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
    	int Ti = 128;
    	int* d_mapKernel;
    	checkCudaErrors(cudaMalloc((void**)&d_mapKernel, blocks*sizeof(int)));
    	checkCudaErrors(cudaMemcpy(d_mapKernel, mapKernel, blocks*sizeof(int), cudaMemcpyHostToDevice));
    	int* d_mapBlk;
    	checkCudaErrors(cudaMalloc((void**)&d_mapBlk, blocks*sizeof(int)));
    	checkCudaErrors(cudaMemcpy(d_mapBlk, mapBlk, blocks*sizeof(int), cudaMemcpyHostToDevice));

    	scheduler<<<blocks, threads>>>((float2 *)d_CallResult,
                (float2 *)d_PutResult,
                (float2 *)d_StockPrice,
                (float2 *)d_OptionStrike,
                (float2 *)d_OptionYears,
                RISKFREE,
                VOLATILITY,
                OPT_N,
                dummyIterations, d_mapBlk, d_mapKernel, TBi, Ti);

    	getLastCudaError("BlackScholesGPU() execution failed\n");
    	checkCudaErrors(cudaDeviceSynchronize());
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
    	    //Calcdummy(dummyLoopIterations);ulate options values on CPU
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
    }



    exit(EXIT_SUCCESS);
}
