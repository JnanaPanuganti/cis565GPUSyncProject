/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// *********************************************************************
// Barrier Notes:  
//
// A simple OpenCL API demo application that implements 
// element by element vector addition between 2 float arrays. 
//
// Runs computations with OpenCL on the GPU device and then checks results 
// against basic host CPU/C++ computation.
//
// Uses some 'shr' and 'ocl' functions from oclUtils and shrUtils libraries for 
// compactness, but these are NOT required libs for OpenCL developement in general.
// *********************************************************************

// common SDK header for standard utilities and system libs 
#include <oclUtils.h>
#include <shrQATest.h>
#include <stdio.h>

#define NUM_BLOCKS 2 // MAX WORK GROUP
#define NUM_THREADS 128 // NUMBER OF THREADS PER BLOCK 
// Name of the file with the source code for the computation kernel
// *********************************************************************
const char* cSourceFile = "Barrier.cl";
    


// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel


cl_mem array_in;			// GLOBAL MUTEX VALUE
cl_mem array_out;
size_t szGlobalWorkSize;        // 1D var for Total # of work items
size_t szLocalWorkSize;		    // 1D var for # of work items in the work group	
size_t szParmDataBytes;			// Byte size of context information
size_t szKernelLength;			// Byte size of kernel code
cl_int ciErr1, ciErr2;			// Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation
const char* cExecutableName = NULL;
cl_event ceEvent;               // OpenCL event
// demo config vars
int iNumElements = NUM_BLOCKS* NUM_THREADS;	// total num of threads
// BARRIER GOAL
int goal_val = NUM_BLOCKS;
int original_goal = 0;

int* input;

shrBOOL bNoPrompt = shrFALSE;  

// Forward Declarations
// *********************************************************************

void Cleanup (int argc, char **argv, int iExitCode);

// Main function 
// *********************************************************************
int main(int argc, char **argv)
{
    //shrQAStart(argc, argv);


	// get command line arg for quick test, if provided
    bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
    
    // start logs 
	cExecutableName = argv[0];
    shrSetLogFileName ("Barrier.txt");
    printf("%s Starting...\n\n# of THREADS \t= %i\n", argv[0], iNumElements); 

    // set and log Global and Local work size dimensions
    szLocalWorkSize = NUM_THREADS ;
    szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, iNumElements);  // rounded up to the nearest multiple of the LocalWorkSize
    printf("Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n", 
           szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 

    

    //Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);

    printf("clGetPlatformID...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    //Get the devices
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
    printf("clGetDeviceIDs...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    //Create the context
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
    printf("clCreateContext...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, CL_QUEUE_PROFILING_ENABLE, &ciErr1);
    printf("clCreateCommandQueue...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }


	

    // Read the OpenCL kernel in from source file
    printf("oclLoadProgSource (%s)...\n", cSourceFile); 
    cPathAndName = shrFindFilePath(cSourceFile, argv[0]);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);

    // Create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErr1);
    printf("clCreateProgramWithSource...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Build the program with 'mad' Optimization option
    #ifdef MAC
        char* flags = "-cl-fast-relaxed-math -DMAC";
    #else
        char* flags = "-cl-fast-relaxed-math";
    #endif
    ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    printf("clBuildProgram...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Create the kernel
    ckKernel = clCreateKernel(cpProgram, "Barrier", &ciErr1);
    printf("clCreateKernel (Barrier)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }



	 // Allocate and initialize host arrays 
    printf( "Allocate and Init Host Mem...\n"); 
    input = (int *)malloc(sizeof(int) * NUM_BLOCKS);

	for(int i =0; i<=NUM_BLOCKS; i++)
	{
		input[i]=0;

	}

	// Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    array_in = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int)* NUM_BLOCKS, NULL, &ciErr1);
    array_out = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int)* NUM_BLOCKS, NULL, &ciErr1);
	
	if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }


    // Set the Argument values
    
    ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_int), (void*)&goal_val);
	ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&array_in);
	ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&array_out);

   // ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_int), (void*)&iNumElements);
    printf("clSetKernelArg 0 - 2...\n\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }






    // --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back



	ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, array_in, CL_FALSE, 0, sizeof(int) * NUM_BLOCKS,(void*) input, 0, NULL, NULL);
    
    printf("clEnqueueWriteBuffer (SrcA and SrcB)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }


    // Launch kernel
    ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, &ceEvent);
    printf("clEnqueueNDRangeKernel (Barrier)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

   /*ciErr1 = clEnqueueReadBuffer(cqCommandQueue, global_mutex, CL_TRUE, 0, sizeof(cl_int), &original_goal, 0, NULL, NULL);
    printf("clEnqueueReadBuffer (Dst)...%d \n\n", original_goal); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }*/


	//GPU_PROFILING
    ciErr1=clWaitForEvents(1, &ceEvent);
	if (ciErr1 != CL_SUCCESS)
    {
        printf("Error 1 !\n\n");
        Cleanup(argc, argv, EXIT_FAILURE);
    }
       
        cl_ulong start, end;
     ciErr1 =   clGetEventProfilingInfo(ceEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
      ciErr1 |= clGetEventProfilingInfo(ceEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

	if (ciErr1 != CL_SUCCESS)
    {
        printf("Error 2 !\n\n");
        Cleanup(argc, argv, EXIT_FAILURE);
    }
        double dSeconds = 1.0e-9 * (double)(end - start);
		printf("Done! time taken %llu \n",end - start );
      // printf("Done! Kernel execution time: %.5f s\n\n", dSeconds);


		// Release event
       clReleaseEvent(ceEvent);
       ceEvent = 0;

    Cleanup (argc, argv,  EXIT_SUCCESS);
}

void Cleanup (int argc, char **argv, int iExitCode)
{
    // Cleanup allocated objects
    printf("Starting Cleanup...\n\n");
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
	if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(array_in)clReleaseMemObject(array_in);
	 if(array_out)clReleaseMemObject(array_out);

	// free(input);
    // finalize logs and leave
    shrQAFinishExit(argc, (const char **)argv, (iExitCode == EXIT_SUCCESS) ? QA_PASSED : QA_FAILED);
}


