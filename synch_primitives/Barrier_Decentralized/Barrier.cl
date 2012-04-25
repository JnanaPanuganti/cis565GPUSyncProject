#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable


void gpu_sync(int goalVal,volatile __global int *Arrayin, volatile __global int *Arrayout)
 {
	// thread ID in a block
	//int iGID = get_global_id(0);
	int tid_in_block = get_local_id(0);
	int bid = get_group_id(0);


	// only thread 0 is used for synchronization
	if (tid_in_block == 0)
	{
		Arrayin[bid] = goalVal;
	}



	 if (bid == 0)
	 {
		if (tid_in_block <goalVal )
		{
			while (Arrayin[tid_in_block] != goalVal);
						 
		}
	
		barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
		if (tid_in_block < goalVal)
		{
			Arrayout[tid_in_block] = goalVal;
		}

	 }


	 if (tid_in_block == 0)
	 {
		while (Arrayout[bid] != goalVal);
	 }
	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

}





 // OpenCL Kernel Function for element by element vector addition
__kernel void Barrier(int goal_val,volatile __global int* array_in ,volatile __global int* array_out )
{
    // get index into global data array
 __local float blah[10];
	float a = 12.56;
	float b = 20.56;

	float mean;

	for(int i =0; i<=10000; i++)
	{
		mean = (a+b)/2;

	}
	
	
		
	// Gpu synchronization stuff
	

	gpu_sync(goal_val,array_in ,array_out );
	
   
}


