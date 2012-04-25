
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

 void group_barrier(volatile __global int* array_in, int goal_val)
{

	int iLID = get_local_id(0);
	int iBLD = get_group_id(0);

	
	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

	if(iLID==0)
	{
		//update the global value
		
		atom_inc(&array_in[0]);
		
		
		
			while(array_in[0]!=goal_val)
			{
			
			}
		
		
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
	group_barrier(&array_in[0],goal_val);


   
}


