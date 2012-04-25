#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable


void lock(volatile __global int* lock_mutex)
{
	int old_val=1;
	while(old_val)
	{
		old_val = atom_xchg(&lock_mutex[0],1);
		if (old_val==0)
		{
			break;
		}
	}
}


void unlock(volatile __global int* lock_mutex)
{
	
	atom_xchg(&lock_mutex[0],0);

}    



 // OpenCL Kernel Function for element by element vector addition
__kernel void Mutex(volatile __global int* lock_mutex ,volatile __global int* sum )
{
    // get index into global data array
   int tid_in_block = get_local_id(0);

	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

	if(tid_in_block==0)
	{
		// lock the mutex
		lock(&lock_mutex[0]);

		// critical section
		sum[0]=sum[0]+1;

		// unlock the mutex
		unlock(&lock_mutex[0]);
	}
   
}


