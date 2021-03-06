#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#define I_MIN 1
#define I_MAX 1

void lock(volatile __global int* lock_mutex, volatile __global int* sleep_for)
{
	int old_val=1;
	int prev_sleep_time=0;
	int bID = get_group_id(0);
	sleep_for[bID] = I_MIN;
	while(old_val)
	{
		old_val = atom_xchg(&lock_mutex[0],1);
		if (old_val==0)
		{
			break; //got the lock
		}
		else
		{
			//backoff --- this is more like a delay before trying for the lock again. Not like actual CPU sleep
				prev_sleep_time = sleep_for[bID];
				while(sleep_for[bID]--);
				if(prev_sleep_time<I_MAX)
				{
					sleep_for[bID] = prev_sleep_time+1;
				}
				else
				{
					sleep_for[bID] = I_MIN;
				}		
		}//else

	}//while

}// end of lock


void unlock(volatile __global int* lock_mutex)
{
	int val=0;
	val=atom_xchg(&lock_mutex[0],0);

}    



 // OpenCL Kernel Function for element by element vector addition
__kernel void Mutex(volatile __global int* sleep_for, volatile __global int* lock_mutex ,volatile __global int* sum )
{
    // get index into global data array
   
	int tid_in_block = get_local_id(0);

	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

	if(tid_in_block==0)
	{
		// lock the mutex
		lock(&lock_mutex[0],sleep_for);

		// critical section
		sum[0]=sum[0]+1;

		// unlock the mutex
		unlock(&lock_mutex[0]);
	}
   
}


