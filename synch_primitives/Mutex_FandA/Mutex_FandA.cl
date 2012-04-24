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
#pragma  OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable


void lock(volatile __global int* lock_turn,volatile __global int* lock_ticket)
{
	int ticket_num=0;
	ticket_num = atom_inc(&lock_ticket[0]);
	
	while(ticket_num!=lock_turn[0])
	{
		
	}

}


void unlock(volatile __global int* lock_turn)
{
	
	atom_inc(&lock_turn[0]);

}    



 // OpenCL Kernel Function for element by element vector addition
__kernel void Mutex(volatile __global int* lock_turn ,volatile __global int* lock_ticket ,volatile __global int* sum )
{
    // get index into global data array
   
	int tid_in_block = get_local_id(0);

	barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

	if(tid_in_block==0)
	{
		// lock the mutex
		lock(&lock_turn[0],&lock_ticket[0]);

		// critical section
		sum[0] = sum[0]+1;

		// unlock the mutex
		unlock(&lock_turn[0]);
	}
   
}


