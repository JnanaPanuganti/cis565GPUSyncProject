To run the code

Download the all the folders and open the Implementations in visual studio.

Each folder namely , Barrier_centralized, Barrier_Dencteralized , Mutex, Mutex_backoff and Mutex_FandA have the implementations of the algorithims in .cl files

Each of the projects run mulplie invications of the kernel with block size varying between 10 to 120 in steps of 10.

Some of the exsisting NVDIA APIS have been used in the project, specifically ShrLogs(). This function writes the results to respective .txt files.

The results include debug information as well as number of blocks and therads and the time taken to execute the kernal.


**Please ignore the Barrier folder**