#!/usr/bin/python
#-*-coding:utf-8-*-
import os, sys, time
import numpy as np
import mpi4py.MPI as MPI

# 公共通信变量
comm = MPI.COMM_WORLD
# 当前进程获取当前进程的id
comm_rank = comm.Get_rank()
# 获取整个通信结点的数量
comm_size = comm.Get_size()

# test MPI
if __name__ == '__main__':

    #create a matrix
    if comm_rank == 0:
        # 0进程生成数据
        all_data = np.arange(1000000).reshape(1000, 1000)
        # print '************ data ******************'
        # print all_data

    #广播
    all_data = comm.bcast(all_data if comm_rank == 0 else None, root = 0)

    # 起始时间
    start_time = time.time()

    #divide the data to each processor
    num_samples = all_data.shape[0]
    local_data_offset = np.linspace(0, num_samples, comm_size + 1).astype('int')

    #get the local data which will be processed in this processor
    local_data = all_data[local_data_offset[comm_rank] :local_data_offset[comm_rank + 1]]
    # print '****** %d/%d processor gets local data ****'  %(comm_rank, comm_size)
    # print local_data

    #reduce to get sum of elements
    local_sum = local_data.sum()

    # 结束时间
    end_time = time.time()
    print "process %d use %f" % (comm_rank,end_time -start_time)

    all_sum = comm.reduce(local_sum, root = 0, op = MPI.SUM)

    #process in local
    local_result = local_data ** 2



    #gather the result from all processors and broadcast it
    result = comm.allgather(local_result)
    result = np.vstack(result)



    if comm_rank == 0:
        # print '*** sum: , all_sum'
        # print '************ result ******************'
        # print result
        # print "cost time %f" % (end_time - start_time)
        pass