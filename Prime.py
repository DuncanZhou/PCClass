#!/usr/bin/python
#-*-coding:utf-8-*-
'''@author:duncan'''

import math
from mpi4py import MPI
import time

MAXX = 1000000

# 公共通信变量
comm = MPI.COMM_WORLD
# 当前进程获取当前进程的id
comm_rank = comm.Get_rank()
# 获取整个通信结点的数量
comm_size = comm.Get_size()



def isprime(n):
    '''

    :param n: 对n判断是否是素数
    :return:
    '''
    if(n < 2):
        return 0
    i = 2
    while(i <= math.sqrt(n)):
        if(n % i == 0):
            return 0
        i += 1
    return 1


if __name__ == '__main__':
    # 统计素数个数
    count = 0

    start_time = time.time()
    it = comm_rank * MAXX / comm_size + 1
    while(it <= (comm_rank + 1) * MAXX / comm_size):
        count += isprime(it)
        it += 1

    end_time = time.time()
    print "Duration of process %d is %f" % (comm_rank,end_time - start_time)

    if(comm_size == 1):
        print "%d primes found(single node)" % count

    if(comm_rank == 0):
        # 收集起来
        i = 1
        while(i < comm_size):
            tempcount = comm.recv(source=i)
            count += tempcount
            i += 1
        print "%d primes found(multiple nodes)" % count
    else:
        comm.send(count,dest=0)
