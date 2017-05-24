#!/usr/bin/python
#-*-coding:utf-8-*-
'''@author:duncan'''
import re
import os
import math
import time
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mpi4py import MPI
tweets_path = '%dUsers/' % 100

tags = 10

def getStopWords(path):
    stopwords = set()
    with open(path,"r") as f:
        lines = f.readlines()
    for line in lines:
        stopwords.add(line.replace("\r\n","").rstrip())
    return stopwords
stopwords = getStopWords("/home/duncan/stopwords.txt")
# 公共通信变量
comm = MPI.COMM_WORLD
# 当前进程获取当前进程的id
comm_rank = comm.Get_rank()
# 获取整个通信结点的数量
comm_size = comm.Get_size()

# 计算一段文本中的tags,返回tag列表
def GetTags(text):
    '''

    :param text: 文本
    :return: 返回tag列表
    '''
    # 首先针对推文中去除其他符号
    text = re.sub(r'[@|#][\d|\w|_]+|http[\w|:|.|/|\d]+',"",text)
    wordslist = []
    if text == "" or text == None:
        return []

    # 利用nltk分词
    words = word_tokenize(text)
    for word in words:
        # 去除停用词
        if word not in stopwords:
            if(len(word) > 2 and word.isalpha()):
                wordslist.append(word.lower())
    # 继续对词性进行标注
    try:
        pos = nltk.pos_tag(wordslist)
    except Exception as e:
        pos = []
    if(len(pos) < 1):
        return []
    tags = []
    lemmatizer = WordNetLemmatizer()
    for w in pos:
        word = ""
        # 是动词,做词性还原
        if(w[1][0] == 'V'):
            word = lemmatizer.lemmatize(w[0],'v')
            tags.append(word)
        # 是名词做词性还原
        elif(w[1][0] == "N"):
            word = lemmatizer.lemmatize(w[0])
            tags.append(word)
    i = 0
    multicandidates = []
    while(i < len(pos) - 2):
        phase = ""
        # 动名词 | 动词 + 形容词 +名词
        if (pos[i])[1][0] == 'V' and (pos[i + 1][1][0] == 'N' or (pos[i + 1][1][0] == "J" and pos[i + 2][1][0] == "N")):
            if pos[i + 1][1][0] == 'N':
                suffix = lemmatizer.lemmatize(pos[i + 1][0],'n')
            else:
                suffix = lemmatizer.lemmatize(pos[i + 1][0],'a')
            phase += lemmatizer.lemmatize((pos[i])[0],'v') + " " + suffix
            i = i + 2
            while(i < len(pos) and (pos[i])[1][0] == 'N'):
                phase += " " + lemmatizer.lemmatize((pos[i])[0])
                i += 1
            multicandidates.append(phase)
        # 形容词　+ 名词
        elif(pos[i][1][0] == "J" and pos[i + 1][1][0] == "N"):
            if((i !=0 and pos[i - 1 ][1][0] != "V") or i == 0):
                phase +=lemmatizer.lemmatize((pos[i])[0],"a") + " " + lemmatizer.lemmatize((pos[i + 1])[0])
                i += 2
                while(i < len(pos) and (pos[i])[1][0] == 'N'):
                    phase += " " + lemmatizer.lemmatize((pos[i])[0],"n")
                    i += 1
            multicandidates.append(phase)
        else:
            i += 1
    if len(multicandidates) != 0:
        tags += multicandidates
    return tags


# 计算words列表中词频
def CalcTF(words,number):
    '''
    :param number: 限制返回的tag数量
    :param words: 传入words的list列表
    :return: 返回对应字典形式 {tag:TF},最终返回列表
    '''
    worddic = {}
    # 先转换成集合
    dic = set(words)
    for word in dic:
        TF = words.count(word)
        worddic[word] = TF
    # 将字典按照词频排序,取前10个
    worddic = sorted(worddic.items(),key = lambda val:val[1],reverse=True)
    if(len(worddic) < number):
        # 不足number需要填充
        i = len(worddic)
        while(i < number):
            worddic.append(("null",0))
            i += 1
    return worddic[:number]


if __name__ == '__main__':
    # text = "how do i love you? Beautiful girls like eating cakes. I like dancing and swimming. How about you?he loves me? all we eat cakes."
    # print CalcTF(GetTags(text),10)

    flag = False
    # 获取文件夹中用户总数及名称
    names_list = os.listdir(tweets_path)
    users_number = len(names_list)
    if(comm_rank == 0):
        print "共%d个用户" % users_number
    block_size = math.ceil(users_number * 1.0 / comm_size)
    # print "block_size大小%d" % block_size
    start = i = int(comm_rank * block_size)
    users_tags = []
    start_time = time.time()
    while(i < block_size + start and i < users_number):
        with open(tweets_path + names_list[i],'r') as f:
            text = f.read()
        # 得到某一用户的前n个tags,并有词频
        user_tags = CalcTF(GetTags(text.decode("utf-8")),tags)
        # print user_tags
        users_tags += user_tags
        i += 1
        print "已处理%d个用户" % i

    if(comm_rank == 0):
        i = 1
        while(i < comm_size):
            rev_tags = comm.recv(source=i)
            # 合并起来
            users_tags += rev_tags
            i += 1
            # print users_tags
        # 写入文件
        with open("/home/duncan/tags_tf","w") as f:
            # 使得从第一行开始写
            for tag in users_tags:
                f.write(tag[0])
                f.write("\t")
                f.write(str(tag[1]))
                f.write("\n")
        print "TF值写入完成"
        # 当对文件写入完成时发送广播
        flag = True
        comm.bcast(flag,root=0)
    else:
        # 除了0号进程外,其他进程发送数据,先发送自己的进程id
        comm.send(users_tags,dest=0)

    if(comm_rank != 0):
        flag = comm.bcast(None,root=0)

    # 当flag为True时继续向下计算
    if(flag == True):
        print "process %d 开始计算idf" % comm_rank
        # 对保存好的文件中的TF词频tags继续计算IDF
        with open("/home/duncan/tags_tf","r") as f:
            lines = f.readlines()
        # 每个进程结点需要处理的tags数
        block_size = (int)(math.ceil(users_number * 1.0 / comm_size)) * tags
        start = lineid = comm_rank * block_size
        tags_tfidf = []
        while(lineid < block_size + start and lineid < users_number * tags):
            # 是填充的,则不匹配了,直接加入tfidf列表("null",0)
            if((lines[lineid].split("\t"))[0] == "null"):
                tags_tfidf.append(("null",0))
                lineid += 1
                continue
            # 对lines[i]去计算它的idf,在其他用户tags中搜寻
            user_id = int(math.ceil((lineid + 1) * 1.0 / tags))
            # 所以该用户tags行标在[(user_id - 1) * tags,user_id * tags - 1]
            search_rowid = 0
            count = 1
            while(search_rowid < users_number * tags):
                # 当在其中一个用户的tags中搜寻到后直接跳至下一个用户的tags
                if(search_rowid < (user_id - 1) * tags or search_rowid > user_id * tags - 1):
                    try:
                        if((lines[search_rowid].split("\t"))[0] == (lines[lineid].split("\t"))[0]):
                            count += 1
                            # 跳到下一个用户tags开始处
                            current_user_id = int(math.ceil((search_rowid + 1) * 1.0 / tags))
                            search_rowid = current_user_id * tags
                        else:
                            search_rowid += 1
                    except Exception as e:
                        # 如果行号超出异常
                        print search_rowid,lineid
                        print lines[search_rowid],lines[lineid]
                else:
                    # 在所需要判断的用户的tags范围内,则跳出
                    search_rowid = user_id * tags
            idf = math.log(users_number * 1.0 / count,2)
            # print count,idf
            tags_tfidf.append(((lines[lineid].split("\t"))[0],idf * int((lines[lineid].split("\t"))[1])))
            lineid += 1

        end_time = time.time()
        # 每个结点都处理完自己的用户推文
        print "process %d cost %f" % (comm_rank,end_time - start_time)

        # 该进程计算完其下所有用户的tags的tfidf
        if(comm_rank == 0):
            # 收集tags_tfidf
            i = 1
            while(i < comm_size):
                recv_tags = comm.recv(source = i)
                tags_tfidf += recv_tags
                i += 1
            # 最终结果写入文件
            with open("/home/duncan/tags_tfidf","w") as f:
                for tag in tags_tfidf:
                    f.write(tag[0])
                    f.write("\t")
                    f.write(str(tag[1]))
                    f.write("\n")
        else:
            comm.send(tags_tfidf,dest=0)
