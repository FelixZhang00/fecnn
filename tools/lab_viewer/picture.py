# -*- coding: utf-8 -*-
''' config:
file_list:the different models tained result output in shell,and save as files.
dirname:the dir of file_list.
'''
import numpy as np
import matplotlib.pyplot as plt

dirname = 'lab/fecnn_lab/网络结构'
Iteration = []
loss = []
Eval = []
file_list = ['CNN-1.txt','CNN-2.txt','CNN-3.txt','CNN-4.txt']
filenamelist = []
namelist = []

#file_test = file('/home/lilonglong/workspace/python workspace/python workspace/python_picture/test.txt','rb')
def FileList(filelist):
    for f in filelist:
        filename = dirname+'/'+f
        filenamelist.append(filename)
        
def readData(file_test):
    j = 0
    test1 = []
    test2 = []
    test3 = []
    for fil in file_test:
        fileNum = file(fil,'rb')
        for line in fileNum.read().split('\n'):
            if line.split(' ')[0] == 'Iteration' or line.split(' ')[0] == 'Epoch':
                for n in xrange(0,len(line.split(' '))):
                    if line.split(' ')[n] == 'Iteration':
                        test1.append(int(line.split(' ')[n+1]))
                        if line.split(' ')[n] not in namelist:
                            namelist.append(line.split(' ')[n])
                    elif line.split(' ')[n] == 'loss':
                        test2.append([line.split(' ')[n+2],line.split(' ')[n+4]])
                        if line.split(' ')[n] not in namelist:
                            namelist.append(line.split(' ')[n])
                    elif line.split(' ')[n] == 'eval':
                        test3.append(float(line.split(' ')[n+2]))
                        if line.split(' ')[n] not in namelist:
                            namelist.append(line.split(' ')[n])
        for li in test2:
            test2[j] = float(li[0]) * float(li[1])
            j += 1   
        test1.pop(0)
        test2.pop(0)
        test3.pop(0)
        Iteration.append(test1)
        loss.append(test2)
        Eval.append(test3)  
        test1 = []
        test2 = []
        test3 = []
        j = 0
        fileNum.close()
def draw(x,y,name):  
    judge = ['s-','o-','x-','^-','*-','.--','o--','x--','+--','*--']
    i = 0
    for num in y:
        plt.plot(x[0],num,judge[i],label = file_list[i])
        i += 1
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel(name)
    plt.show()    
if __name__ == '__main__':
    FileList(file_list)
    readData(filenamelist)
    #draw(Iteration,loss,'loss')
    #draw(Iteration,Eval,'eval')
    plt.figure(2) # 创建图表2
    ax1 = plt.subplot(211) # 在图表2中创建子图1
    ax2 = plt.subplot(212) # 在图表2中创建子图2
    judge = ['s-','o-','x-','^-','*-','.--','o--','x--','+--','*--']
    i = 0
    plt.sca(ax1)
    for num in loss:
        plt.plot(Iteration[0],num,judge[i],label = file_list[i].split('.txt')[0])
        i += 1
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    
    i = 0
    plt.sca(ax2)
    plt.figure(2)
    for num in Eval:
        plt.plot(Iteration[0],num,judge[i],label = file_list[i].split('.txt')[0])
        i += 1
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('eval')
    plt.show() 