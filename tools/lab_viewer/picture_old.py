# -*- coding: utf-8 -*-
"""
Created on Sun May 15 12:17:09 2016

@author: lilonglong
"""
import matplotlib.pyplot as plt

dirname = 'files'
Iteration = []
loss = []
Eval = []
file_list = ['test.txt','test_conv_kernel_4.txt']
filenamelist = []
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
        print j
        fileNum = file(fil,'rb')
        for line in fileNum.read().split('\n'):
            if line.split(' ')[0] == 'Iteration':
                test1.append(int(line.split(' ')[1]))
                test2.append([line.split(' ')[8],line.split(' ')[10]])
                test3.append(float(line.split(' ')[14]))
        for li in test2:
            test2[j] = float(li[0]) * float(li[1])
            j += 1   
        Iteration.append(test1)
        loss.append(test2)
        Eval.append(test3)  
        test1 = []
        test2 = []
        test3 = []
        j = 0
        fileNum.close()
def draw(x,y,name):    
    for num in y:
        plt.plot(x[0],num)
    plt.xlabel('Iteration')
    plt.ylabel(name)
    plt.show()    
if __name__ == '__main__':
    FileList(file_list)
    readData(filenamelist)
    draw(Iteration,loss,'loss')
    draw(Iteration,Eval,'eval')