#coding:utf-8

import struct
import numpy as np
import  matplotlib.pyplot as plt
import Image

# 存储手写数字图片（28*28pixel，channel=1）的文件夹
imgDir = 'tmp_bmp_data'
numImages = 14 # 总的图片数
magic = 2051 # MNIST标准数据库存放images的魔数
width = 28 # 每张图片的长宽
height = 28

# 写入的二进制文件名
filename = 'my-train-images.idx3-ubyte'

binfile = open(filename,'wr')

index=0

# 以大端法写入数据
bytes=struct.pack('>IIII',magic,numImages,width,height)
binfile.write(bytes) # 写入头文件

# 写入每张图片的数据
for image in range(0,numImages):
	imgfile = imgDir+'/train_%s.bmp'%image
	# print 'imgfile=%s' % imgfile
	im = Image.open(imgfile)
	# plt.imshow(im,cmap='gray')
	# plt.show()
	pixels=im.load()
	# print 'pixels=%d' % pixels[14,9]
	for h in xrange(height):
		for w in xrange(width):
			p = chr(pixels[w,h])
			bytes=struct.pack('>c',p)
			binfile.write(bytes)

binfile.close()