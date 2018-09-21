__author__ = 'yxzhang'
import os
from PIL import Image
import numpy as np

path = '/DATA3_DB7/data/yxzhang/WorkSpace/Separate/chinese/Experiments/5000_train/Results_10-10/train_result1'
filelist = os.listdir(path+'/images')
filelists = []
for file in filelist:
    if "outputs" in file:
        filelists.append(file)

filelists = sorted(filelists,reverse=True)
filelists = filelists[0:1000]
print filelists

index_path = os.path.join(path, "index_sample.html")

index = open(index_path, "wa")
index.write("<html><body><table><tr>")
index.write("<th>name</th><th>targets</th><th>outputs</th></tr>")

for i in range(len(filelists)):
    file1 = filelists[i]
    file2 = file1.replace('outputs','targets')
    print file1,file2
    index.write("<tr>")
    name = file1.split('-outputs')[0]
    print name
    index.write("<td>%s</td>" % name)
    index.write("<td><img src='images/%s'></td>" % file2)
    index.write("<td><img src='images/%s'></td>" % file1)
    index.write("</tr>")