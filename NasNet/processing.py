import os, shutil, tqdm
import numpy as np

with open('train.csv') as f:
    data = list(map(lambda x:x.strip().split(','), f.readlines()))

os.chdir('E:\\课程文件\\蛋白质预测\\2020年春节机器学习大作业训练集数据\\train')
idx = 0
number = 0
for i in tqdm.tqdm(data):
    base_path = os.path.join(os.getcwd(), i[0])
    number = number + 1
    if number < 501:
        for j in os.listdir(base_path):
            shutil.copy(os.path.join(base_path, j), 'E:\课程文件\蛋白质预测\蛋白质多分类\imgtrain\{}_{}.jpg'.format(idx, i[1]))
            idx += 1
    else:
        for j in os.listdir(base_path):
            shutil.copy(os.path.join(base_path, j), 'E:\课程文件\蛋白质预测\蛋白质多分类\imgtest\{}_{}.jpg'.format(idx, i[1]))
            idx += 1
