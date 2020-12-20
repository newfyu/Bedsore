import torchvision
import torch

"读取train.txt, 清除其中没有object的样本，生成新的train_clearning.txt"

ds = torchvision.datasets.VOCDetection('data',year='2007')
 
with open('train_clearning.txt','a') as f:
    for i in vocds:
        name = i[1]['annotation']['filename'][:-4]
        if 'object' in i[1]['annotation'].keys():
            print(name)
            f.write(name+'\n')
