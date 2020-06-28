import csv
from imutils import paths
import os
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image

#Reading train.csv and output foldername-label pairs
def read_train(train_dir):
    label_dict={}
    with open(train_dir) as f:
        for row in csv.reader(f):
            label=row[1].rsplit(';')
            label=[int(x) for x in label]
            label_hot=np.zeros(10)
            label_hot[label]=1
            label_dict[row[0]]=label_hot
    return label_dict

#Definition of ENSG-protein dataset
class ENSG_Protein_Dataset(Dataset):
    def __init__(self,data_dir):
        self.data_dir=data_dir
        self.label_dict=read_train(self.data_dir)
        self.data=[]
        self.label=[]
        for key in self.label_dict.keys():
            img_path=list(paths.list_images(os.join(self.data_dir,key)+'/'))
            for ip in img_path:
                self.data.append(ip)
                self.label.append(label_dict[key])

        self.transform=transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.toTensor()
        ])

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self,index):
        return self.transform(Image.open(self.data[index])),self.label[index]