from transform_data import *
import matplotlib.pyplot as plt
shit=extract_data("train","../mini-imagenet")
ds=miniImagenet(shit["train"],trans_data,128,128,5,15,False)
img=ds[10][0]
img=img.reshape(img.shape[1],img.shape[2],img.shape[0])
img1=ds[10][1]
img1=img1.reshape(img1.shape[1],img1.shape[2],img1.shape[0])
print(img.shape)
print(img1.shape)
img=np.array(img)
img=np.uint8(img)
img1=np.array(img1)
img1=np.uint8(img1)
img=Image.fromarray(img,'RGB')
img1=Image.fromarray(img1,'RGB')
img.show()
img1.show()