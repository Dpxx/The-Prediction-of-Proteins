import os
from imutils import paths
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from tqdm import tqdm
import cv2
import csv


def classify(p):
    res = [i for i in range(10) if p[i] >= 0.4]
    return res if len(res) > 0 else [np.argmax(p)]


# def classify_2(p):
#     p = p.tolist()
#     b = sorted(p, reverse=True)
#     if b[0] > 0.5 and b[1] > 0.5:
#         return sorted([p.index(b[0]), p.index(b[1])], reverse=False)
#     elif b[0] - b[1] < 0.1:
#         return sorted([p.index(b[0]), p.index(b[1])], reverse=False)
#     return [p.index(b[0])]


def change_lable(l: list):
    lable = [0] * 10
    for i in l:
        lable[int(i)] = 1
    return lable


img_folders = os.listdir("test")
img_folders.sort(key=lambda x: int(x[4:])) # 按照文件夹所带数字排序

model = load_model("el.model")

f = open('test_0.4.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(f)

# 按照每个文件夹分别打开
for folder in tqdm(img_folders):

    # 遍历文件夹下所有图片进行预测
    imagePaths = list(paths.list_images("test/" + folder))

    img_num = 0
    folder_res = np.zeros(shape=(10,))
    for img in imagePaths:
        img_num += 1
        image = cv2.imread(img)
        image = cv2.resize(image, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        proba = model.predict(image)[0]
        folder_res += proba
    folder_res /= img_num
    class_img = classify(folder_res)
    csv_writer.writerow([folder, ";".join(str(round(i, 4)) for i in folder_res), ";".join(str(i) for i in class_img)])

f.close()
