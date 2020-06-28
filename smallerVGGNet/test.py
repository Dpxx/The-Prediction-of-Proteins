import os
from imutils import paths
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import cv2
import csv


def classify(p):
    res = [i for i in range(10) if p[i] >= 0.4]
    return res if len(res) > 0 else [np.argmax(p)]


def classify_1(p):
    res = [i for i in range(10) if p[i] >= 0.45]
    return res if len(res) > 0 else [np.argmax(p)]


def classify_2(p):
    p = p.tolist()
    b = sorted(p, reverse=True)
    if b[0] > 0.5 and b[1] > 0.5:
        return sorted([p.index(b[0]), p.index(b[1])], reverse=False)
    elif b[0] - b[1] < 0.1:
        return sorted([p.index(b[0]), p.index(b[1])], reverse=False)
    return [p.index(b[0])]


def change_lable(l: list):
    lable = [0] * 10
    for i in l:
        lable[int(i)] = 1
    return lable


folder_num = len(os.listdir("train"))
img_folders = os.listdir("train")[-int(folder_num * 0.20):]

model = load_model("el.model")

f = open('test.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(f)
csv_data = csv.reader(open("train.csv"))
dic = {}
for i in csv_data:
    dic[i[0]] = i[1]

y_true = []
y_score = []
y_pred = []
y_t = []
y_f = []
# 按照每个文件夹分别打开
for folder in tqdm(img_folders):
    y_true.append(change_lable(dic[folder].split(";")))

    # 遍历文件夹下所有图片进行预测
    imagePaths = list(paths.list_images("train/" + folder))
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
    class_img_2 = classify_2(folder_res)
    class_img_1 = classify_1(folder_res)
    csv_writer.writerow([folder, ";".join(str(round(i, 4)) for i in folder_res), ";".join(str(i) for i in class_img)])

    y_score.append(folder_res)
    y_pred.append(change_lable(class_img))
    y_t.append(change_lable(class_img_2))
    y_f.append(change_lable(class_img_1))
f.close()
y_true = np.array(y_true)
y_score = np.array(y_score)
y_pred = np.array(y_pred)
y_t = np.array(y_t)
y_f = np.array(y_f)
# 计算AUC
roc_auc_score = metrics.roc_auc_score(y_true, y_score, average="macro")
print("AUC:", roc_auc_score)
print("threshold 0.4:")
# 计算macro F1
macro_f1 = metrics.f1_score(y_true, y_pred, average="macro")
print("macro F1:", macro_f1)
# 计算
micro_f1 = metrics.f1_score(y_true, y_pred, average="micro")
print("micro F1:", micro_f1, end='\n\n')

print("threshold 0.45:")
# 计算macro F1
macro_f1 = metrics.f1_score(y_true, y_f, average="macro")
print("macro F1:", macro_f1)
# 计算
micro_f1 = metrics.f1_score(y_true, y_f, average="micro")
print("micro F1:", micro_f1)
print('')

# 计算macro F1
macro_f1 = metrics.f1_score(y_true, y_t, average="macro")
print("macro F1:", macro_f1)
# 计算
micro_f1 = metrics.f1_score(y_true, y_t, average="micro")
print("micro F1:", micro_f1)
