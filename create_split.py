import os
import random
import pandas as pd

base_path = "datasets/FaceForensics/raw"

classes = {
    "original": 0,
    "Deepfakes": 1,
    "Face2Face": 1,
    "FaceSwap": 1,
    "FaceShifter": 1,
    "NeuralTextures": 1
}

data = []

for folder, label in classes.items():
    folder_path = os.path.join(base_path, folder)
    videos = os.listdir(folder_path)
    for video in videos:
        data.append([os.path.join(folder, video), label])

random.shuffle(data)

train_split = int(0.7 * len(data))
val_split = int(0.85 * len(data))

train = data[:train_split]
val = data[train_split:val_split]
test = data[val_split:]

os.makedirs("datasets/FaceForensics/splits", exist_ok=True)

pd.DataFrame(train).to_csv("datasets/FaceForensics/splits/train.csv", sep=" ", header=False, index=False)
pd.DataFrame(val).to_csv("datasets/FaceForensics/splits/val.csv", sep=" ", header=False, index=False)
pd.DataFrame(test).to_csv("datasets/FaceForensics/splits/test.csv", sep=" ", header=False, index=False)

print("Split Created Successfully")
