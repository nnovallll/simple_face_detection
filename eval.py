import cv2
import os
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

# Inisialisasi
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training/training.xml')  # Memuat model yang telah dilatih
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Fungsi untuk mendapatkan wajah dan label dari dataset
def getImagesWithLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(imageNp)
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)
    return faceSamples, Ids

# Evaluasi metrik menggunakan data yang sama
path = 'DataSet'
faces, true_labels = getImagesWithLabels(path)

predicted_labels = []
for face in faces:
    id, _ = recognizer.predict(face)
    predicted_labels.append(id)

# Menghitung metrik evaluasi
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

# Menampilkan hasil evaluasi
print(f"Presisi: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Skor F1: {f1:.2f}")