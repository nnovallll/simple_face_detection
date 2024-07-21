import cv2
import os
import numpy as np
from PIL import Image
import logging

# Inisialisasi logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Membuat pengenal wajah LBPH
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    logging.info("LBPHFaceRecognizer berhasil dibuat.")
except Exception as e:
    logging.error(f"Gagal membuat LBPHFaceRecognizer: {e}")
    exit()

# Load detektor wajah
detector_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(detector_path):
    logging.error(f"File detektor wajah '{detector_path}' tidak ditemukan.")
    exit()
detector = cv2.CascadeClassifier(detector_path)

def getImagesWithLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []

    for imagePath in imagePaths:
        try:
            logging.debug(f"Memproses gambar: {imagePath}")
            pilImage = Image.open(imagePath).convert('L')  # Konversi gambar ke grayscale
            imageNp = np.array(pilImage, 'uint8')  # Konversi gambar ke array numpy
            Id = int(os.path.split(imagePath)[-1].split(".")[1])  # Mendapatkan ID dari nama file
            faces = detector.detectMultiScale(imageNp)

            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                Ids.append(Id)
        except Exception as e:
            logging.error(f"Error processing image {imagePath}: {e}")

    return faceSamples, Ids

# Mendapatkan gambar dan ID dari folder DataSet
try:
    faces, Ids = getImagesWithLabels('DataSet')
    if len(faces) == 0 or len(Ids) == 0:
        logging.error("Tidak ada wajah atau ID yang ditemukan di folder 'DataSet'.")
        exit()
except Exception as e:
    logging.error(f"Error mendapatkan gambar dan label: {e}")
    exit()

# Melatih pengenal wajah dengan gambar yang diambil
try:
    recognizer.train(faces, np.array(Ids))
    logging.info("Pengenal wajah berhasil dilatih.")
except Exception as e:
    logging.error(f"Error saat melatih pengenal wajah: {e}")
    exit()

# Pastikan folder training ada, jika tidak buat baru
training_folder = 'training'
if not os.path.exists(training_folder):
    os.makedirs(training_folder)
    logging.info(f"Folder '{training_folder}' dibuat.")

# Menyimpan model yang telah dilatih
model_path = os.path.join(training_folder, 'training.xml')
try:
    recognizer.save(model_path)
    logging.info(f"Model disimpan di '{model_path}'")
except Exception as e:
    logging.error(f"Error saat menyimpan model: {e}")
    exit()
