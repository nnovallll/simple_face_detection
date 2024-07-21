import cv2

# Inisialisasi kamera
camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

# Pastikan kamera terbuka
if not video.isOpened():
    print("Tidak bisa membuka kamera")
    exit()

# Menggunakan detektor wajah dari OpenCV
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Mengambil ID pengguna dari input
id = input('Masukkan ID pengguna: ')

# Inisialisasi penghitung gambar
a = 0

while True:
    # Membaca frame dari kamera
    check, frame = video.read()
    if not check:
        print("Tidak dapat membaca frame dari kamera")
        break

    # Konversi ke skala abu-abu untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    # Menyimpan gambar dan menggambar kotak di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        a += 1
        cv2.imwrite(f"DataSet/User.{str(id)}.{str(a)}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Tampilkan frame dengan wajah yang terdeteksi
    cv2.imshow("Deteksi Wajah", frame)

    # Tunggu tombol tekan, dan keluar jika tombol 'q' ditekan atau jika 24 gambar telah disimpan
    if cv2.waitKey(1) & 0xFF == ord('q') or a >= 25:
        break

# Membersihkan dan menutup jendela
video.release()
cv2.destroyAllWindows()
print(f"{a} gambar telah disimpan dalam folder DataSet.")
