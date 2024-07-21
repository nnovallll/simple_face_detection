import cv2
import sqlite3
import numpy as np
from PIL import Image, ImageDraw, ImageFont

camera = 0
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
a = 0
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("c://uas/training/training.xml")
id = 0
font_path = "D:/Kuliah/Kuliah Noval/KKN/Design/Font/Cheltenham-BoldItalic.otf"  # Path ke font kustom
font_size = 20  # Ukuran font
font = ImageFont.truetype(font_path, font_size)
path = 'DataSet'

def getProfile(id):
    conn = sqlite3.connect("c://uas/uas_ai")
    
    # Get column names
    cursor = conn.execute("PRAGMA table_info(store)")
    columns = [row[1] for row in cursor.fetchall()]
    
    # Get profile data
    cmd = "SELECT * FROM store WHERE id=" + str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return columns, profile

def draw_text_with_background(img, text, position, font, font_color, bg_color, padding=5):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    text_size = draw.textsize(text, font=font)
    text_x, text_y = position
    box_coords = [text_x - padding, text_y - padding, text_x + text_size[0] + padding, text_y + text_size[1] + padding]
    draw.rectangle(box_coords, fill=bg_color)
    draw.text((text_x, text_y), text, font=font, fill=font_color)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img

while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        a = a + 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        columns, profile = getProfile(id)
        if profile is not None:
            for i, (col, value) in enumerate(zip(columns[1:], profile[1:])):  # Skip 'id' column
                text = f"{col}: {value}"
                text_position = (x, y + h + 30 + 30 * i)
                frame = draw_text_with_background(frame, text, text_position, font, (255, 255, 255), (0, 0, 0))
    cv2.imshow("wajah", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    print(a)

video.release()
cv2.destroyAllWindows()
