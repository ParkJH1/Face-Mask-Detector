# 얼굴 랜드마크 추출

import face_recognition
import math
from PIL import Image, ImageDraw

face_image_path = 'data/without_mask/100.jpg'
mask_image_path = 'data/mask.png'

face_image_np = face_recognition.load_image_file(face_image_path)
face_locations = face_recognition.face_locations(face_image_np)
face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)

mask_image = Image.open(mask_image_path)
face_landmark_image = Image.fromarray(face_image_np)
draw = ImageDraw.Draw(face_landmark_image)

for face_landmark in face_landmarks:
    chin1 = face_landmark['chin'][0]
    chin15 = face_landmark['chin'][16]
    chin8 = face_landmark['chin'][8]
    nose_bridge1 = face_landmark['nose_bridge'][0]

    draw.point(chin1)
    draw.point(chin15)

    # 마스크 너비: (턱)chin1, (턱)chin15 거리
    mask_width = int(math.sqrt((chin1[0] - chin15[0]) ** 2 + (chin1[1] - chin15[1]) **2))

    # 마스크 높이: (턱)chin8, (콧대)nose_bridge1 거리
    mask_height = int(math.sqrt((chin8[0] - nose_bridge1[0]) ** 2 + (chin8[1] - nose_bridge1[1]) **2))

    mask_image = mask_image.resize((mask_width, mask_height))

    face_landmark_image.paste(mask_image, chin1, mask_image)

face_landmark_image.show()
