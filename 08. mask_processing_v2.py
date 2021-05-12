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
    chin_begin = face_landmark['chin'][0]
    chin_end = face_landmark['chin'][16]
    chin_centor = face_landmark['chin'][8]
    nose_bridge_begin = face_landmark['nose_bridge'][0]

    draw.point(chin_begin)
    draw.point(chin_end)

    mask_centor_x = (chin_begin[0] + chin_end[0]) // 2
    mask_centor_y = (chin_centor[1] + nose_bridge_begin[1]) // 2

    mask_width_ratio = 1.1
    # 마스크 너비: (턱)chin1, (턱)chin15 거리
    mask_width = math.sqrt((chin_begin[0] - chin_end[0]) ** 2 + (chin_begin[1] - chin_end[1]) ** 2)
    mask_width = int(mask_width * mask_width_ratio)

    # 마스크 높이: (턱)chin8, (콧대)nose_bridge1 거리
    mask_height = int(math.sqrt((chin_centor[0] - nose_bridge_begin[0]) ** 2 + (chin_centor[1] - nose_bridge_begin[1]) **2))

    mask_image = mask_image.resize((mask_width, mask_height))

    mask_position_x = mask_centor_x - mask_image.width // 2
    mask_position_y = mask_centor_y - mask_image.height // 2

    face_landmark_image.paste(mask_image, (mask_position_x, mask_position_y), mask_image)

face_landmark_image.show()
