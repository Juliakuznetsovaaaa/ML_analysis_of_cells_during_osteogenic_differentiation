import cv2
import numpy as np
import os
from PIL import Image

input_folder = "cntrl"
output_folder = "cntrl_cut_new"
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]

if not image_files:
    print("В папке 'od' нет TIFF-файлов.")
    exit()

min_size = 0  # Минимальная площадь контура
min_solidity = 0.85  # Минимальная "сплошность" контура
binary_threshold = 10  # Порог бинаризации
colonies_saved = 0
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue

    h, w = image.shape
    base_name = os.path.splitext(image_file)[0]

    image_eq = cv2.equalizeHist(image)
    image_blur = cv2.GaussianBlur(image_eq, (7, 7), 0)  # Увеличенное ядро

    # Адаптивная бинаризация
    binary = cv2.threshold(image_blur, binary_threshold, 255, cv2.THRESH_BINARY)[1]

    # Морфологическое закрытие для устранения мелких дырок
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)


    # Поиск контуров
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    base_name = os.path.splitext(image_file)[0]
    print(base_name)
    col_num=0



    for cnt in contours:
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)

        # Фильтр по границам
        if x < 2 or y < 2 or (x + w_cnt) > (w - 2) or (y + h_cnt) > (h - 2):
            continue

        # Фильтр по площади
        area = cv2.contourArea(cnt)
        if area < min_size:
            continue

        # Фильтр по "сплошности" (отношение площади контура к площади его выпуклой оболочки)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area
        if solidity < min_solidity:
            continue

        # Сохранение
        col_num+=1
        colony = image[y:y + h_cnt, x:x + w_cnt]
        Image.fromarray(colony).save(f"{output_folder}/{base_name}_colony_{col_num}.tiff")
        colonies_saved += 1


    print(f"{image_file}: сохранено {colonies_saved} колоний")

print("\nГотово!")