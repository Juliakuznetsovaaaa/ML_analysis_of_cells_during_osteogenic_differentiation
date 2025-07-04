import cv2
import numpy as np
import os
from PIL import Image

input_folder = "dapi_bad_96/cntrl/"
output_folder = "dapi_bad_96/cntrl_cut"
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]

if not image_files:
    print("В папке 'od' нет TIFF-файлов.")
    exit()

min_size = 100  # Минимальная площадь контура
min_solidity = 0.85  # Минимальная "сплошность" контура
binary_threshold = 35  # Порог бинаризации
colonies_saved = 0
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Не удалось прочитать файл: {image_file}")
        continue

    h, w = image.shape
    base_name = os.path.splitext(image_file)[0]


    # Увеличение контрастности
    alpha = 1.5
    beta = 0
    image_contrasted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Используем глобальную пороговую бинаризацию с порогом
    _, thresh = cv2.threshold(image_contrasted, binary_threshold, 255, cv2.THRESH_BINARY)

    # Инвертируем изображение, чтобы овалы стали черными на белом фоне
    # Морфологическое закрытие для устранения мелких дырок
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.destroyAllWindows()# Морфологическое закрытие для устранения мелких дырок

    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)

        # Фильтр по границам
        if x < 2 or y < 2 or (x + w_cnt) > (w - 2) or (y + h_cnt) > (h - 2):
            continue
        area = cv2.contourArea(cnt)
        if area < min_size:
            continue

        # Сохранение
        colony = image[y:y + h_cnt, x:x + w_cnt]
        alpha = 1.5
        beta = 0
        image_contrasted = cv2.convertScaleAbs(colony, alpha=alpha, beta=beta)
        Image.fromarray(image_contrasted).save(f"{output_folder}/{base_name}_colony_{colonies_saved + 1}.tiff")
        colonies_saved += 1

    print(f"{image_file}: сохранено {colonies_saved} колоний")


print("\nГотово!")
