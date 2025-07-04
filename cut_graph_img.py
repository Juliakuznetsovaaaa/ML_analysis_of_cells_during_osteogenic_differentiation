from PIL import Image
import os

# Настройки
input_folder = "dir/good"  # Папка с исходными изображениями
output_folder = "good_red_cut"  # Папка для сохранения фрагментов
valid_extensions = ['.tif', '.tiff']  # Поддерживаемые форматы

# Создаем папку для результатов, если её нет
os.makedirs(output_folder, exist_ok=True)


# Функция для разделения изображения
def split_image(img_path, output_path):
    try:
        with Image.open(img_path) as img:
            width, height = img.size

            # Вычисляем размеры фрагментов
            chunk_width = width // 5
            chunk_height = height // 3

            # Перебираем все части
            for row in range(3):
                for col in range(5):
                    # Вычисляем координаты области
                    left = col * chunk_width
                    upper = row * chunk_height
                    right = (col + 1) * chunk_width
                    lower = (row + 1) * chunk_height

                    # Вырезаем и сохраняем фрагмент
                    chunk = img.crop((left, upper, right, lower))
                    chunk.save(os.path.join(output_path,
                                            f"{os.path.splitext(filename)[0]}_row{row}_col{col}.png"))

    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {str(e)}")


# Обрабатываем все изображения в папке
for filename in os.listdir(input_folder):
    # Проверяем расширение файла
    ext = os.path.splitext(filename)[1].lower()
    if ext in valid_extensions:
        file_path = os.path.join(input_folder, filename)
        split_image(file_path, output_folder)
        print(f"Обработано: {filename}")
    else:
        print(f"Пропущен файл {filename} (неподдерживаемый формат)")

print("Готово! Все изображения обработаны.")