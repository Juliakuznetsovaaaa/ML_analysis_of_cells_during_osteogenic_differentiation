inputDir = "C:/Users/Julia/PycharmProjects/imageJ/dapi_bad_7/10x/cntrl_cut/";
outputDir = "C:/Users/Julia/PycharmProjects/imageJ/dapi_bad_7/";

// Настраиваем параметры анализа
minSize = 0; // Минимальный размер колонии (в пикселях)
maxSize = 5500000; // Максимальный размер колонии (в пикселях)
thresholdValue = 120;
// Настраиваем измерения
run("Set Measurements...", "area perimeter shape integrated Feret's display add redirect=None decimal=3 fit");

// Получаем список файлов
fileList = getFileList(inputDir);

for (i=400; i<fileList.length; i++) {
    open(inputDir + fileList[i]);
    title = getTitle(); // Получить название текущего изображения

    // Преобразование в 8-битное изображение
    run("8-bit");

    // Размытие Гаусса
    run("Gaussian Blur...", "sigma=2");

    // Усиление контраста
    run("Enhance Contrast", "saturated=0.35 normalize equalize");

    // Установка порога
    setThreshold(thresholdValue, 255);

    // Опция черного фона
    setOption("BlackBackground", true);

    // Преобразование в маску
    run("Convert to Mask");

    // Анализ частиц
    run("Analyze Particles...", "size=" + minSize + "-" + maxSize + " circularity=0.00-1.00 show=Outlines display summarize add fit");

    close(title);
}

// Путь к файлу для результатов
resultsPath = outputDir + "cntrl_ellipse.csv";

// Сохранение результатов
saveAs("Results", resultsPath);

print("Batch analysis complete. Results saved to: " + resultsPath);
