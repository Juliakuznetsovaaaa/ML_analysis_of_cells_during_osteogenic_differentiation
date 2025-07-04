// Set input and output directories
inputDir = "C:/Users/Julia/PycharmProjects/imageJ/red_cntrl/";
outputDir = "C:/Users/Julia/PycharmProjects/imageJ/results/";

// Set analysis parameters
minSize = 100; // Minimum size of the colony (pixels)
maxSize = 10000; // Maximum size of the colony (pixels)

// --- Set Measurements ---


// Get list of files in input directory
fileList = getFileList(inputDir);
// Variable to hold the title of the last image processed.
lastImageTitle = "";

for (i=0; i<fileList.length; i++) {
    open(inputDir + fileList[i]);
    title = getTitle(); // Получить название текущего изображения
    lastImageTitle = title; // Update the last image title

    // Сохранить исходное изображение
    originalImage = getTitle();

    // Преобразовать в 8-bit (если необходимо)
    run("8-bit");

    // Улучшить контраст (настройте параметры под ваши данные)
    run("Enhance Contrast", "saturated=0"); // Уменьшите saturated, если изображение темное

    // Применить сглаживание
    run("Smooth");

    // Пороговая обработка
    run("Auto Threshold", "method=huang");

    // Применить маску

    run("Convert to Mask"); // Преобразовать изображение после пороговой обработки в маску

    // Сохранить маску (опционально, если нужно сохранить изображение маски)
    maskImage = getTitle();

    // Вернуться к исходному изображению
    selectWindow(originalImage);
    imageCalculator("Multiply create", originalImage, maskImage);

    // Измерить интенсивность на исходном изображении
    run("Set Measurements...", "mean redirect=None decimal=3");
    run("Measure");

    // Закрыть маску
    selectWindow(maskImage);
    close();

    // Закрыть исходное изображение
    close(originalImage);
}

// Prepare the output file name
resultsPath = outputDir + "intens_cntrl_new.csv";

// Save the results to a CSV file
saveAs("Results", resultsPath);


print("Batch analysis complete. Results saved to: " + resultsPath);