# Привязка тем курса к реализации

Ниже — краткая карта тем курса и их связь с кодом проекта.

## 1) Детектирование объектов и transfer learning
- Faster R-CNN (Region-Based CNN, Fast/Faster R-CNN): `src/train_detector.py`, `src/evaluate_detector.py`, `src/infer_detector.py`.
- Метрики локализации: IoU и mAP@IoU реализованы в `src/utils.py` и используются в `src/evaluate_detector.py`.

## 2) Границы, Хафф, особые точки
- Постановка задачи обнаружения границ: `src/classical_cv.py`.
- Детектор Кэнни: `src/classical_cv.py` → `edges_canny.png`.
- Преобразование Хафа (линии/окружности): `src/classical_cv.py` → `hough_lines.png`, `hough_lines_fast.png`, `hough_circles.png`.
- Быстрое преобразование Хафа (HoughLinesP): `src/classical_cv.py` → `hough_lines_fast.png`.
- Детектор Моравеца: `src/classical_cv.py` → `moravec.png`.
- Детектор Харриса и Shi–Tomasi: `src/classical_cv.py` → `harris.png`, `shi_tomasi.png`.
- SIFT (или ORB при отсутствии SIFT): `src/classical_cv.py` → `sift.png`.

## 3) Классификация изображений
- Линейный классификатор SVM и мешок визуальных слов (BoVW): `src/classification_bovw_svm.py`.
- Архитектуры LeNet-5 и VGG-подобная: `src/cnn_classification.py`.

## 4) Оптический поток
- Метод Лукаса–Канаде: `src/optical_flow.py` → `flow_lk.png`.
- Метод Horn–Schunck: `src/optical_flow.py` → `flow.png`.

## 5) Локализация и слабая локализация
- Локализация объектов в виде боксов и оценка качества реализованы в детекторе.
- Weakly Supervised Localization: в рамках проекта отражено концептуально через классификацию на кропах (см. `src/cnn_classification.py`).

## 6) Дополнительные темы (теория/ориентиры)
- Single Shot Detector (SSD): реализован через SSD Lite (см. `src/train_detector.py` с `--model ssdlite`).
- Mask R-CNN, FCN, U-Net, Deconvolution Network: не реализованы в коде, но относятся к тем же классам задач.
- Объектный трекинг, распознавание лиц, поиск изображений, OCR: не реализованы — требуют отдельных датасетов и сценариев.

## 7) Данные
- Датасет: `archive/annotations.csv` + `archive/images`.
- Классы и боксы читаются напрямую из CSV (см. `src/datasets.py`).
