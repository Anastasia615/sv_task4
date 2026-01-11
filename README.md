# Детектирование объектов с transfer learning (Synthetic Sign Language)

Проект решает задачу детектирования объектов на локальном датасете `archive/` и включает демонстрации классических тем курса компьютерного зрения.

## 1) Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Если SIFT недоступен, замените пакет:
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

## 2) Данные

Датасет уже лежит в папке `archive/`:
```
archive/
  images/
  annotations.csv  (filename,width,height,class,xmin,ymin,xmax,ymax)
```

## 3) Transfer Learning: Faster R-CNN или SSD Lite

Обучение (Faster R-CNN):
```bash
python src/train_detector.py --data archive --epochs 15 --batch-size 4 --output runs/sign
```

Обучение (SSD Lite, быстрее на CPU):
```bash
python src/train_detector.py --data archive --epochs 15 --batch-size 4 --output runs/sign_ssd --model ssdlite
```

Оценка (mAP@0.5 IoU):
```bash
python src/evaluate_detector.py --data archive --checkpoint runs/sign/checkpoint_last.pt
```

Инференс и визуализация:
```bash
python src/infer_detector.py --data archive --checkpoint runs/sign/checkpoint_last.pt --images archive/images --output runs/sign/preds
```

Если обучали SSD Lite, добавьте `--model ssdlite` в команды оценки и инференса (или используйте чекпоинт, сохраненный новой версией скрипта — он подхватит модель автоматически).

## 4) Классические темы курса

Границы, Хафф, особые точки, SIFT:
```bash
python src/classical_cv.py --image archive/images/image_000000.jpg --outdir runs/classical
```

Оптический поток (Lucas–Kanade, Horn–Schunck):
```bash
python src/optical_flow.py --image1 archive/images/image_000000.jpg --image2 archive/images/image_000001.jpg --out runs/flow.png
```

Мешок визуальных слов + линейный SVM:
```bash
python src/classification_bovw_svm.py --data archive --k 128 --max-per-image 5
```

LeNet-5 / VGG-подобная классификация:
```bash
python src/cnn_classification.py --data archive --model lenet --epochs 5
```

## 5) Покрытие тем курса

См. `course_notes.md` — краткая привязка тем курса к реализованным скриптам и архитектурам.
