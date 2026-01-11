# Детектирование объектов с transfer learning 

Проект решает задачу детектирования объектов на локальном датасете `archive/`

## 1) Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Если SIFT недоступен
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

## 2) Данные

Датасет лежит в папке `archive/`
```
archive/
  images/
  annotations.csv  (filename,width,height,class,xmin,ymin,xmax,ymax)
```

## 3) Transfer Learning: Faster R-CNN или SSD Lite

Обучение (Faster R-CNN)
```bash
python src/train_detector.py --data archive --epochs 15 --batch-size 4 --output runs/sign
```

Обучение (SSD Lite)
```bash
python src/train_detector.py --data archive --epochs 15 --batch-size 4 --output runs/sign_ssd --model ssdlite
```

Оценка (mAP@0.5 IoU)
```bash
python src/evaluate_detector.py --data archive --checkpoint runs/sign/checkpoint_last.pt
```

Инференс и визуализация
```bash
python src/infer_detector.py --data archive --checkpoint runs/sign/checkpoint_last.pt --images archive/images --output runs/sign/preds
```

Если  SSD Lite, то надо добавить `--model ssdlite` в команды оценки и инференса (или использовать чекпоинт, сохраненный новой версией скрипта — он должен подхватить модель автоматически).


