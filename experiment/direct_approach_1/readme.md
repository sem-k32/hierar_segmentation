# Модель прямой сегментации

Обучается end-to-end модель, пытающаяся предсказать все классы сразу. Архитектура типа UNet.

**В общем подход не оправдал ожиданий, т.к. требует больших вычислительных ресурсов.**

### Плюсы подхода

- всего одна модель для получения полной иерархической сегментации

### Минусы подхода

- трудно настраивать лосс для балансировки неравномерности по классам, настривать парметры оптимизации и параметры самой модели для получения приемлимого результата
- нужна модель с очень большим количеством параметров для решения трудоёмкой задачи предсказания сразу всех классов

Хотя здравый смысл подсказывает, что разные части тела человека имеют разную текстуру и расположение для их отделения, модель должна быть действительно большой для решения этой задачи. Возможно, добиться того же результата можно и с меньшим количеством парметорв

### Инфраструктура

Параметры стадии обучения каждой модели находятся в ```params.yaml``` и ```train_stage.py```. Модель задаётся в ```model.py```. Стадия препроцессинга включает подсчёт частоты классов и поиск чёрно-белых изображений. Финальная стадия включает сборку итогового сегментатора, подсчёт *mIoU* и визуализация сегментаций.

Данный подход имеет свою директорию ```src```, т.к. разрабатывался раньше всего.

### Результаты

Артефакты пайплайна, метрики, примеры итоговой мегментации validate изображений находятся в ```results/```.

Итоговые метрики

|mIoU_0|mIoU_1|mIoU_2|
|---|---|---|
|-|-|0.1839|

Для просмотра всех метрик и примеров сегментации:

```bash
    tensorboard --logdir=results/metrics
```

### Репликация эксперимента

Для полного повторения всего пайплайна подхода запустите ```pipeline.sh```