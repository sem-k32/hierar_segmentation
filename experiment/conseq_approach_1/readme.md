# Модель последовательной сегментации

Обучаются 4 модели, каждая из которых ответственна за определённые классы иерархии.

- первая модель обучается отделять человека и фон
- вторая модель обучается отделять lower/upper body
- третья модель обучается отделять upper body классы
- четвёртая модель обучается отделять lower body классы

Для финальной сегментации изображение последовательно проходит через все модели; каждая следующая модель использует предсказание вышестоящей по иерархии для сегментации только тех пикселей, которые принадлежат вышестоящему подклассу. При этом пиксели всех остальных классов зануляются (как на обучении, так и на валидации). 

### Плюсы подхода

- каждая отдельная модель может научиться надёжно классифицировать классы своей ответственности, становясь в каком-то смысле "экспертом"
- размеры моделей-экспертов могут быть намного меньше, чем размеры модели-монолита, предсказывающей все классы сразу

### Минусы подхода

- из-за неудачной классификации вышестоящей по иерархии модели, прогноз текущего "эксперта" также может ухудшится из-за нерелевантных пикселей на входе. Поэтому каждую нелистовую по иерархии модель необходимо обучить быть максимально надёжной

Каждая модель имеет архитектуру типа UNet. Структура энкодеров/декодеров практически аналогична. Энкодер дополнительно использует дропаут для ансамблизации и дообучения декодера. Используется layer norm на выходе каждого блока. Также для моделей 2-4 из-за большого числа нулевых пискелей входного изображения используется дополнительный начальный maxpool, чтобы модель быстрее смогла извлекать признаки.

### Инфраструктура

Параметры стадии обучения каждой модели находятся в ```params.yaml``` и ```params.py```. Модель задаётся в ```model.py```. Стадия препроцессинга включает подсчёт частоты классов и поиск чёрно-белых изображений. Финальная стадия включает сборку итогового сегментатора, подсчёт *mIoU* и визуализация сегментаций.

### Результаты

Артефакты пайплайна, метрики, примеры итоговой мегментации validate изображений находятся в ```results/```. Пример использования сегментатора находится в ```/final_model_stage/valid_segmentation.py```.

Итоговые метрики

|mIoU_0|mIoU_1|mIoU_2|
|---|---|---|
|0.36025|0.32516|0.77505|

Для просмотра метрик и примеров сегментации каждой модели-эксперта:

```bash
    tensorboard --logdir=results/metrics
```

### Репликация эксперимента

Для полного повторения всего пайплайна подхода запустите ```pipeline.sh```
