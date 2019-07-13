НЕАКТУЛЬНО от 13.07.2019

# HI-19 Полная документация
Не рекомендуется использовать эту документацию для старта, для этого есть README.

## Возможности HI-19

### config.py

Перед началом работы вам нужно создать конфиг с некоторым набором параметров. Этот конфиг вы будете использователь в течении всей работы. Файл config позволяет использовать конфиг по умолчанию. Конфигурации вы можете менять либо в config, либо загрузить свой, либо создавать и редактировать на ходу. Однако рекомендуется сразу создать свой класс конфига и унаследоваться, при этом поменять определенные параметры
Пример создания конфига:
```python
from hypo2.config import RunConfig

class MyConfig(RunConfig):
    BACKUP_DIRECTORY = ???
    MODEL_PATH = ???

config = MyConfig()
```
Параметры проекта:
  1. BACKUP_DIRECTORY - папка сохранения моделей при обучении сети
  2. MODEL_PATH - путь к основной модели
  3. VERT_RAY_THRESHOLD - минимальное количество пересечений луча с следом от ручки. Чем больше этот параметр, тем меньше будет разрезов и наоборот.
  4. VERT_RAY_CHUNKMINSIZE - минимальная высота строки. На тот случай, если луч случайно все же пролетел, но строчка получилась в 4 пикселя.
  5. VERT_RAY_CHUNKMAXSIZE - максимальная высота строки.
  6. HORIZ_RAY_* - то же самое, только для горизонтального обхода
  7. FINAL_SIZE - размер изображения, которое возвращается WordSegmentator и идет на вход HI model. Не рекомендуется менять этот параметр в ран-тайме.
  8. CLASS_COUNT - количество классов, то есть разных людей, на которых мы будем обучать.
  9. SHEET_ANGLE - угол для поворота изображения. Либо 0, либо "adaptive"
  10. DEVICE - "cpu" или "cuda"
  11. CACHE_PATH - путь для кэша
  12. Далее идут параметры обучения

### preprocessor.py

```python
from hypo2.preprocessor import Preprocessor
pr = Preprocessor(config)
```

  1. norm - нормализует изображение, принимая Pillow Image и возвращая ndarray в (w, h, 3) в [0; 255]
  2. open_norm - нормализует изображение из path
  3. open_norm_segment - сегментирует слова после того, как нормализирует страницу
  4. segment_words - сегментирует нормализованное изображение

Требования к входу segment_words:
  1. Все рукописные строчки должны были быть строго параллельны горизонту
  2. Хотя вход должен быть ndarray (w, h, 3) и [0; 255]
Выход:
Массив изображений размера config.FINAL_SIZE, где каждое изображение - слово.

### dataset.py

gen_dataset позволяет получить датасет сразу в HI-compatible виде и требует массив массивов ссылок:
```python
paths = [
  [
  "0/0.jpg",
  "0/1.jpg",
  "0/2.jpg",
  ],
  [
  "1/0.jpg",
  "1/1.jpg"
  ],
  [
  "2/0.jpg"
  ]
]
```
И чтобы получить входные и выходные данные обучающей выборки мы используем
```python
from hypo2.dataset import Dataset
dataset = Dataset(config)
X, y = dataset.gen_dataset(paths)
```
Если датасет лежит в правильной иерархии на диске, можно получить пути:
```python
paths = ds.gen_paths(path_to_your_dataset)
```

### model.py

Основной класс - HIModel.
```python
from hypo2.model import HIModel
```

#### fit
X, y - входные и выходные параметры, X - np array изображений размера FINAL_SIZE, y - np array целых чисел (номеров классов), X.shape[0] == y.shape[0]
verbose - если стоит True, будет происходить вывод loss-а по каждой эпохе и результат валидации по окончанию обучения
plot - вывод динамики loss-а по окончанию обучения
```python
X = np.random.randn(20 * config.FINAL_SIZE[0] * config.FINAL_SIZE[1] * 3).reshape((20, config.FINAL_SIZE[0], config.FINAL_SIZE[1], 3))
y = np.random.randint(3, size=[20])
model = HIModel(config)
model.fit(X, y)
```

#### extract
Возвращает Центр для (BS, 3, \*config.NN_INPUT_SIZE) и [-0.5; +0.5]

#### open, save
Открывает и сохраняет модель. При этом, open не является конструктором и модель должна быть инициализирована.

### api

Это файл API к HI-19. 

#### RunEnv (использование модели)
Инициализируем RunEnv
```python
from hypo2.api import RunEnv
runenv = RunEnv(config)
```
Модель подгружается по пути config.MODEL_PATH

##### get_center, open_image

get_center вернет Центр по изображению.
```python
center, weight = runenv.get_center(runenv.open_image("D:/image.jpg"))
```

##### differ, dist, differ_from_paths

differ находит расстояние между Центрами.
```python

```

#### FitTime (обучение модели)
Нам понадобится dataset в привычном формате (массив массивов ссылок)
```python
from hypo2.runtime import RunTime
fittime = FitTime(config, dataset)
```
Обучим:
```python
model, nm, ws = fittime.fit(verbose=True, plot=True)
```
Где model - обученная модель HIModel, nm - Normalizer, ws - WordSegmentator. Последние два необязательно использовать в дальнейшем.

## Системное описание

Почти каждый объект проекта - дочерний BaseHIObj.
Почти все классы требуют config. Хотя и не все объекты требуют все параметры проекта, это создано для удобства, так как этот конфиг универсален.

### basef.py

Предоставляет описание BaseHIObj и BaseConfig (не рекомендуется здесь что-то менять)

