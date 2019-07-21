НЕАКТУЛЬНО от 13.07.2019

# HI-19 Полная документация
Не рекомендуется использовать эту документацию для старта, для этого есть README.

## Возможности HI-19

### config

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
  12. DS_MIN_WORDS_PER_PAGE - минимальное количество слов на страницу
  13. NN_INPUT_SIZE - размер входа изображения (tuple, например, (224, 224) )
  14. BATCH_SIZE, N_EPOCHS, LEARNING_RATE, DS_SHUFFLE_LEN - параметры обучения. DS_SHUFFLE - это по сути размер подгружаемого в память датасета. Чем больше, тем лучше, ограничено лишь возможностью компьютера.
  15. VAL_PERIOD, VAL_EPOCHS - периодичность и точность валидации. Чем меньше VAL_PERIOD - тем чаще будет обновляться информация о валидацонной точности, чем выше VAL_EPOCHS - тем выше точность. Однако, если VAL_PERIOD слишком сильно уменьшить, а VAL_EPOCHS - увеличить, может умеьшиться скорость обучения. (рекомендуемое: 4 и 4)
  16. FEATURES_COUNT, RESNET_LAYERS - параметры нейросети. FEATURES_COUNT отвечает за мерность пространства Центров, а RESNET_LAYERS - за количество слоев в модели (по умолчанию: [4, 12, 46, 4]). Эти значения крайне НЕ рекомендуется менять.


### preprocessor

```python
from hypo2.preprocessor import Preprocessor
pr = Preprocessor(config)
```

#### norm, open_norm, open_norm_segment
norm - нормализует изображение, принимая Pillow Image и возвращая ndarray в (w, h, 3) в [0; 255]
open_norm - нормализует изображение из path
open_norm_segment - сегментирует слова после того, как нормализирует страницу

#### segment_words
segment_words - сегментирует нормализованное изображение

Требования к входу segment_words:
  1. Все рукописные строчки должны были быть строго параллельны горизонту
  2. Хотя вход должен быть ndarray (w, h, 3) и [0; 255]
Выход:
Массив изображений размера config.FINAL_SIZE, где каждое изображение - слово.

### dataset

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

### model

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

differ находит расстояние между изображениями.
```python
image1 = runenv.open_image("D:/image1.jpg")
image2 = runenv.open_image("D:/image2.jpg")
dist = runenv.differ(image1, image2)
```
dist - это расстояние напрямую между Центрами
```python
image1 = runenv.open_image("D:/image1.jpg")
image2 = runenv.open_image("D:/image2.jpg")
center1 = runenv.get_center(image1)
center2 = runenv.get_center(image2)
dist = runenv.dist(center1, center2)
```

differ_from_paths
```python
dist = runenv.differ_from_paths("D:/image1.jpg", "D:/image2.jpg")
```

#### FitEnv (обучение модели)
Нам понадобится dataset в привычном формате (массив массивов ссылок)
```python
ds = Dataset(config)
dataset = ds.gen_dataset(ds.gen_paths("D:/dataset"))
from hypo2.api import FitEnv
fitenv = FitEnv(config)
```
Обучим:
```python
fittime.fit(dataset, verbose=True, plot=True)
```
verbose позволит выводить текстовые сообщения, plot - графики изменения показателей точности модели.

### visualizer

Предоставляет небольшой набор инструментов визуализации.

#### get_centers_from_xy
Вернет центры по словам
```python
from hypo2.addit.visualizer import Visualizer
vs = Visualizer(config)
X, y = ds.gen_dataset(paths)
fXl, fyl = vs.get_centers_from_xy(X, y)
```
В get_centers_from_xy есть параметр classes. На тот случай, если мы хотим отобразить только какие-то определенные классы (в classes тогда указать номер класса).

#### build_comp
Построит визуализацию распределения центров по классам. Аргумент comp - это алгоритм для получения проекции Центров на плоскость.
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
X, y = ds.gen_dataset(paths)
fXl, fyl = vs.get_centers_from_xy(X, y)
build_comp(fXl, fyl, PCA)
build_comp(fXl, fyl, TSNE)
```

#### get_distance_distribution, build_dist_distr
Позволит получить распределение расстояний между Центрами одного человека и разных.

```python
X, y = ds.gen_dataset(paths)
fXl, fyl = vs.get_centers_from_xy(X, y)
the_same_person, different_people = vs.build_dist_distr(fXl, fyl)
vs.build_dist_distr(the_same_person, different_people)
```

### functions

Предоставляет широкий набор функций, большинство из которых дублируются в публичных классах.

```python
from hypo2.addit.functions import Functional as F
```

#### count_distr

Аргумент - массив целых чисел (номеров классов). Результат - количество сэмплов по каждому из этих чисел.

## Системное описание

Почти каждый объект проекта - дочерний BaseHIObj.
Почти все классы требуют config. Хотя и не все объекты требуют все параметры проекта, это создано для удобства, так как этот конфиг универсален.

### basef

Предоставляет описание BaseHIObj и BaseConfig. Не рекомендуется к использованию напрямую.

### projecttest

Предоставляет тест самого проекта. Не рекомендуется к использованию напрямую.

### cache

Это файл кэша препроцессора. Рекомендуемое использование:
```python
from hypo2.base.cache import Cache
cache = Cache(config)
cache.clear()
```
