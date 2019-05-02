# HI-19 Полная документация
Не рекомендуется использовать эту документацию для старта, для этого есть README.

## Возможности HI-19

### config.py

Перед началом работы вам нужно создать конфиг с некоторым набором параметров. Этот конфиг вы будете использователь в течении всей работы. Файл config позволяет использовать конфиг по умолчанию. Конфигурации вы можете менять либо в config, либо загрузить свой, либо создавать и редактировать на ходу.
Пример создания конфига:
```python
from hypo2.config import RunConfig
config = RunConfig()
```
Параметры проекта:
  1. VERT_RAY_THRESHOLD - минимальное количество пересечений луча с следом от ручки. Чем больше этот параметр, тем меньше будет разрезов и наоборот.
  2. VERT_RAY_CHUNKMINSIZE - минимальная высота строки. На тот случай, если луч случайно все же пролетел, но строчка получилась в 4 пикселя.
  3. VERT_RAY_CHUNKMAXSIZE - максимальная высота строки.
  4. HORIZ_RAY_* - то же самое, только для горизонтального обхода
  5. FINAL_SIZE - размер изображения, которое возвращается WordSegmentator и идет на вход HI model. Не рекомендуется менять этот параметр в ран-тайме.
  6. CLASS_COUNT - количество классов, то есть разных людей, на которых мы будем обучать.
  7. Далее идут параметры обучения

### normalization.py

  1. norm - нормализует изображение, принимая Pillow Image и возвращая ndarray в (w, h, 3) в [0; 255]
  2. open_norm - нормализует изображение из path

### preprocessing.py

Единственная функция, рекомендуемая к использованию - segment_words.
Требования к входу:
  1. Все рукописные строчки должны были быть строго параллельны горизонту
  2. Хотя вход должен быть ndarray (w, h, 3) и [0; 255], должно быть только два цвета: (0, 0, 0) для чернил и (255, 255, 255) для всего остального
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
from hypo2.normalization import Normalizator
from hypo2.preprocessing import WordSegmentator
nm = Normalizator(config)
ws = WordSegmentator(config)
dataset = Dataset(config, nm, ws)
X, y = dataset.gen_dataset(paths)
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

#### predict, predict_proba
predict_proba возвращает вероятность принадлежности к каждому классу
predict - это просто argmax от predict_proba

#### open, save
Открывает и сохраняет модель. При этом, open не является конструктором и модель должна быть инициализирована.

### runtime

Это файл API к HI-19. 

#### RunTime (использование модели)
Для инициализации среды RunTime помимо конфига потребуются экземпляры WordSegmentator и Normalizator, а также модель HIModel.
Создадим WordSegmentator и Normalizator
```python
from hypo2.normalization import Normalizator
from hypo2.preprocessing import WordSegmentator
nm = Normalizator(config)
ws = WordSegmentator(config)
```
Откроем обученную модель
```python
model = HIModel(config)
model.open("HI.19")
```
Инициализируем RunTime
```python
from hypo2.runtime import RunTime
runtime = RunTime(config, model, ws, nm)
```
Для классифиакции используем classify для получения id класса, classify_proba для получения вероятностей по классам и differ для сравнения
```python
from PIL import Image
image1 = Image.open("im1.jpg")
image2 = Image.open("im2.jpg")
class_id = runtime.classify(image1)
class_probabilities = runtime.classify_proba(image1)
distance = runtime.differ(image1, image2)
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

