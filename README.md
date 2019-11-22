<!--annstart-->
### HI19
HI-19 is a system developed for identifying and distinguishing peoples' handrwritings. This is the first such product.
<!--annend-->
## Generally

### How does HI-19 work?
  1. First, we no. Листы бумаги далее сегментируются на слова.
  3. Теперь на них должна обучиться модель. HIModel из model
  4. Для работы с API модели рекомендуется использовать классы RunEnv для рантайма и FitEnv для обучения из hypo2.api

### Структура файлов
  - config.py - файл конфигурации всего проекта.
  - model.py - описание и функционал модели HI
  - api.py - API для использования модели

## Работа с моделью

### Прежде всего
Перед тем, как использовать HI19, нужно настроить проект. Создадим myconfig.py (в папке "hypo2/..") и впишем туда код:
```python
from hypo2.config import RunConfig

class MyConfig(RunConfig):
    BACKUP_DIRECTORY = ???
    MODEL_PATH = ???
```
Вместо ??? следует вписать либо путь, либо None. BACKUP_DIRECTORY - путь к папке, куда будут сохраняться резервные копии при обучении. MODEL_PATH - путь к основной модели.

Теперь необходимо создать переменную конфига. Это просто:
```python
from myconfig import MyConfig
config = MyConfig()
```
При этом, вы можете менять какие-то параметры конфигурации прямо на ходу. Чтобы посмотреть текущие значения конфига просто сделайте
```
print(config)
```

### Обучение
Датасет, который мы имеем, должен представлять из себя массив следующего вида:
```
[
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
Иначе говоря каждый элемент датасета - это массив ссылок на картинки одного класса. При этом необходимо изменить конфигурацию проекта, как минимум, CLASS_COUNT.
Итак, пусть мы уже имеем Датасет dataset.
Импортируем api
```python
from hypo2.api import FitEnv
```
FitTime позволяет очень легко обучить модель. Создадим экземпляр среды:
```python
fitenv = FitEnv(config)
```
Среда подготовлена к обучению. FitEnv сам создаст модель, обработает входные данные и настроит систему. Обучим модель:
```python
model = fitenv.fit(dataset, verbose=True, plot=False)
```
verbose - выводить ли результаты эпох.
plot - вывести ли динамику loss по окончанию обучения.
model - это обученная модель HI.

#### Работа с моделью
Модель можно сохранить с помощью
```python
model.save()
```
То есть без указания пути. Путь будет равным указанному в config.MODEL_PATH

Либо (устаревший метод)
```python
model.saveto("D:/HI.19")
```
Тут путь указывается вручную

Чтобы открыть модель, достаточно указать в config.MODEL_PATH путь к модели и уже после создать модель.
```python
config.MODEL_PATH = "D:/HI.19"
model = HIModel(config)
```
Если требуется переоткрыть модель, можно использовать
```python
model.open()
```
Либо использовать устаревший метод:
```python
model.openfrom("HI.19")
```

### Использование

Импортирум и создаем RunEnv
```python
from hypo2.api import RunEnv
runenv = RunEnv(config)
```
Теперь попробуем получить Центр по фото:
```python
image = runenv.open_image("johns_text.jpg")
cw = runenv.get_center(image)
assert cw is not None, "0 words found"
center, weight = cw
```
Где center - это вектор длиной config.FEATURES_COUNT, а weight - целое число, обозначающее количество найденных слов (оно же - вес центра, требуемый впоследствии если понадобится двигать центр при наличии дополнительных данных о человеке)
Чтобы сравнить два изображения и получить разницу между ними нужно использовать differ
```python
distance = runenv.differ(image1, image2)
```
А если уже есть центры, то расстояние между ними можно получить функцией
```python
distance = runenv.dist(john_center, mary_center)
```
Посмотреть UML-диаграммы можно в единственном файле \*.png

Все.
