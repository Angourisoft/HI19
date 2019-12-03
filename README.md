# HI19
HI-19 is a system developed for identifying and distinguishing peoples' handrwritings. This is the first such product.
## Generally

### How does HI-19 work?
  1. First, the received photo is segmented into words
  2. Afterwards, we need to fit the model. HIModel из model
  3. To work with HI-19 API we recommend to use RunEnv for run-time and FitEnv for fitting, both from hypo2.api

### Structure
  - config.py - project configuration file
  - model.py - implementation and functions of the HIModel
  - api.py

## Model

### First of all
First, we configure the project. Create file myconfig.py and insert the following code:
```python
from hypo2.config import RunConfig

class MyConfig(RunConfig):
    BACKUP_DIRECTORY = ???
    MODEL_PATH = ???
```
Substitute ??? with either the appropriate path or None. BACKUP_DIRECTORY is a path to the directory, where backup models will be saved. MODEL_PATH - a path to the final model.

Then, let us create the config variable. It is simple:
```python
from myconfig import MyConfig
config = MyConfig()
```
So, you can change some settings right in the run-time. To have a look at the configs just type
```
print(config)
```

### Fitting
The dataset we have should be structured in the following way:
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
In other words, every item of the array is an array of links to photos of one class. You have to change the config variable, at least, CLASS_COUNT.
Let us say we already have dataset.
Imporint api
```python
from hypo2.api import FitEnv
```
FitTime enables to fit the model easily. Creating an instance of FitEnv:
```python
fitenv = FitEnv(config)
```
The environment is ready to fit. FitEnv will create a model, process the input, and set the system. Let us fit the model:
```python
model = fitenv.fit(dataset, verbose=True, plot=False)
```
verbose is whether to output epoch results.
plot is whether to plot loss diagram.
model is the result of fitting.

#### Model methods
You can save your model via
```python
model.save()
```
The path the model will be saved to is config.MODEL_PATH

If you want to do it manually, specify the path (deprecated method)
```python
model.saveto("D:/HI.19")
```

To open the model, specify it in config.MODEL_PATH and then create an instance.
```python
config.MODEL_PATH = "D:/HI.19"
model = HIModel(config)
```
If you want to reopen the model
```python
model.open()
```
Or use a deprecated method:
```python
model.openfrom("HI.19")
```

### Usage

Importing and creating RunEnv
```python
from hypo2.api import RunEnv
runenv = RunEnv(config)
```
Let us try to get a Center by photo:
```python
image = runenv.open_image("johns_text.jpg")
cw = runenv.get_center(image)
assert cw is not None, "0 words found"
center, weight = cw
```
Where center is a vector of length config.FEATURES_COUNT, and weight is an integer, that is equal to the total amound of 
found words.
To compare two images and get the difference between them, use differ
```python
distance = runenv.differ(image1, image2)
```
If you already have centers, then you can find out the distance between them using function
```python
distance = runenv.dist(john_center, mary_center)
```
