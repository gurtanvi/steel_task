Task description
----------------
The file task_data.csv contains an example data set that has been artificially
generated. The set consists of 400 samples where for each sample there are 10
different sensor readings available. The samples have been divided into two
classes where the class label is either 1 or -1. The class labels define to what
particular class a particular sample belongs.

Your task is to rank the sensors according to their importance/predictive power
with respect to the class labels of the samples. Your solution should be a
Python script or a Jupyter notebook file that generates a ranking of the sensors
from the provided CSV file. The ranking should be in decreasing order where the
first sensor is the most important one.

Additionally, please include an analysis of your method and results, with
possible topics including:

* your process of thought, i.e., how did you come to your solution?
* properties of the artificially generated data set
* strengths of your method: why does it produce a reasonable result?
* weaknesses of your method: when would the method produce inaccurate results?
* scalability of your method with respect to number of features and/or samples
* alternative methods and their respective strengths, weaknesses, scalability

## Prerequisites

Before you begin, ensure you have met the following requirements:

* Python \(version 3.7\)

## Local setup

* Clone the repo

```text
git clone https://github.com/gurtanvi/steel_task
```

* Install the dependencies

```text
pip install -r requirements.txt
```

* Execute the code

```text
python sensor.py
```

## 
