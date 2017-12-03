# toy-sentiment-classifier
This is small project showing how to build a basic document classifier.

In order to install the required dependencies first create a virtualenv
with the following command:

```bash
virtualenv venv
```

then activate the environment with this command:


```bash
source venv/bin/activate
```


Once the environment is activated, install the required dependencies
through this command:

```bash
pip install -r requirements
```


Training and test datasets for Sentipolc 2016 tasks available at the following url:

https://notebooks.azure.com/andreacimino/libraries/sentipolc-dataset

Run training with:


```bash
python classifier.py -t -i training.parsed -m model
```

Run test with:

```bash
python classifier.py  -i test.parsed -m model
```

Link to the Sentipolc 2016 Report:
http://ceur-ws.org/Vol-1749/paper_026.pdf

