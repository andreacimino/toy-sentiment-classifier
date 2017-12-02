# toy-sentiment-classifier
This is small project showing how to build a basic document classifier.

In order to install the required dependencies first create a virtualenv
with the following command:

virtualenv venv

then activate the environment with this command:

source venv/bin/activate

Once the environment is activated, install the required dependencies
through this command:

pip install -r requirements


Training and Test datasets for sentipolc 2016 tasks available at the following url:

https://notebooks.azure.com/andreacimino/libraries/sentipolc-dataset

Run training with:

python classifier.py -t -i training.parsed -m model

Run test with:

python classifier.py  -i test.parsed -m model


Link to the Sentipolc 2016 Report:
http://ceur-ws.org/Vol-1749/paper_026.pdf
