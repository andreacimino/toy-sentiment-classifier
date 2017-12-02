import argparse
import re
import codecs
from feature_extractor import FeatureExtractor
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import sklearn.metrics
import cPickle as pickle
from evaluator import evaluate
import csv


class Sentence:

  def __init__(self):
    self.tokens = []

  def add_token(self, token):
    self.tokens.append(token)

class Document(object):

  def __init__(self):
    self.sentences = []

  def add_sentence(self, sentence):
    self.sentences.append(sentence)

  def get_tokens(self):
    return [token for sentence in self.sentences for token in sentence.tokens]

class Token:

  def __init__(self, word, lemma, pos, cpos):
    self.word = word
    self.lemma = lemma
    self.cpos = cpos
    self.pos = pos

class InputReader(object):

  def __init__(self, input_file_name):
    self.current_sentence = Sentence()
    self.input_file_name = input_file_name

  def generate_documents(self):
    current_document = Document()
    current_sentence = Sentence()
    self.input_file = codecs.open(self.input_file_name, 'r', 'utf-8')
    while True:
      l = self.input_file.readline()
      if l == '\n':
        current_document.add_sentence(current_sentence)
        current_sentence = Sentence()
      elif "<doc" in l:
        current_document = Document()

        # Read ID
        if "id" in l:
          current_document.id = re.compile('id="(.*?)"').findall(l)[0]
        else:
          current_document.id = "0"

        # Read labels
        is_positive, is_negative = False, False
        if ' pos="1"' in l or 'opos="1"' in l or "positive" in l:
          is_positive = True
        if ' neg="1"' in l or 'oneg="1"' in l or "negative" in l:
          is_negative = True
        if is_positive and is_negative:
          current_document.label = "POS_NEG"
        elif is_positive:
          current_document.label = "POS"
        elif is_negative:
          current_document.label = "NEG"
        else:
          current_document.label = "O"
      elif "</doc>" in l:
        yield current_document
      elif l == '':
        raise StopIteration
      else:
        split_token = l.rstrip('\n').split("\t")
        tok = Token(split_token[1], split_token[2],
                    split_token[3], split_token[4])
        current_sentence.add_token(tok)


class ToySentimentClassifier(object):

  def __init__(self):
    self.feature_extractor = FeatureExtractor()

  def extract_features(self, doc):
    all_features = {}
    for i in range(1, 3):
      all_features.update(self.feature_extractor.extract_word_ngrams(doc, i))
    for i in range(1, 3):
      all_features.update(self.feature_extractor.extract_lemma_ngrams(doc, i))
    for i in range(1, 3):
      all_features.update(self.feature_extractor.compute_n_chars(doc, i))
    all_features.update(self.feature_extractor.compute_document_length(doc))
    return all_features

  def train(self, model_name, input_file_name):
    reader = InputReader(input_file_name)
    all_docs = []
    for doc in reader.generate_documents():
      doc.features = self.extract_features(doc)
      all_docs.append(doc)

    # Encoding of samples
    all_collected_feats = [doc.features for doc in all_docs]
    X_dict_vectorizer = DictVectorizer(sparse=True)
    encoded_features = X_dict_vectorizer.fit_transform(all_collected_feats)

    # Scale to increase performances and reduce training time
    scaler = preprocessing.StandardScaler(with_mean=False).fit(encoded_features)
    encoded_scaled_features = scaler.transform(encoded_features)

    # Encoding of labels
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit([doc.label for doc in all_docs])
    encoded_labels = label_encoder.transform([doc.label for doc in all_docs])

    # Classifier Algorithm
    clf = LinearSVC()

    # Cross validation
    cross_val_scores = cross_val_score(clf, encoded_scaled_features,
                                       encoded_labels, scoring='f1_weighted')
    print "Average F1 Weighted: %s" % (reduce(lambda x, y: x + y, cross_val_scores) / len(cross_val_scores),)

    clf.fit(encoded_scaled_features, encoded_labels)

    # Save model
    joblib.dump(clf, 'clf.pkl')
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    pickle.dump(X_dict_vectorizer, open("vectorizer.pickle", "wb"))

  def evaluate_sentipolc(self, docs):
    def clz_to_opos_oneg(clz):
      if clz == "POS":
        opos = 1
        oneg = 0
      if clz == "NEG":
        opos = 0
        oneg = 1
      if clz == "O":
        opos = 0
        oneg = 0
      if clz == "POS_NEG":
        opos = 1
        oneg = 1
      return (opos, oneg)

    predicted_csv_file = open("predicted.csv", 'w')
    field_names = ["id", "sub", "opos", "oneg", "iro", "lpos", "lneg", "top"]
    writer = csv.DictWriter(predicted_csv_file, fieldnames=field_names)
    for doc in docs:
      opos, oneg = clz_to_opos_oneg(doc.labeled_prediction)
      writer.writerow({'id': doc.id, 'opos': opos, 'oneg': oneg})
    predicted_csv_file.close()

    # Generate gold file
    gold_csv_file = open("gold.csv", 'w')
    writer = csv.DictWriter(gold_csv_file, fieldnames=field_names)
    for doc in docs:
      opos, oneg = clz_to_opos_oneg(doc.label)
      writer.writerow({'id': doc.id, 'opos': opos, 'oneg': oneg})
    gold_csv_file.close()

    # Evaluation
    evaluate("gold.csv", "predicted.csv")

  def parse(self, model_name, input_file_name):
    classifier = joblib.load('clf.pkl')
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    vectorizer = joblib.load("vectorizer.pickle", "wb")
    reader = InputReader(input_file_name)
    all_docs = []
    original_labels = []
    predicted_labels = []
    for doc in reader.generate_documents():
      doc.features = self.extract_features(doc)
      all_docs.append(doc)

      # Encoding of samples
      encoded_features = vectorizer.transform(doc.features)
      encoded_scaled_features = scaler.transform(encoded_features)
      predictions = classifier.predict(encoded_scaled_features)
      labeled_prediction = label_encoder.inverse_transform(predictions)[0]
      original_labels.append(doc.label)
      predicted_labels.append(labeled_prediction)
      doc.labeled_prediction = labeled_prediction

    # print sklearn.metrics.confusion_matrix(original_labels,
    #                                        predicted_labels)
    print sklearn.metrics.classification_report(original_labels,
                                                predicted_labels)
    self.evaluate_sentipolc(all_docs)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Sentiment Classifier')
  parser.add_argument('-i','--input_file', help='The input file in CONLL Format', required=True)
  parser.add_argument('-m','--model_name', help='The model name', required=True)
  parser.add_argument('-o','--output_file', help='The output file')
  parser.add_argument('-t','--train', help='Trains the model', action='store_true')
  args = parser.parse_args()
  input_file = args.input_file
  output_file = args.output_file
  model_name = args.model_name
  train_mode = args.train
  classifier = ToySentimentClassifier()
  if train_mode:
    classifier.train(model_name, input_file)
  else:
    classifier.parse(model_name, input_file)
