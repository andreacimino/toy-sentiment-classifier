from collections import Counter
import math

class FeatureExtractor(object):

  def extract_word_ngrams(self, doc, length):
    words = [x.word for x in doc.get_tokens()]
    char_counter = Counter()
    for i in range(0, len(words)-length+1):
      char_counter["%sWG_%s" % (length, "_".join(words[i: i+length]))] += 1
    return char_counter

  def extract_lemma_ngrams(self, doc, length):
    words = [x.lemma for x in doc.get_tokens()]
    char_counter = Counter()
    for i in range(0, len(words)-length+1):
      char_counter["%sWGL_%s" % (length, "_".join(words[i: i+length]))] += 1
    return char_counter

  def compute_n_chars(self, doc, length):
    tokens = doc.get_tokens()
    word_normal = " " .join([x.word for x in tokens])
    word = word_normal.lower()
    feats = []
    char_counter = Counter()
    for i in range(0, len(word)-length+1):
        value = "NC_" + word[i:i+length]
        word_val_normal = "NRM_" + word_normal[i:i+length]
        char_counter[value] += 1
        char_counter[word_val_normal] += 1
    float_len_word = float(len(word))
    feats = {}
    for x in char_counter:
      val = math.log(char_counter[x]+1)/float_len_word
      feats[x] = val
    return feats
