#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

# suggested imports
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from collections import defaultdict

import numpy as np
import tensorflow

import gensim
import transformers
import string
import random
import re

from typing import List

def tokenize(s):
    """
    a naive tokenizer that splits on punctuation and whitespaces.
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    candidates= []
    lemmas = wn.lemmas(lemma,pos)
    for l in lemmas:
        syn = l.synset()
        for syn_lemma in syn.lemmas():
            name = syn_lemma.name()
            if name != lemma:
                candidates.append(name)
    return set(candidates)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    candidates = defaultdict(int)
    lemmas = wn.lemmas(context.lemma,context.pos)
    for l in lemmas:
        syn = l.synset()
        for syn_lemma in syn.lemmas():
            if syn_lemma.name() != context.lemma:
                candidates[syn_lemma.name()]+=syn_lemma.count()
    ret = max(candidates,key=candidates.get)
    return ret # replace for part 2

def wn_simple_lesk_predictor(context : Context) -> str:
    overlap_list = {}
    stop_words = stopwords.words('english')
    sentence = context.left_context + context.right_context
    sentence_filtered = [word for word in sentence if word not in stop_words]
    for lemma in wn.lemmas(context.lemma, context.pos):
        syn = lemma.synset()
        definition = syn.definition()
        for ex in syn.examples():
            definition = definition + " " + ex
        for hypernym in syn.hypernyms():
            definition = definition + " " + hypernym.definition()
            for hypernym_ex in hypernym.examples():
                definition = definition + " " + hypernym_ex
        tokens_def = tokenize(definition)
        definition_filtered = [word for word in tokens_def if word not in stop_words]
        overlap_list[lemma] = (len(set(sentence_filtered) & set(definition_filtered)))

    max_overlap = max(overlap_list.values())
    keys = [key for key, value in overlap_list.items() if value == max_overlap]
    counts = {}
    for lemma in wn.lemmas(context.lemma, context.pos):
        if lemma in keys:
            for l in lemma.synset().lemmas():
                counts[l.name()] = l.count()

    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    replacement =''
    for i in list(sorted_counts.keys()):
        if i != context.lemma:
            replacement = i
            break
    return replacement #replace for part 3


class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self,context : Context) -> str:
        sim = {}
        ret = []
        ret = get_candidates(context.lemma, context.pos)
        for i in ret:
            try:
                sim[i] = self.model.similarity(i, context.lemma)
            except KeyError:
                continue
        sorted_sim = dict(sorted(sim.items(), key=lambda x: x[1], reverse=True))
        return list(sorted_sim.keys())[0] # replace for part 4


    def predict_context(self, context:Context) -> str:
        sim = defaultdict(int)
        ret = []
        stop_words = stopwords.words('english')
        left_context = context.left_context[-1]
        right_context = context.right_context[0]
        ret = get_candidates(context.lemma, context.pos)
        for i in ret:
            try:
                sim[i] += 0.5*self.model.similarity(i,context.lemma)
                sim[i] += 0.25*self.model.similarity(i,left_context)
                sim[i] += 0.25*self.model.similarity(i,right_context)
            except KeyError:
                continue
        sorted_sim = dict(sorted(sim.items(), key=lambda x: x[1], reverse=True))
        return list(sorted_sim.keys())[0]

class BertPredictor(object):

    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        best_word = ''
        candidates = get_candidates(context.lemma, context.pos)

        left_ctxt = ''
        for i in context.left_context:
            if i.isalpha():
                left_ctxt = left_ctxt + ' ' + i
            else:
                left_ctxt += i

        ctxt = left_ctxt + ' ' + '[MASK]'

        for j in context.right_context:
            if j.isalpha():
                ctxt = ctxt + ' ' + j
            else:
                ctxt += j

        input_toks = self.tokenizer.encode(ctxt)
        sent_tokenized = self.tokenizer.convert_ids_to_tokens(input_toks)
        mask_idx = sent_tokenized.index('[MASK]')

        input_mat = np.array(input_toks).reshape((1, -1)) # batch size
        outputs = self.model.predict(input_mat,verbose=0)
        predictions = outputs[0]

        best_probs = np.argsort(predictions[0][mask_idx])[::-1]
        best_words = self.tokenizer.convert_ids_to_tokens(best_probs)
        for word in best_words:
            if word.replace('_', ' ') in candidates:
                best_word = word.replace('_', ' ')
                break
        return best_word # replace for part 5



if __name__=="__main__":
    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    #bert = BertPredictor()
    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = predictor.predict_context(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
