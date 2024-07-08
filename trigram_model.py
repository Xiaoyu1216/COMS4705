import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2023
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile,'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1
    """

    if len(sequence) == 0:
        return sequence

    if n == 1:
        new_sequence = ['START']
    else:
        new_sequence = ['START']*(n-1)
    for i in range(len(sequence)):
        new_sequence.append(sequence[i])
    new_sequence.append('STOP')

    ngrams = []
    for idx, word in enumerate(new_sequence):
        ngram = tuple(new_sequence[idx:idx +n])
        ngrams.append(ngram)
        if ngram[n-1] == 'STOP':
            return ngrams
    return ngrams


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        word_count = sum(self.unigramcounts.values())
        self.denominator = word_count - self.unigramcounts[('START',)]

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """

        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ##Your code here
        for sentence in corpus:
            unigrams = get_ngrams(sentence,1)
            bigrams = get_ngrams(sentence,2)
            trigrams = get_ngrams(sentence,3)
            for gram in unigrams:
                self.unigramcounts[gram] += 1
            for gram in bigrams:
                self.bigramcounts[gram] += 1
            for gram in trigrams:
                self.trigramcounts[gram] += 1


        return None

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        # unseen bigrams

        if self.bigramcounts[trigram[0:2]] == 0:
            prob = self.unigramcounts[trigram[2]]/self.denominator
        else:
            numerator = self.trigramcounts[trigram]
            denominator = self.bigramcounts[trigram[0:2]]
            prob = numerator/denominator
        return prob

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        numerator = self.bigramcounts[bigram]
        denominator = self.unigramcounts[bigram[0:1]]
        prob = numerator/denominator
        return prob

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        numerator = self.unigramcounts[unigram]
        prob = numerator/self.denominator
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.
        return prob

    def generate_sentence(self,t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        uni_prob = self.raw_unigram_probability(trigram[2:])
        bi_prob = self.raw_bigram_probability(trigram[1:])
        tri_prob = self.raw_trigram_probability(trigram)
        smooth_prob = lambda1 * tri_prob + lambda2 * bi_prob + lambda3 * uni_prob
        return smooth_prob

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        trigrams = get_ngrams(sentence,3)
        log_prob = 0.0
        for trigram in trigrams:
            log_prob  = log_prob + math.log2(self.smoothed_trigram_probability(trigram))
        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """
        total_prob = 0.0
        M = 0.0
        for sentence in corpus:
            total_prob += self.sentence_logprob(sentence)
            # Add stop occurances
            M += len(sentence) +1
        l = total_prob/M
        return 2**(-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
        # training_file1 is train_high, file2 is train_low
        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0

        for f in os.listdir(testdir1):
            total += 1
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp1 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp < pp1:
                correct +=1
            # ..

        for f in os.listdir(testdir2):
            total += 1
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp < pp2:
                correct +=1
            # ..

        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.


    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment:
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)
