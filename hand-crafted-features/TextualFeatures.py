from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import re
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
import scipy
import numpy as np
from numpy import mean
import codecs
import Levenshtein
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
# from LoadData import Load_AESW_Data

from nltk.tag import StanfordNERTagger
# english.all.7class.distsim.crf.ser.gz
# english.muc.7class.distsim.crf.ser.gz
# st_ner = StanfordNERTagger('/Users/tazinafrin/Documents/stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz',
#                            '/Users/tazinafrin/Documents/stanford-ner/stanford-ner.jar',
#                            encoding='utf-8')

def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    return continuous_chunk

# Count of number of Named Entity in S2-S1
def NamedEntityDiff(s1,s2):
    a1 = get_continuous_chunks(s1)
    a2 = get_continuous_chunks(s2)
    l1 = len(a1)
    l2 = len(a2)
    return l2-l1

def NER_Count(text):
    tokenized_text = word_tokenize(text)
    classified_text = st_ner.tag(tokenized_text)
    #print(classified_text)
    # Location, Person, Organization, Money, Percent, Date, Time
    count = [0, 0, 0, 0, 0, 0, 0]

    for word,tag in classified_text:
        if tag == 'LOCATION':
            count[0] += 1
        elif tag == 'PERSON':
            count[1] += 1
        elif tag == 'ORGANIZATION':
            count[2] += 1
        elif tag == 'MONEY':
            count[3] += 1
        elif tag == 'PERCENT':
            count[4] += 1
        elif tag == 'DATE':
            count[5] += 1
        elif tag == 'TIME':
            count[6] += 1
    return count

def getCapitalizedWords(sentence):
    pattern = re.compile(r'[A-Z0-9\\.]+')
    words = word_tokenize(sentence)#sentence.split(' ')
    print(words)
    importantWords = []
    for word in words:
        word = word.strip()
        if word == '' or len(word) < 2:
            continue
        if pattern.match(word):
            matchLength = pattern.match(word).span()[1]
            if (matchLength == len(word) or matchLength == 1) and (word not in importantWords):
                importantWords.append(word)
    return importantWords

def CapitalizedWordDiff(s1,s2):
    #a1 = getCapitalizedWords(s1)
    #a2 = getCapitalizedWords(s2)
    s1 = word_tokenize(s1)
    s2 = word_tokenize(s2)
    w1 = [word for word in s1 if word[0].isupper()]
    w2 = [word for word in s2 if word[0].isupper()]
    l1 = len(w1)
    l2 = len(w2)
    return l2-l1

def NumberOfDigitsDiff(s1,s2):
    a1 = re.findall(r'\d+', s1)
    l1 = 0
    for a in a1:
        l1 += len(a)
    a2 = re.findall(r'\d+', s2)
    l2 = 0
    for a in a2:
        l2 += len(a)
    return l2-l1

def CosineSimilarity(s1,s2):
    # word-lists to compare
    s1 = re.sub(r'[^\w]', ' ', s1)
    s2 = re.sub(r'[^\w]', ' ', s2)
    s1 = s1.split()
    s2 = s2.split()

    # count word occurrences
    s1_vals = Counter(s1)
    s2_vals = Counter(s2)

    # convert to word-vectors
    words = list(s1_vals.keys() | s2_vals.keys())
    vect1 = [s1_vals.get(word, 0) for word in words]  # [0, 0, 1, 1, 2, 1]
    vect2 = [s2_vals.get(word, 0) for word in words]  # [1, 1, 1, 0, 1, 0]

    # find cosine
    # len1 = sum(av * av for av in vect1) ** 0.5  # sqrt(7)
    # len2 = sum(bv * bv for bv in vect2) ** 0.5  # sqrt(4)
    # dot = sum(av * bv for av, bv in zip(vect1, vect2))  # 3
    # cosine = dot / (len1 * len2)  # 0.5669467
    return cosine_similarity(np.asarray(vect1).reshape(1, -1), np.asarray(vect2).reshape(1, -1))[0][0]
    # return 1 - spatial.distance.cosine(vect1, vect2)

def bleuScore(s1,s2):
    s1 = re.sub(r'[^\w]', ' ', s1)
    s2 = re.sub(r'[^\w]', ' ', s2)
    reference = s1.split()
    hypothesis = s2.split()

    return sentence_bleu([reference], hypothesis)

#Computation of Kullback-Leibler (KL) distance between text
def KL(s1,s2):
    # word-lists to compare
    s1 = re.sub(r'[^\w]', ' ', s1)
    s2 = re.sub(r'[^\w]', ' ', s2)
    s1 = s1.split()
    s2 = s2.split()

    # count word occurrences
    s1_vals = Counter(s1)
    s2_vals = Counter(s2)

    # convert to word-vectors
    words = list(s1_vals.keys() | s2_vals.keys())
    vect1 = [s1_vals.get(word, 0) for word in words]  # [0, 0, 1, 1, 2, 1]
    vect2 = [s2_vals.get(word, 0) for word in words]  # [1, 1, 1, 0, 1, 0]

    vect1 = [0.001 if x == 0 else x for x in vect1]
    vect2 = [0.001 if x == 0 else x for x in vect2]
    #print(vect1, vect2)
    return scipy.stats.entropy(vect2, vect1, base=None)

def avgWordLength(sentence):
    sentence = re.sub(r'[^\w]', ' ', sentence)
    sentence = sentence.split()
    return round(mean([len(x) for x in sentence]),3)

def numSymbols(sentence):
    v = len([x for x in sentence.split() if not x.isalnum() and x!=" "])
    return v

def speciteller(filename):
    f = codecs.open(filename, "r", encoding='utf-8')
    specificity = []
    for line in f:
        specificity.append(line.replace('\n','').replace('\r','').strip())
    return specificity

def get_textual_features(S1,S2):
    # Textual features
    l1 = len(S1)
    l2 = len(S2)
    lenDiff = l2 - l1

    #NEdiff = [x - y for x, y in zip(NER_Count(S2), NER_Count(S1))]  # NamedEntityDiff(S1, S2)

    CapWordDiff = CapitalizedWordDiff(S1, S2)
    NumDiff = NumberOfDigitsDiff(S1, S2)
    CommaDiff = S2.count(',') - S1.count(',')

    LevenDist = round(Levenshtein.distance(S1, S2), 3)
    CosSim = round(CosineSimilarity(S1, S2), 3)
    bleu = bleuScore(S1, S2)
    kl = KL(S1, S2)
    symS1 = len(re.findall('[^A-Za-z0-9,. ]', S1))  # numSymbols(S1)
    symS2 = len(re.findall('[^A-Za-z0-9,. ]', S2))  # numSymbols(S2)

    symDiff = symS2 - symS1


    features = [l1, l2, lenDiff, CapWordDiff, NumDiff, CommaDiff, LevenDist, CosSim, bleu, kl,
                       symS1, symS2, symDiff] #+ NEdiff
    featureNames = ['l1', 'l2', 'lenDiff', 'CapWordDiff', 'NumDiff', 'CommaDiff', 'LevenDist', 'CosSim', 'bleu', 'kl',
                       'symS1', 'symS2', 'symDiff']
    return featureNames, features

if __name__ == '__main__':
    S1 = 'Technology is changing the world, and in particular the way we communicate, $500.'
    S2 = 'Technology is changing the way #we communicate, $5000.'

    text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'


    # l1 = len(S1)
    # l2 = len(S2)
    # l = l2 - l1
    #
    # # NEdiff = [x - y for x, y in zip(NER_Count(S2), NER_Count(S1))]  # NamedEntityDiff(S1, S2)
    #
    # CapWordDiff = CapitalizedWordDiff(S1, S2)
    # NumDiff = NumberOfDigitsDiff(S1, S2)
    # CommaDiff = S2.count(',') - S1.count(',')
    #
    # LevenDist = round(Levenshtein.distance(S1, S2), 3)
    # CosSim = CosineSimilarity(S1, S2)
    CosSim = round(CosineSimilarity(S1, S2), 3)
    print(CosSim)
    # bleu = bleuScore(S1, S2)
    # kl = KL(S1, S2)
    # symS1 = len(re.findall('[^A-Za-z0-9,. ]',S1))#numSymbols(S1)
    # symS2 = len(re.findall('[^A-Za-z0-9,. ]',S2))#numSymbols(S2)
    #
    # symDiff = symS2 - symS1
    #
    # TextualFeatures = [l1, l2, l, CapWordDiff, NumDiff, CommaDiff, LevenDist, CosSim, bleu, kl,
    #                    symS1, symS2, symDiff] + NEdiff
    # print(TextualFeatures)
    #specificity = speciteller('./speciteller-master/study2_s1_312.probs')


