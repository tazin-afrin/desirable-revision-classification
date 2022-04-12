from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
from pprint import pprint

def removeStopWords(sentence):
    filtered_words = [word.strip() for word in sentence.split(' ') if (word.strip() not in stopwords) and (word.strip() != '')]
    newSentence = ''
    for word in filtered_words:
        newSentence += word + ' '
    return newSentence.strip()

# returns count of gram
def CountNgrams(s1, ngram):
    s1 = re.sub(r'[^\w]', ' ', s1)
    s1 = s1.split()
    count = 0
    s1 = ngrams(s1, ngram)
    s = []
    for word in s1:
        s += list(word)
        count += 1
    print(s)
    return count

def double_quote(word):
    return '%s' % word

def CommonNgrams(s1, s2, ngram):
    s1 = word_tokenize(s1)#re.sub(r'[^\w]', ' ', s1)
    s2 = word_tokenize(s2)#re.sub(r'[^\w]', ' ', s2)

    #s1 = removeStopWords(s1).split(' ')
    #s2 = removeStopWords(s2).split(' ')
    # s1 = s1.split()
    # s2 = s2.split()

    s1 = ngrams(s1, ngram)
    s2 = ngrams(s2, ngram)
    s1 = [list(w) for w in s1]
    s2 = [list(w) for w in s2]

    count = set()
    s = []
    for word in s1:
        if word in s2:
            s += list(word)
            count.add(tuple(word))
    print(ngram)
    print('\n', count)

    # print('common  ', s)
    return len(count)

def UncommonNgrams(s1, s2, ngram):

    # s1 = re.sub(r'[^\w]', ' ', s1)
    # s2 = re.sub(r'[^\w]', ' ', s2)
    #
    # #s1 = removeStopWords(s1).split(' ')
    # #s2 = removeStopWords(s2).split(' ')
    # s1 = s1.split()
    # s2 = s2.split()

    s1 = word_tokenize(s1)
    # print(s1)
    s2 = word_tokenize(s2)
    # print(s2)

    s1 = ngrams(s1, ngram)
    s2 = ngrams(s2, ngram)
    s1 = [list(w) for w in s1]
    s2 = [list(w) for w in s2]

    count = 0
    # for word1 in s1:
    #     if not (word1 in s2):
    #         count += 1
    s = []
    ss = set()
    for word1 in s2:
        if not (word1 in s1):
            s += list(word1)
            ss.add(tuple(word1))
            count += 1
    print(ngram)
    # print('\nuncommon  ', s)
    print(ss)
    return count


def get_ngram_features(S1,S2):

    unigramS1 = CountNgrams(S1, 1)
    unigramS2 = CountNgrams(S2, 1)
    diffUnigrams = unigramS2 - unigramS1

    commonUnigrams = CommonNgrams(S1, S2, 1)
    uniqueUnigrams = UncommonNgrams(S1, S2, 1)

    commonBigrams = CommonNgrams(S1, S2, 2)
    uniqueBigrams = UncommonNgrams(S1, S2, 2)
    commonTrigrams = CommonNgrams(S1, S2, 3)
    uniqueTrigrams = UncommonNgrams(S1, S2, 3)

    features = [unigramS1, unigramS2, diffUnigrams, commonUnigrams, uniqueUnigrams, commonBigrams, uniqueBigrams, commonTrigrams, uniqueTrigrams]
    featureNames = ['uniS1', 'uniS2', 'uniDiff', 'uniCommon', 'uniUnique', 'biCommon', 'biUnique', 'triCommon', 'triUnique']
    return featureNames, features

def get_ngram_features_Article(S1,S2, article):

    unigramS1 = CountNgrams(S1, 1)
    unigramS2 = CountNgrams(S2, 1)
    diffUnigrams = unigramS2 - unigramS1

    commonUnigrams = CommonNgrams(S1, S2, 1)
    uniqueUnigrams = UncommonNgrams(S1, S2, 1)

    commonBigrams = CommonNgrams(S1, S2, 2)
    uniqueBigrams = UncommonNgrams(S1, S2, 2)
    commonTrigrams = CommonNgrams(S1, S2, 3)
    uniqueTrigrams = UncommonNgrams(S1, S2, 3)

    commonUnigramsS1A = CommonNgrams(S1, article, 1)
    commonUnigramsS2A = CommonNgrams(S2, article, 1)

    commonBigramsS1A = CommonNgrams(S1, article, 2)
    commonBigramsS2A = CommonNgrams(S2, article, 2)

    commonTrigramsS1A = CommonNgrams(S1, article, 3)
    commonTrigramsS2A = CommonNgrams(S2, article, 3)


    features = [unigramS1, unigramS2, diffUnigrams, commonUnigrams, uniqueUnigrams, commonBigrams, uniqueBigrams, commonTrigrams, uniqueTrigrams, commonUnigramsS1A, commonUnigramsS2A, commonBigramsS1A, commonBigramsS2A, commonTrigramsS1A, commonTrigramsS2A]
    featureNames = ['uniS1', 'uniS2', 'uniDiff', 'uniCommon', 'uniUnique', 'biCommon', 'biUnique', 'triCommon', 'triUnique', 'uniCommonS1A', 'uniCommonS2A', 'biCommonS1A', 'biCommonS2A', 'triCommonS1A', 'triCommonS2A']
    return featureNames, features

def get_ngram_features_Article_Essay(S1,S2, E1, E2, article):

    unigramS1 = CountNgrams(S1, 1)
    unigramS2 = CountNgrams(S2, 1)
    diffUnigrams = unigramS2 - unigramS1

    commonUnigrams = CommonNgrams(S1, S2, 1)
    uniqueUnigrams = UncommonNgrams(S1, S2, 1)

    commonBigrams = CommonNgrams(S1, S2, 2)
    uniqueBigrams = UncommonNgrams(S1, S2, 2)
    commonTrigrams = CommonNgrams(S1, S2, 3)
    uniqueTrigrams = UncommonNgrams(S1, S2, 3)

    commonUnigramsS1A = CommonNgrams(S1, article, 1)
    commonUnigramsS2A = CommonNgrams(S2, article, 1)

    commonBigramsS1A = CommonNgrams(S1, article, 2)
    commonBigramsS2A = CommonNgrams(S2, article, 2)

    commonTrigramsS1A = CommonNgrams(S1, article, 3)
    commonTrigramsS2A = CommonNgrams(S2, article, 3)

    commonUnigramsE1A = CommonNgrams(E1, article, 1)
    commonUnigramsE2A = CommonNgrams(E2, article, 1)

    commonBigramsE1A = CommonNgrams(E1, article, 2)
    commonBigramsE2A = CommonNgrams(E2, article, 2)

    commonTrigramsE1A = CommonNgrams(E1, article, 3)
    commonTrigramsE2A = CommonNgrams(E2, article, 3)


    features = [unigramS1, unigramS2, diffUnigrams, commonUnigrams, uniqueUnigrams, commonBigrams, uniqueBigrams, commonTrigrams, uniqueTrigrams,
                commonUnigramsS1A, commonUnigramsS2A, commonBigramsS1A, commonBigramsS2A, commonTrigramsS1A, commonTrigramsS2A,
                commonUnigramsE1A, commonUnigramsE2A, commonBigramsE1A, commonBigramsE2A, commonTrigramsE1A,
                commonTrigramsE2A]
    featureNames = ['uniS1', 'uniS2', 'uniDiff', 'uniCommon', 'uniUnique', 'biCommon', 'biUnique', 'triCommon', 'triUnique',
                    'uniCommonS1A', 'uniCommonS2A', 'biCommonS1A', 'biCommonS2A', 'triCommonS1A', 'triCommonS2A',
                    'uniCommonE1A', 'uniCommonE2A', 'biCommonE1A', 'biCommonE2A', 'triCommonE1A', 'triCommonE2A']
    return featureNames, features


if __name__ == '__main__':
    S1 = 'Technolgy is changing the world, and at particular the way we didn\'t communicate.'
    S3 = 'Technolgy is chnging the world, and in particular the way we communicate.'
    S2 = 'Technology is changing teh way we communicate.'
    S4 = "'Nobody talks face to face anymore' seems to be the mantra of the technology-wary."

    S1 = "So many people will be dead by the time they can do anything"
    S2 = "By the time it is 2025, a lot of the people will be sick and they can die if they don't get treated fast enough"

    featureNames, features = get_ngram_features(S1, S2)
    print(featureNames)
    print(features)