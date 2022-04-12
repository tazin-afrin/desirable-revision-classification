from nltk.tag import pos_tag, map_tag
from nltk.tokenize import word_tokenize
from collections import Counter
import re, pickle, language_check, enchant, codecs
from textstat.textstat import textstat

from nltk.corpus import wordnet as WN
from nltk.corpus import stopwords

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
from nltk import Tree
# from LoadData import LoadData

stop_words_en = set(stopwords.words('english'))
dictionary = enchant.Dict("en_US")
tool = language_check.LanguageTool('en-US')

tagset = ['NOUN', 'DET', 'ADJ', 'ADP', '.', 'VERB', 'CONJ', 'NUM', 'ADV', 'PRT', 'PRON', 'X']
    #         [ 0       1       2     3     4       5      6       7     8     9       10    11]

def SpellingMistakes(sentence):
    count = 0
    for word in sentence:
        #print(word)
        if dictionary.check(word.lower()) == False:
            count += 1
    return count

def SpellingMistakeCount(sentence):
    sentence = re.sub(r'[^\w]', ' ', sentence)
    #print(sentence)
    sentence = sentence.split()
    count = SpellingMistakes(sentence)
    return count

def LanguageMistakeCount(sentence):
    #sentence = re.sub(r'[^\w]', ' ', sentence)
    errors = tool.check(sentence)
    return len(errors)

def countPOS(sentence):
    postags = pos_tag(word_tokenize(sentence))
    simplifiedTags = [(word, map_tag('brown', 'universal', tag)) for word, tag in postags]
    return Counter([j for i, j in simplifiedTags])

def POSFeature(S1,S2):

    tagsS1 = countPOS(S1)
    tagsS2 = countPOS(S2)

    countTagS1 = []
    countTagS2 = []
    for tag in tagset:
        countTagS1.append(tagsS1[tag])
        countTagS2.append(tagsS2[tag])

    diffTags = ([i - j for i, j in zip(countTagS1, countTagS2)])

    #ratio of POS tags
    ratioTagS1 = [round(i/sum(countTagS1),3) for i in countTagS1]
    ratioTagS2 = [round(i/sum(countTagS2),3) for i in countTagS2]
    diffRatioTags = [round(i - j,3) for i, j in zip(ratioTagS1, ratioTagS2)]

    # 12 * 6
    #return countTagS1+countTagS2+diffTags+ratioTagS1+ratioTagS2+diffRatioTags
    return diffTags#+diffRatioTags

def Readability(sentence):
    return textstat.flesch_reading_ease(sentence)

def tokens(sent):
    return word_tokenize(sent)

def Check_POS_change(s1,s2, pos):
    #print(countPOS(s1))
    postags1 = pos_tag(word_tokenize(s1))
    #postags2 = pos_tag(word_tokenize(s2))
    #simpleTags1 = [(word, map_tag('brown', 'universal', tag)) for word, tag in postags1]
    #simpleTags2 = [(word, map_tag('brown', 'universal', tag)) for word, tag in postags2]

    count = 0
    for word,tag in postags1:
        if tag == pos:
            if word in tokens(s2):
                continue
            else:
                count = 1

    return count
def POS_diff(s1,s2, pos):
    return (countPOS(s1)[pos] - countPOS(s1)[pos])


def SpellChecker(sent):
    count = 0
    sent = removePunct(sent.lower())
    #print(sent)
    for i in tokens(sent):
        strip = i.rstrip()
        if not WN.synsets(strip):
            if strip in stop_words_en:  # <--- Check whether it's in stopword list
                continue
            else:
                count += 1
    return count
def removePunct(str):
    return "".join(c for c in str if c not in ('!', '.', ':', ','))

def tree_features(cfg, outfile):
    tree_features = []
    i = 0
    for trees in cfg:
        i += 1
        print(i)
        t1 = Tree.fromstring(trees[0])
        t2 = Tree.fromstring(trees[1])
        features = 15 * [0]
        for subtree in t1.subtrees():
            if subtree.label() == "SBAR":
                features[0] += 1
            elif subtree.label() == "VP":
                features[1] += 1
            elif subtree.label() == "NP":
                features[2] += 1
            features[3] += 1  # number of subtrees
        features[4] = t1.height()
        for subtree in t2.subtrees():
            if subtree.label() == "SBAR":
                features[5] += 1
            elif subtree.label() == "VP":
                features[6] += 1
            elif subtree.label() == "NP":
                features[7] += 1
            features[8] += 1  # number of subtrees
        features[9] = t2.height()

        # diff
        features[10] = features[0] - features[5]
        features[11] = features[1] - features[6]
        features[12] = features[2] - features[7]
        features[13] = features[3] - features[8]
        features[14] = features[4] - features[9]
        tree_features.append(features)

    print(tree_features[0])

    pickle.dump(tree_features, open(outfile, "wb"))

def parser():
    writeDataS1 = codecs.open('./pickled/study2_s1_cfg', 'w', 'utf-8')
    writeDataS2 = codecs.open('./pickled/study2_s2_cfg', 'w', 'utf-8')
    i = 0
    # revision_data = pickle.load(open("revision_data_undersample312.p", "rb"))
    revision_data = LoadData('../../Data/annotated revisions study2.csv')
    for data in revision_data:
        i += 1
        print(i)
        output = nlp.annotate(data[0], properties={'annotators': 'parse','outputFormat': 'json'})
        tree1 = ' '.join(str(output['sentences'][0]['parse']).split())
        output = nlp.annotate(data[1], properties={'annotators': 'parse','outputFormat': 'json'})
        tree2 = ' '.join(str(output['sentences'][0]['parse']).split())

        writeDataS1.write(tree1 + '\n')
        writeDataS2.write(tree2 + '\n')

    writeDataS1.flush()
    writeDataS2.flush()
    writeDataS1.close()
    writeDataS2.close()
    print('Done')

def get_language_features(S1,S2):
    # Language features
    # POSFeatureVector = POSFeature(S1, S2)
    SpellErrorS1 = SpellingMistakeCount(S1)
    SpellErrorS2 = SpellingMistakeCount(S2)
    SpellDiff = SpellErrorS2 - SpellErrorS1
    LangErrorS1 = LanguageMistakeCount(S1)
    LangErrorS2 = LanguageMistakeCount(S2)
    LangErrorDiff = LangErrorS2 - LangErrorS1

    # IN_change = Check_POS_change(S1, S2, 'IN')
    # pos_diff = []
    # for tag in tagset:
    #     pos_diff.append(POS_diff(S1, S2, tag))
    # adj = POS_diff(S1, S2, 'ADJ')
    # adp = POS_diff(S1, S2, 'ADP')
    # adv = POS_diff(S1, S2, 'ADV')
    # conj = POS_diff(S1, S2, 'CONJ')
    # POSFeatureVector = [IN_change] + pos_diff #, adj, adp, adv, conj]

    #readS1 = Readability(S1)
    #readS2 = Readability(S2)

    features = [SpellErrorS1, SpellErrorS2, SpellDiff, LangErrorS1, LangErrorS2,
                        LangErrorDiff] #+ POSFeatureVector #+ [readS1, readS2, readS2 - readS1]
    featureNames = ['SpellErrorS1', 'SpellErrorS2', 'SpellDiff', 'LangErrorS1', 'LangErrorS2',
                        'LangErrorDiff']
    return featureNames, features

if __name__ == '__main__':
    S1 = 'Technolgy is chnging the world, and at particular the way we communicate.'
    S3 = 'Technolgy is chnging the world, and in particular the way we communicate.'
    S2 = 'Technology is changing teh way we communicate.'
    s1 = "'Nobody talks face to face anymore' seems to be the mantra of the technology-wary."

    featureNames, features = get_language_features(S1, S3)
    print(featureNames)
    print(features)
    # parser()
    # cfg1 = []
    # with codecs.open('./pickled/study2_s1_cfg', 'r', 'utf-8') as f:
    #     for line in f:
    #         cfg1.append(line)
    # cfg2 = []
    # with codecs.open('./pickled/study2_s2_cfg', 'r', 'utf-8') as f:
    #     for line in f:
    #         cfg2.append(line)
    # cfg = list(zip(cfg1,cfg2))
    # print(len(cfg))
    #
    # tree_features(cfg, './pickled/study2_cfg_features.p')

    # #POSFeatureVector = POSFeature(S1, S2)
    # SpellErrorS1 = SpellingMistakeCount(S1)
    # SpellErrorS2 = SpellingMistakeCount(S2)
    # SpellDiff = SpellErrorS2 - SpellErrorS1
    # LangErrorS1 = LanguageMistakeCount(S1)
    # LangErrorS2 = LanguageMistakeCount(S2)
    # LangErrorDiff = LangErrorS2 - LangErrorS1
    #
    # IN_change = Check_POS_change(S1, S2, 'IN')
    #
    # pos_diff = []
    # for tag in tagset:
    #     pos_diff.append(POS_diff(S1, S2, tag))
    # adj = POS_diff(S1, S2, 'ADJ')
    # adp = POS_diff(S1, S2, 'ADP')
    # adv = POS_diff(S1, S2, 'ADV')
    # conj = POS_diff(S1, S2, 'CONJ')
    # POSFeatureVector = [IN_change, adj,adp,adv,conj]
    #
    # LanguageFeatures = [SpellErrorS1, SpellErrorS2, SpellDiff, LangErrorS1, LangErrorS2,
    #                     LangErrorDiff] + POSFeatureVector
    #
    # print(LanguageFeatures)