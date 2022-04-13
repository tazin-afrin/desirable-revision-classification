from collections import OrderedDict
from nltk import word_tokenize
from nltk.corpus import wordnet
import numpy as np

def find_synonyms(word):
    synonyms = []
    for synset in wordnet.synsets(word):
        for syn in synset.lemma_names():
            syn = syn.replace('_', ' ')  # restore space character
            synonyms.append(syn)

    # drop duplicate words (closest synonyms comes first)
    synonyms_set = list(OrderedDict.fromkeys(synonyms))
    return synonyms_set

def find_augmented_sentences(sentence, syn_per_word = 5):
    augmented_sentences = []
    for word in word_tokenize(sentence):
        if len(word) <= 5: continue
        for synonym in find_synonyms(word)[0:syn_per_word]:
            new_sentence = sentence.replace(word, synonym)
            augmented_sentences.append(new_sentence)
    return augmented_sentences

def find_augmented_sentences_new(sentence, syn_per_sent = 0.2, syn_per_word = 5):
    augmented_sentences = []
    for word in word_tokenize(sentence):
        if len(word) <= 5: continue
        syns = find_synonyms(word)#[0:syn_per_word]
        # print(word, syns)
        count = 0
        for synonym in syns:
            if synonym == word:
                continue
            # print(word, synonym)
            new_sentence = sentence.replace(word, synonym)
            augmented_sentences.append(new_sentence)
            count += 1
            if count == syn_per_word:
                break

    return augmented_sentences


def find_augmented_sentences_oversample(sentence, syn_per_sent = 0.2, syn_per_word = 5):
    augmented_sentences = []
    for word in word_tokenize(sentence):
        if len(word) <= 5: continue
        syns = find_synonyms(word)#[0:syn_per_word]
        # print(word, syns)
        count = 0
        for synonym in syns:
            if synonym == word:
                continue
            count += 1
            if count > syn_per_word:
                # print(word, synonym)
                new_sentence = sentence.replace(word, synonym)
                augmented_sentences.append(new_sentence)
            if count == syn_per_word+syn_per_word:
                break

    return augmented_sentences

def create_augmented_data(data, label, type = ['synonym', 'flip_position']):
    augmented_data = []
    augmented_label = []
    if 'synonym_new' in type:
        print('data augmenting using synonym replacement ...')
        for i in range(len(data)):
            s1 = data[i][0]
            s2 = data[i][1]
            y = label[i]

            if s1 != '':
                augmented_sentences = find_augmented_sentences(s1)
                if len(augmented_sentences) > 1:
                    for new_sentence in augmented_sentences:
                        augmented_data.append([new_sentence, s2])
                        augmented_label.append(y)

    if 'synonym' in type:
        # print('data augmenting using synonym replacement ...')
        for i in range(len(data)):
            s1 = data[i][0]
            s2 = data[i][1]
            y = label[i]

            if s1!= '':
                augmented_sentences = find_augmented_sentences(s1)
                if len(augmented_sentences) > 1:
                    for new_sentence in augmented_sentences:
                        augmented_data.append([new_sentence, s2])
                        augmented_label.append(y)

            if s2!= '':
                augmented_sentences = find_augmented_sentences(s2)
                if len(augmented_sentences) > 1:
                    for new_sentence in augmented_sentences:
                        augmented_data.append([s1, new_sentence])
                        augmented_label.append(y)


    if 'flip_position' in type:
        print('data augmenting using flipping added/deleted sentence position ...')
        for i in range(len(data)):
            s1 = data[i][0]
            s2 = data[i][1]
            y = label[i]
            if s1 == '' or s2 == '':
                augmented_data.append([s2, s1])
                augmented_label.append(y)

    return augmented_data, augmented_label


def create_augmented_data_balance(data, label, type = ['synonym', 'flip_position']):
    augmented_data = []
    augmented_label = []
    if 'synonym' in type:

        # print('data augmenting using synonym replacement ...')
        for i in range(len(data)):
            s1 = data[i][0]
            s2 = data[i][1]
            y = label[i]
            # print([s1, s2])
            if s1!= '':
                augmented_sentences = find_augmented_sentences_new(s1)
                if len(augmented_sentences) > 1:
                    for new_sentence in augmented_sentences:
                        augmented_data.append([new_sentence, s2])
                        # print([new_sentence, s2])
                        augmented_label.append(y)

            if s2!= '':
                augmented_sentences = find_augmented_sentences_new(s2)
                if len(augmented_sentences) > 1:
                    for new_sentence in augmented_sentences:
                        augmented_data.append([s1, new_sentence])
                        # print([s1, new_sentence])
                        augmented_label.append(y)


    alllabels = np.concatenate((label, np.array(augmented_label)),axis=0)
    l1 = sum(alllabels)
    l0 = len(alllabels) - sum(alllabels)
    label_diff = abs(l1-l0)/2
    l = 0 if l1>l0 else 1
    # print('label_diff : ', label_diff)

    # undersampling
    count, label_diff = 0, int(3*label_diff/4)
    indices = []
    for i in range(len(augmented_label)-1,-1,-1):
        if augmented_label[i] != l:
            indices.append(i)
            count += 1
            if count == label_diff:
                break
    for i in indices:
        del augmented_data[i]
        del augmented_label[i]
    # print('undersampled')
    return augmented_data, augmented_label


    # oversampling
    augmented_data_oversampled = []
    augmented_label_oversampled = []
    for i in range(len(data)):
        if label[i] != l:
            continue
        s1 = data[i][0]
        s2 = data[i][1]
        y = label[i]
        # print([s1, s2])
        if s1 != '':
            augmented_sentences = find_augmented_sentences_oversample(s1)
            if len(augmented_sentences) > 1:
                for new_sentence in augmented_sentences:
                    augmented_data_oversampled.append([new_sentence, s2])
                    # print([new_sentence, s2])
                    augmented_label_oversampled.append(y)

        if s2 != '':
            augmented_sentences = find_augmented_sentences_oversample(s2)
            if len(augmented_sentences) > 1:
                for new_sentence in augmented_sentences:
                    augmented_data_oversampled.append([s1, new_sentence])
                    # print([s1, new_sentence])
                    augmented_label_oversampled.append(y)

        if len(augmented_label_oversampled) > label_diff:
            break

    augmented_data = augmented_data + augmented_data_oversampled
    augmented_label = augmented_label + augmented_label_oversampled


    return augmented_data, augmented_label

def find_augmented_sentences_context(sentence, context, syn_per_word = 5):
    augmented_sentences = []
    for word in word_tokenize(sentence):
        if len(word) <= 5: continue
        for synonym in find_synonyms(word)[0:syn_per_word]:
            new_sentence = sentence.replace(word, synonym)
            new_context = context.replace(word,synonym)
            augmented_sentences.append([new_sentence, new_context])
    return augmented_sentences

def create_augmented_data_context(data, label, type = ['synonym', 'flip_position']):
    augmented_data = []
    augmented_label = []

    if 'synonym' in type:
        print('data augmenting using synonym replacement ...')
        for i in range(len(data)):
            s1 = data[i][0]
            s2 = data[i][1]
            c1 = data[i][2]
            c2 = data[i][3]
            y = label[i]

            if s1!= '':
                augmented_sentences = find_augmented_sentences_context(s1, c1)
                if len(augmented_sentences) > 1:
                    for new_sentence, new_context in augmented_sentences:
                        augmented_data.append([new_sentence, s2, new_context, c2])
                        augmented_label.append(y)

            if s2!= '':
                augmented_sentences = find_augmented_sentences_context(s2, c2)
                if len(augmented_sentences) > 1:
                    for new_sentence, new_context in augmented_sentences:
                        augmented_data.append([s1, new_sentence, c1, new_context])
                        augmented_label.append(y)


    return augmented_data, augmented_label

def create_augmented_data_feedback(data, label, type = ['synonym', 'flip_position']):
    augmented_data = []
    augmented_label = []
    if 'synonym' in type:
        print('data augmenting using synonym replacement ...')
        for i in range(len(data)):
            s1 = data[i][0]
            s2 = data[i][1]
            fb = data[i][2]
            y = label[i]

            if s1!= '':
                augmented_sentences = find_augmented_sentences(s1)
                if len(augmented_sentences) > 1:
                    for new_sentence in augmented_sentences:
                        augmented_data.append([new_sentence, s2, fb])
                        augmented_label.append(y)

            if s2!= '':
                augmented_sentences = find_augmented_sentences(s2)
                if len(augmented_sentences) > 1:
                    for new_sentence in augmented_sentences:
                        augmented_data.append([s1, new_sentence, fb])
                        augmented_label.append(y)

    return augmented_data, augmented_label

def create_augmented_data_context_feedback(data, label, type = ['synonym', 'flip_position']):
    augmented_data = []
    augmented_label = []

    if 'synonym' in type:
        print('data augmenting using synonym replacement ... Context and Feedback')
        for i in range(len(data)):
            s1 = data[i][0]
            s2 = data[i][1]
            c1 = data[i][2]
            c2 = data[i][3]
            fb = data[i][4]
            y = label[i]

            if s1!= '':
                augmented_sentences = find_augmented_sentences_context(s1, c1)
                if len(augmented_sentences) > 1:
                    for new_sentence, new_context in augmented_sentences:
                        augmented_data.append([new_sentence, s2, new_context, c2, fb])
                        augmented_label.append(y)

            if s2!= '':
                augmented_sentences = find_augmented_sentences_context(s2, c2)
                if len(augmented_sentences) > 1:
                    for new_sentence, new_context in augmented_sentences:
                        augmented_data.append([s1, new_sentence, c1, new_context, fb])
                        augmented_label.append(y)


    return augmented_data, augmented_label



if __name__ == "__main__":
    sentence = "They can do that by assuring that the people of Sauri, Kenya have food, water, liter, and a place to stay"
    sentence = "If the plans are going to be achieved in 2025 than their plans will be achieved in only 7 more years which would be in our life time"
    augsent = find_augmented_sentences(sentence)

    for sent in augsent:
        print(sent)