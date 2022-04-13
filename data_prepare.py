import pandas as pd
import os
from tokenizer import sentence_tokenizer
from collections import defaultdict


def preprocess_revision_pair(params):

    datafile = params.revision_name + 'SentencePairData.xlsx'
    if datafile.endswith(".xlsx"):
        file = os.path.join(params.data_path, datafile)
    data = pd.read_excel(file, engine='openpyxl')

    # extract revision pairs and labels from the datafile
    revision_pairs = list()
    ID = list()
    adm = list()
    data_label = list()
    purpose_label = list()
    maxlen = 0
    max_context_len = 0
    max_feedback_len = 0
    avg_context_len = []
    avg_feedback_len = []
    c, f = 0, 0
    for item, row in data.iterrows():
        id = row['ID']
        s1 = '' if str(row['S1']).lower() == 'nan' else str(row['S1'])
        s2 = '' if str(row['S2']).lower() == 'nan' else str(row['S2'])
        cur = [s1, s2]

        maxlen = max(maxlen, len(sentence_tokenizer(s1)), len(sentence_tokenizer(s2)))

        if params.isContext and 'ContextD1' in data.columns and 'ContextD2' in data.columns:
            context1 = '' if str(row['ContextD1']).lower() == 'nan' else str(row['ContextD1'])
            context2 = '' if str(row['ContextD2']).lower() == 'nan' else str(row['ContextD2'])

            l1 = len(sentence_tokenizer(context1))
            l2 = len(sentence_tokenizer(context2))
            max_context_len = max(max_context_len, l1, l2)
            if l1 > 512:
                c+=1
                # print('context1')
                # print('\n', s1,'\n',s2,'\n', context1, '\n\n')
            if l2 > 512:
                c+=1
                # print('context2')
                # print('\n', s1,'\n',s2,'\n', context2, '\n\n')
            if l1 != 0:
                avg_context_len.append(l1)
            if l2 != 0:
                avg_context_len.append(l2)

            cur += [context1, context2]
        if params.isFeedback:
            feedback = '' if str(row['Feedback']).lower() == 'nan' else str(row['Feedback'])
            cur += [feedback]

            max_feedback_len = max(max_feedback_len, len(sentence_tokenizer(feedback)))
            if len(sentence_tokenizer(feedback)) > 512:
                f+=1
            avg_feedback_len.append(len(sentence_tokenizer(feedback)))


        revision_pairs.append(cur)

        if str(row['Label2']) == params.revision_label:
            data_label.append(1)
        else:
            data_label.append(0)

        if 'RevisionPurpose' in data.columns:
            if str(row['RevisionPurpose']) == 'Evidence':
                purpose_label.append(1)
            else:
                purpose_label.append(0)

        if row['Add'] == 1:
            adm.append(1)
        elif row['Delete'] == 1:
            adm.append(2)
        else:
            adm.append(3)

        ID.append(id)

    if params.isFeedback:
        print('*******   Remember to set the max len for feedback in the model ****** --- done')
        print('max_feedback_len:', max_feedback_len)

        print('Feedback min max avg')
        print(min(avg_feedback_len), max(avg_feedback_len), sum(avg_feedback_len) / len(avg_feedback_len))

        print('** longer than 512: ', f)
    if params.isContext:
        print('Context min max avg')
        print(min(avg_context_len), max(avg_context_len), sum(avg_context_len)/len(avg_context_len))
        print('** longer than 512: ', c)


    return ID, adm, revision_pairs, data_label, maxlen

def read_multidata(data, params):
    datasets = defaultdict(defaultdict)
    data_path = params.data_path
    for i in range(len(data)):
        params.data_path = data_path + data[i] + '/'
        ID, adm, X, y, maxlen = preprocess_revision_pair(params)
        datasets[data[i]]['ID'] = ID
        datasets[data[i]]['adm'] = adm
        datasets[data[i]]['X'] = X
        datasets[data[i]]['y'] = y
        datasets[data[i]]['maxlen'] = maxlen
        params.data_path = data_path
    return datasets



