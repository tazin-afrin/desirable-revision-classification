import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save_plot(history, modelpath):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(modelpath +'.png')
    plt.close()


def printPredictedResult(id_pred_folds, data_path, revision_name, isContext, isFeedback):
    output = []
    for id,val in id_pred_folds:
        label = 'Desirable' if val == 1 else 'Undesirable'
        output.append([id, val, label])
    model_ext = 'SimpleContext' if isContext else ('Feedback' if isFeedback else 'BERT')
    if isContext and isFeedback:
        model_ext = 'CntxFdbk'
    df = pd.DataFrame(np.array(output), columns=['ID_'+model_ext, 'Label2_'+model_ext+'Val','Label2_'+model_ext])
    df.to_csv(data_path+model_ext+'_'+revision_name+'_Predicted.csv', sep=',', encoding='utf-8')
