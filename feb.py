from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pickle
import os
import librosa
import numpy as np
import numpy
from math import pi
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import joblib


#====================================================================================
#--------------------------------------Functions----------------------------------
#====================================================================================


def normalize(k):
    mxzc = max(k)
    divisor = mxzc
    for i in range(len(k)):
        k[i] = k[i]/divisor

def normalize_freq(freq):
    max_freq = freq[-1]
    for i in range(len(freq)):
        freq[i] /= max_freq
    return freq

def check_bicoh(bicoh):
    for i in range(len(bicoh)):
        for j in range(len(bicoh[0])):
            if bicoh[i][j].real == float('inf'):
                bicoh[i][j] = 1
            if bicoh[i][j] != bicoh[i][j]:
                bicoh[i][j] = 0
    return bicoh

def N_peaks(bicoh):
    peaks = 0
    for i in range(1, len(bicoh)-1):
        if (bicoh[i-1] < bicoh[i]) and (bicoh[i] > bicoh[i+1]):
            peaks += 1
    return peaks

def DOV(bicoh):
    n = len(bicoh)
    Np = N_peaks(bicoh)
    if Np == 0:
        return 0
    mean = sum(bicoh)/n
    first = 0
    last = 0
    for i in range(n):
        if bicoh[i]-mean > 0:
            bicoh[i] = 1
            last = i
        else:
            bicoh [i] = 0
    dist_peaks = 0
    curr = 0
    peak_seen = 0
    while (1):
        if bicoh[curr] == 1:
            peak_seen = 1
            first = curr
            break
        curr += 1
    return (last-first)/Np

def VOV(bicoh):
    return N_peaks(bicoh)*np.var(bicoh)

def Truth_value(audio, hop_length, samples):
    word_file = audio[:-3]
    word_file += 'WRD'
    epsilon = 400
    #print(word_file, flush=True)
    file = open(word_file,"r")
    lines = file.readlines()
    truth = []
    words = []
    for line in lines:
        w_start, w_end, w = line.split()
        print(w_start, w_end, w, flush=True)
        w_start = int(w_start)
        w_end = int(w_end)
        words.append([w_start, w_end])
    start = 0
    end = 400
    i = 0
    while start <= samples:
        if i == len(words):
            while start <= samples:
                truth.append(1)                         #changed
                start += hop_length
            break
        w_start = words[i][0]
        w_end = words[i][1]
        while end < w_start - epsilon:
            truth.append(1)                             #changed
            start += hop_length
            end += hop_length
        while start < w_start + epsilon:
            truth.append(1)
            start += hop_length
            end += hop_length
        while end < w_end - epsilon:
            truth.append(0)
            start += hop_length
            end += hop_length
        while start < w_end + epsilon:
            truth.append(1)
            start += hop_length
            end += hop_length
        i += 1
    return truth

def find_pitch(audio, le_n0, le_n1, frames_k):
    fs, y = wavfile.read(audio)
    pm_times, pm, f0_times, f0, corr = pyreaper.reaper(y[le_n0:le_n1], fs)
    # if value of pitch is 0(low) then a possible word boundary
    indx = len(pm)//2-1
    pitch_prediction = 0
    count = 0
    flag0 = 0
    flag1 = 0
    while count <= frames_k:
        if indx-count > 0 and (pm[indx-count] == 0 or pm[indx+count] == 0):
            flag0 = 1
        if indx+count < len(pm) and (pm[indx+count] == 1 or pm[indx-count] == 1):
            flag1 = 1
        count += 1
    if flag1 == 1 and flag0 == 1:
        return 1
    return 0



#====================================================================================
#--------------------------------------making Dataframe----------------------------------
#====================================================================================


#k = 5
df8 = pd.read_csv("dataframe/dr8_k_4_added.csv")
df1 = pd.read_csv("dataframe/dr1_k_4_added.csv")
df2 = pd.read_csv("dataframe/dr2_k_4_added.csv")
df3 = pd.read_csv("dataframe/dr3_k_4_added.csv")
df4 = pd.read_csv("dataframe/dr4_k_4_added.csv")
df5 = pd.read_csv("dataframe/dr5_k_4_added.csv")
df6 = pd.read_csv("dataframe/dr6_k_4_added.csv")
df7 = pd.read_csv("dataframe/dr7_k_4_added.csv")


frames = [df1, df2, df3, df4, df5]
df = pd.concat(frames)


df = df.drop('Unnamed: 0', 1)

#df = df.filter(['Skewness', 'Truth'])
# Dividing dataframe into X and y set
X_train = df.iloc[:,:-1]
y_train = df.iloc[:,-1]

#print(Counter(y))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)


# Performing Under-sampling
rus = RandomUnderSampler()
# resampling X, y
X_rus, y_rus = rus.fit_resample(X_train, y_train)

X_rus['truth'] = y_rus
X_rus = shuffle(X_rus)
y_rus = X_rus['truth']
del X_rus['truth']

# y_rus = np.asarray(y_rus).astype('float32').reshape((-1,1))
# n = len(y_rus)
# y = [[0,0] for i in range(n)]
# print(len(y))
# for i in range(n):
#     if y_rus[i] == 0:
#         y[i][0] = 1
#     else:
#         y[i][1] = 1
# y = np.asarray(y).astype('float32').reshape((-1,2))

# X = X_rus.values
# X_train = []
# Y_train = []
# w = 5
# for i in range(w, len(X)-w):
#     X_train.append(np.transpose(X[i-w:i+w+1]))
#     Y_train.append(y[i])
# X_train = np.array(X_train)
# Y_train = np.array(Y_train)


# print('\n\nTensorflow begin\n\n')
# import tensorflow as tf
# from tensorflow.keras.layers import LSTM, Dense, Dropout

# model = tf.keras.Sequential()
# model.add(LSTM(128,input_shape=(7,11)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(2, activation='softmax'))
# model.summary()

# model.load_weights('model_w', by_name=False, skip_mismatch=False, options=None)

# Load the pickled model
# Load the model from the file
clf = joblib.load('ann.pkl')

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#====================================================================================
#--------------------------------------testing----------------------------------
#====================================================================================


tp = tn = fp = fn = n_words = n_words_predicted = 0
frames_k = 4
dialects = ['DR6','DR7','DR8']
for d in dialects:
    dialect = os.listdir('TRAIN/'+d)
    for speaker in dialect:
        files = os.listdir('TRAIN/'+d+'/'+speaker+'/')
        for file in files:
            if file[-3:] == 'WAV':
                audio = 'TRAIN/'+d+'/'+speaker+'/'+file
                x, sr = librosa.load(audio, sr = 16000)
                #========================================================================
                # Pre-emphasis / Preprocessing
                #========================================================================


                y, sr = librosa.load(audio, sr)
                x = librosa.effects.preemphasis(y)

                #========================================================================
                # variables
                #========================================================================

                #for starting 1600ms



                y = x
                Duration = (len(y)/sr)
                samples = int(sr*Duration)
                time_boundary_duration = 0.025
                frame_size = int(sr*time_boundary_duration)
                hop_length = int(0.015*sr)


                # truth stores 1/0 which is our ground truth
                truth = Truth_value(audio, hop_length, samples)
                y = truth

                # Create the pandas DataFrame

                X = pd.read_csv(audio+'.csv')
                X = X.drop('Unnamed: 0', 1)

                # predicting for each audio file
                # y_pred = model.predict(X)
                # y_test = np.asarray(y).astype('float32').reshape((-1,1))
                # n = len(y_test)
                # y = [[0,0] for i in range(n)]
                # for i in range(n):
                #     if y_test[i] == 0:
                #         y[i][0] = 1
                #     else:
                #         y[i][1] = 1
                # y = np.asarray(y).astype('float32').reshape((-1,2))

                # X = X.values
                # X_test = []
                # Y_test = []
                # w = 5
                # for i in range(w, len(X)-w):
                #     X_test.append(np.transpose(X[i-w:i+w+1]))
                #     Y_test.append(y[i])
                # X_test = np.array(X_test)
                # Y_test = np.array(Y_test)


                y_pred=clf.predict(X)
                l = len(y_pred)
                
                # y_pre = np.zeros(l)
                # for i in range(5):
                #     y_pre[i] = 1
                # for i in range(5,l-5):
                #     if y_pred[i-5][0] > 0.5:
                #         y_pre[i] = 0
                #     else:
                #         y_pre[i] = 1
                # for i in range(l-5,l):
                #     y_pre[i] = 1
                # print(y_pre)
                # y_pred = y_pre
                new_pred = []
                s = 0
                for i in range(l):
                    s += y_pred[i]
                    new_pred.append(s)


                #========================================================================
                # modification depending upon series on 1's and 0's
                #========================================================================


                z_pred = []
                for i in range(5):
                    z_pred.append(y_pred[i])
                for i in range(5,l-5):
                    if y_pred[i] == 1:
                        if new_pred[i+3]-new_pred[i-3]<3:      # 0011010      1100101
                            z_pred.append(0)
                        else:
                            z_pred.append(1)
                    else:
                        if new_pred[i+3]-new_pred[i-3]>3:      # 0011010      1100101
                            z_pred.append(1)
                        else:
                            z_pred.append(0)
                for i in range(l-5,l):
                    z_pred.append(y_pred[i])

                print(z_pred,flush=True)
                store1 = []     # for predicted boundaries '1'
                store3 = []     # for predicted words '0'
                i = 0
                while i < len(z_pred):
                    if z_pred[i] == 1:
                        start = i
                        i += 1
                        while i < len(z_pred) and z_pred[i] == 1:
                            i += 1
                        store1.append([start,i-1])
                    else:
                        start = i
                        i += 1
                        while i < len(z_pred) and z_pred[i] == 0:
                            i += 1
                        store3.append([start,i-1])

                store2 = []     # for truth boundaries
                store4 = []     # for truth words
                i = 0
                while i < len(truth):
                    if truth[i] == 1:
                        start = i
                        i += 1
                        while i < len(truth) and truth[i] == 1:
                            i += 1
                        store2.append([start,i-1])
                    else:
                        start = i
                        i += 1
                        while i < len(truth) and truth[i] == 0:
                            i += 1
                        store4.append([start,i-1])

                # print(truth,flush=True)

                print(store1, store2, flush=True)

                # print('\n\n',store3, store4, flush=True)

                # [[0, 13], [47, 50], [52, 52], [68, 74], [80, 84], [129, 133], [151, 181]]
                # [[0, 13], [31, 35], [46, 50], [71, 75], [78, 82], [85, 89], [112, 116], [149, 181]]

                ''' True Positive (TP) is an outcome where the model correctly predicts the positive class or word boundary.

                    True Negative (TN) is an outcome where the model correctly predicts the negative class.

                    False Positive (FP) is an outcome where the model incorrectly predicts the positive class.

                    False Negative (FN) is an outcome where the model incorrectly predicts the negative class.'''

                epsilon = 3
                n_words += len(store2) -1
                n_words_predicted += len(store1) -1

                # for positives 
                i = 0
                j = 0
                while i < len(store1) and j < len(store2):
                    if store1[i][1] + epsilon < store2[j][0]:
                        i += 1
                        fp += 1
                    elif store1[i][0] - epsilon < store2[j][0] and store1[i][1] + epsilon >= store2[j][0]:
                        if store1[i][1] + epsilon >= store2[j][1] or store1[i][1] - epsilon >= store2[j][1]:
                            j += 1
                        i += 1
                        tp += 1
                    elif store1[i][0] + epsilon >= store2[j][0] and store1[i][0] - epsilon <= store2[j][1]:
                        if store1[i][1] + epsilon >= store2[j][1] or store1[i][1] - epsilon >= store2[j][1]:
                            j += 1
                        i += 1
                        tp += 1
                    elif store1[i][0] - epsilon > store2[j][1] or store1[i][0] + epsilon > store2[j][1]:
                        j += 1
                        fp += 1
                    else:
                        print(i,j,'00',flush=True)

                # for negatives
                i = 0
                j = 0
                while i < len(store3) and j < len(store4):
                    if store3[i][1] + epsilon < store4[j][0]:
                        i += 1
                        fn += 1
                    elif store3[i][0] - epsilon < store4[j][0] and store3[i][1] + epsilon >= store4[j][0]:
                        if store3[i][1] + epsilon >= store4[j][1] or store3[i][1] - epsilon >= store4[j][1]:
                            j += 1
                        i += 1
                        tn += 1
                    elif store3[i][0] + epsilon >= store4[j][0] and store3[i][0] - epsilon <= store4[j][1]:
                        if store3[i][1] + epsilon >= store4[j][1] or store3[i][1] - epsilon >= store4[j][1]:
                            j += 1
                        i += 1
                        tn += 1
                    elif store3[i][0] - epsilon > store4[j][1] or store3[i][0] + epsilon > store4[j][1]:
                        j += 1
                        fn += 1
                    else:
                        print(i,j,'11',flush=True)
                

c0_p = c0_r = c1_p = c1_r = c0_f = c1_f = 0

# Precision = tp/(tp+fp)
# Recall = tp/(tp+fn)

c0_p = tn/(tn+fn)
c1_p = tp/(tp+fp)

c0_r = tn/(tn+fp)
c1_r = tp/(tp+fn)

print(tp,tn,fp,fn)

# F1 = 2*p*r/(p+r)
c0_f = 2*c0_p*c0_r/(c0_p+c0_r)
c1_f = 2*c1_p*c1_r/(c1_p+c1_r)
print('===============================================================')
print('No of words: ', n_words)
print('No of predicted words: ',n_words_predicted)
print('===============================================================')
print('Class-0\n')
print('===============================================================')
print('Precision---------',c0_p,'\nRecall---------',c0_r,'\nF1-measure---------', c0_f)
print('===============================================================')
print('Class-1\n')
print('===============================================================')
print('Precision---------',c1_p,'\nRecall---------',c1_r,'\nF1-measure---------', c1_f)