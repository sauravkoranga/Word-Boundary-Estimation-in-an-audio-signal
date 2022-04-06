import soundfile as sf
import librosa
import scipy.io.wavfile as sc
import librosa.display
import numpy as np
import pyreaper
import numpy
from scipy.io import wavfile
from scipy.stats import kurtosis, skew
from polycoherence import _plot_signal, polycoherence, plot_polycoherence
from math import pi
import pandas as pd
from scipy.fftpack import next_fast_len
from itertools import chain



# functions

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
    file = open(word_file,"r")
    lines = file.readlines()
    truth = []
    words = []
    for line in lines:
        w_start, w_end, w = line.split()
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
        while end < w_start-400:
            truth.append(1)                             #changed
            start += hop_length
            end += hop_length
        while start < w_start + 400:
            truth.append(1)
            start += hop_length
            end += hop_length
        while end < w_end-400:
            truth.append(0)
            start += hop_length
            end += hop_length
        while start < w_end + 400:
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
#--------------------------------------Driver code----------------------------------
#====================================================================================



import time
import os
# to measure the time to execute the code
start_time = time.time()

# to store the data-frame values
data = []
frames_k = 4

# adding 1600 samples that is 100ms of audio data at beginning and at the end
'''audio = 'SA1.WAV'
x, sr = librosa.load(audio, sr = 16000)
start_samples = x[:1600]
end_samples = x[-1600:]'''

audio_count = 0
dialect = os.listdir('TIMIT/TRAIN/DR4')
for speaker in dialect:
    files = os.listdir('TIMIT/TRAIN/DR4/'+speaker+'/')
    for file in files:
        if file[-3:] == 'WAV':
            audio = 'TIMIT/TRAIN/DR4/'+speaker+'/'+file
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

            #log-energy
            rms_audio = librosa.feature.rms(y, frame_length=frame_size, hop_length=hop_length)[0]

            #zero-crossings
            x_start = 0
            x_end = Duration
            zc_100 = []
            while x_start < x_end:
                x_n0 = int(x_start*sr)
                x_n1 = int((x_start+time_boundary_duration)*sr)
                x_zero_crossings = librosa.zero_crossings(y[x_n0:x_n1], pad=False)
                zc_100.append(sum(x_zero_crossings))
                x_start += 0.015


            #kurtosis
            x_start = 0
            x_end = Duration
            k = []
            while x_start < x_end:
                x_n0 = int(x_start*sr)
                x_n1 = int((x_start+time_boundary_duration)*sr)
                kurt = kurtosis(y[x_n0:x_n1])
                k.append(kurt)
                x_start += 0.015

            #skewness
            x_start = 0
            x_end = Duration
            s = []
            while x_start < x_end:
                x_n0 = int(x_start*sr)
                x_n1 = int((x_start+time_boundary_duration)*sr)
                sk = skew(y[x_n0:x_n1])
                s.append(sk)
                x_start += 0.015

            y = x[1600:]
            y = y[:-1600]
            Duration = (len(y)/sr)
            start = 0.1
            end = 0.125
            p = []
            vov = []
            dov = []

            while (end <= Duration+0.125):

                # taking time 100ms before and after the test boundary
                le_start = round(start-0.1, 3)
                le_end = round(end+0.1, 3)

                # time to samples
                le_n0 = int(le_start*16000)
                le_n1 = int(le_end*16000)

                #========================================================================
                # Feature: Pitch(short time pitch frequency)
                #========================================================================
                # print(start,flush=True)
                if start < 0.100001:
                    first_pitch = find_pitch(audio, le_n0, le_n1, frames_k)
                    for i in range(11):
                        p.append(first_pitch)
                else:
                    p.append(find_pitch(audio, le_n0, le_n1, frames_k))

                #========================================================================
                # Feature-6: Bispectral features
                #========================================================================


                kw = dict(nperseg=frame_size, noverlap=int(sr*0.01), nfft=int(0.05*sr))
            
                freq1, freq2, bicoh = polycoherence(x[le_n0:le_n1], sr, **kw)
                freq1 = normalize_freq(freq1)
                bicoh = check_bicoh(bicoh)
                column_sums = bicoh.sum(axis=0)
                normalize(column_sums)
                vov_val = round(VOV(column_sums),3)
                dov_val = round(DOV(column_sums),3)
                vov.append(vov_val)
                dov.append(dov_val)
                if start < 0.100001:
                    for i in range(11):
                        vov.append(vov_val)
                        dov.append(dov_val)
                else:
                    vov.append(vov_val)
                    dov.append(dov_val)
                start += 0.01
                end += 0.01

            last_pitch = find_pitch(audio, le_n0, le_n1, frames_k)
            for i in range(10):
                p.append(last_pitch)
            for i in range(10):
                vov.append(vov_val)
                dov.append(dov_val)

            # truth stores 1/0 which is our ground truth
            truth = Truth_value(audio, hop_length, samples)

            normalize(rms_audio)
            normalize(zc_100)
            normalize(k)
            normalize(s)
            normalize(vov)
            normalize(dov)

            #print(len(rms_audio), len(zc_100), len(p), len(k), len(s), len(vov), len(dov), len(truth))
            for i in range(len(k)):
                data.append([rms_audio[i], zc_100[i], p[i], k[i], s[i], vov[i], dov[i], truth[i]])
            
            

# Create the pandas DataFrame
df = pd.DataFrame(data, columns = ['Log-Energy', 'Zero-Crossings rate', 'Pitch', 'Kurtosis', 'Skewness', 'Bi-VOV', 'Bi-DOV', 'Truth'])

# saving the dataframe
df.to_csv('dataframe/dr4_k_4_added.csv')

end_time = time.time()

# total time taken
print(f"Runtime of the program is {end_time - start_time}")
