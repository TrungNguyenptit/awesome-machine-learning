from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np

(rate,sig) = wav.read("C:\\Users\\Admin\\Music\\dulieuhocmay\\test_tatden\\Recording (1009).wav")
mfcc_feat = mfcc(sig,rate, 400/rate, 160/rate)
fbank_feat = logfbank(sig,rate, 400/rate, 160/rate)

for i in sig:
    print(i)

# print(rate)
a = mfcc_feat[0:900]
# print(a.shape)
a = a.ravel()
a = a.tolist()
if 11700 > len(a):
    l = len(a)
    a = a + [0]*(11700-l)
t = 0
# print(a)


