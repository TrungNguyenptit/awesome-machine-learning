from setuptools.command.test import test
import matplotlib.pyplot as plt
import math
import contextlib
import awesome
import numpy as np
from tensorflow import keras
import scipy.io.wavfile as wav
from python_speech_features import mfcc
cutOffFrequency = 400.0

from scipy.signal import butter, lfilter

def get_data(wav_path):
    (rate, sig) = wav.read(wav_path)
    sig = sig.tolist()
    new_rate = 2756
    new_sig = []
    for i in range(int(len(sig) / 16)):
        if i * 16 < len(sig):
            new_sig.append(sig[i * 16])

    mfcc_feat = mfcc(np.array(new_sig), new_rate, 50 / new_rate, 20 / new_rate)
    # print(mfcc_feat)
    a = mfcc_feat[0:100]
    a = a.ravel()
    a = a.tolist()
    if 1300 > len(a):
        l = len(a)
        a = a + [0] * (1300 - l)

    return a

if __name__ == "__main__":
    (test_in, test_out) = awesome.test()
    y_test = []
    for i in test_out:
        y_test.append(i[0])
    print(np.array(test_in[0]).shape)

    a = get_data("C:\\Users\\today\\Desktop\\Recording (2)\\Recording (12).wav")

    new_model = keras.models.load_model('my_model.h5')

    loss, acc = new_model.evaluate(np.array(test_in),np.array(y_test))
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    predictions = new_model.predict_classes(np.array([a]))

    print(predictions[0])

    print("done")
