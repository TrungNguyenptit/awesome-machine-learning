from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import glob
import csv
import tensorflow as tf
from setuptools.command.test import test

from testmfcc.file_config import test_batden_dir, train_batden_dir, test_tatden_dir, train_tatden_dir, nothing_dir


list_wav_paths_train_batden = glob.glob(train_batden_dir)
list_wav_paths_train_tatden = glob.glob(train_tatden_dir)
list_wav_paths_test_batden = glob.glob(test_batden_dir)
list_wav_paths_test_tatden = glob.glob(test_tatden_dir)


def nothing_data():
    nothings_in = []
    nothings_out = []
    (rate, sig) = wav.read(nothing_dir)
    sig = sig.tolist()
    new_rate = 2756
    sig_0 = []
    shift = 20000
    step = 2115
    for i in range(int(len(sig)/step)):
        sig_0.append(sig[i*step: shift+i*step])

    sig_0 = sig_0[0:2000]
    print(len(sig_0))

    for x in sig_0:
        new_sig = []
        for i in range(int(len(x) / 16)):
            if i * 16 < len(x):
                new_sig.append(x[i * 16])

        mfcc_feat = mfcc(np.array(new_sig), new_rate, 50 / new_rate, 20 / new_rate)
        # print(mfcc_feat)
        a = mfcc_feat[0:100]
        a = a.ravel()
        a = a.tolist()
        if 1300 > len(a):
            l = len(a)
            a = a + [0] * (1300 - l)

        nothings_in.append(a)
        nothings_out.append([2])

    print(len(nothings_in))
    return (nothings_in, nothings_out)


def save_data():
    train_in = []
    train_out = []

    for wav_path in list_wav_paths_train_batden:
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

        train_in.append(a)
        train_out.append([1])

    for wav_path in list_wav_paths_train_tatden:
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

        train_in.append(a)
        train_out.append([0])

    (nothings_in, nothings_out) = nothing_data()

    train_in = train_in + nothings_in[0:1800]
    train_out = train_out + nothings_out[0:1800]

    print(np.array(train_in).shape)

    with open('train_in.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(train_in)

    with open('train_out.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(train_out)


def save_test_data():
    test_in = []
    test_out = []

    print(len(list_wav_paths_test_batden))
    print(len(list_wav_paths_test_tatden))

    for wav_path in list_wav_paths_test_batden:
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

        test_in.append(a)
        test_out.append([1])

    for wav_path in list_wav_paths_test_tatden:
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

        test_in.append(a)
        # print(test_in)
        test_out.append([0])

    (nothings_in, nothings_out) = nothing_data()

    test_in = test_in + nothings_in[1800:2000]
    test_out = test_out + nothings_out[1800:2000]
    print(np.array(test_in).shape)

    with open('test_in.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(test_in)

    with open('test_out.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(test_out)


def more_train():
    (train_in, train_out) = get_train_data()
    (test_in, test_out) = test()
    tm = []
    x = []
    y = []
    y_test = []
    for i in train_out:
        y.append(i[0])

    for i in test_out:
        y_test.append(i[0])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1300,)),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(np.array(train_in), np.array(y), epochs=50)
    # model.evaluate(np.array(test_in), np.array(y_test))
    model.save('my_model_2.h5')
    # model.predict(x[0])


def get_train_data():
    train_in = []
    train_out = []

    with open('train_in.csv', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                train_in.append(row)

    with open('train_out.csv', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                train_out.append(row)
    # print(len(train_in))
    print("read done.")
    return (train_in, train_out)


def test():
    test_in = []
    test_out = []

    with open('test_in.csv', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                test_in.append(row)

    with open('test_out.csv', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                test_out.append(row)

    test_in = np.array(test_in)
    test_in = test_in.astype(float)
    test_in = test_in.tolist()

    test_out = np.array(test_out)
    test_out = test_out.astype(int)
    test_out = test_out.tolist()

    return (test_in, test_out)


if __name__ == "__main__":
    more_train()
    # more_train()
    # nothing_data()
    # save_data()
    # save_test_data()
    print("done")

