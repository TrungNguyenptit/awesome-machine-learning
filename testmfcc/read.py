from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from file_config import test_batden_dir
from file_config import test_tatden_dir
from file_config import train_batden_dir
from file_config import train_tatden_dir
import numpy as np
import glob
import ntpath

list_wav_paths_train_batden = glob.glob(train_batden_dir)
list_wav_paths_train_tatden = glob.glob(train_tatden_dir)
list_wav_paths_test_batden = glob.glob(test_batden_dir)
list_wav_paths_test_tatden = glob.glob(test_tatden_dir)
