#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# uncomment for debugging:
# import matplotlib.pyplot as plt

import numpy as np
from scipy.fftpack import rfft
from scipy.io.wavfile import read as readwav
from scipy.signal import spectrogram
import sys
import warnings

# print usage if no argument given
if len(sys.argv[1:]) < 1:
    print('usage %s audio_file.wav' % (sys.argv[0]))
    sys.exit(1)

# read audio samples and ignore warnings, print errors
try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    input_data = readwav(sys.argv[1])
except IOError as e:
    print(e[1])
    sys.exit(e[0])

# process data
freq = input_data[0]
audio = input_data[1]
channel = 0
toffset = 10
tsample_size = 20
samples = len(audio[:, 0])
seconds = int(samples / freq)
seconds = min(seconds, tsample_size)
ssample_size = freq*seconds

f,t,Sxx = spectrogram(audio[freq*toffset:ssample_size + (freq*toffset),
    channel], freq)
color_arr = 10*np.log(Sxx)

# Takes logified Sxx with range 0-100
def get_cutoff(colors):
    for i,row in enumerate(colors):
        if np.quantile(row, 0.60) < -100:
            return f[i]
    return f[len(colors)-1]

print(get_cutoff(color_arr))
# debugging only:
# Use this to label the frequencies appropriately
# ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x/1000) + 'K'))
if 'plt' in globals():
    # plot
    plt.pcolormesh(t, f, color_arr, shading='gouraud')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    # set the title
    plt.title('Spectrogram')
    plt.show()

