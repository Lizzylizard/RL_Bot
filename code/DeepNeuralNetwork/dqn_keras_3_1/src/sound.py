#!/usr/bin/env python

# source

'''
https://realpython.com/playing-and-recording-sound-python/
'''

# and

'''https://pythonbasics.org/python-play-sound/'''

import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.playback import play

def make_sound():
  frequency = 440  # Our played note will be 440 Hz
  fs = 44100  # 44100 samples per second
  seconds = 0.3  # Note duration of 3 seconds

  # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
  t = np.linspace(0, seconds, seconds * fs, False)

  # Generate a 440 Hz sine wave
  note = np.sin(frequency * t * 2 * np.pi)

  wavfile.write('Sine.wav', fs, note)

  song = AudioSegment.from_wav("Sine.wav")
  play(song)