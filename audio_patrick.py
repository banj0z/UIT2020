import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 8
WAVE_OUTPUT_FILENAME = "file.wav"
WAVE_OUTPUT_FILENAME2 = "file2.wav"
INPUT_FRAMES_PER_BLOCK = 25 # FIX THIS!
audio = pyaudio.PyAudio()
audio2 = pyaudio.PyAudio()
# start Recording
stream = audio.open(format = FORMAT,
                                 channels = CHANNELS,
                                 rate = RATE,
                                 input = True,
                                 input_device_index = 2,
                                 frames_per_buffer = INPUT_FRAMES_PER_BLOCK)

stream2 = audio2.open(format = FORMAT,
                                 channels = CHANNELS,
                                 rate = RATE,
                                 input = True,
                                 input_device_index = 6,
                                 frames_per_buffer = INPUT_FRAMES_PER_BLOCK)

frames = []
frames2 = []
print("Recording started")
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    data2 = stream2.read(CHUNK)
    frames.append(data)
    frames2.append(data2)
print("finished recording")

# stop Recording
stream.stop_stream()
stream2.stop_stream()

stream.close()
stream2.close()

audio.terminate()
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

audio2.terminate()
waveFile2 = wave.open(WAVE_OUTPUT_FILENAME2, 'wb')
waveFile2.setnchannels(CHANNELS)
waveFile2.setsampwidth(audio2.get_sample_size(FORMAT))
waveFile2.setframerate(RATE)
waveFile2.writeframes(b''.join(frames2))
waveFile2.close()
