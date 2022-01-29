import collections
import contextlib
import sys
import wave
import os
import webrtcvad
import tqdm
import glob
#from joblib import Parallel, delayed
import io
from pydub import AudioSegment

#Usage
#python <script>.py <data_read_dir> <data_write_dir> <language_name>
import pydub
import soundfile as sf
import numpy as np
def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        # print(num_channels)
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate == 16000
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def read_wave2(path):
    wav = pydub.AudioSegment.from_file(path, sample_width=2, frame_rate=16000, channels=1)
    sarray = wav.get_array_of_samples()
    fp_arr = np.array(sarray).T.astype(np.float64)
    fp_arr /= np.iinfo(sarray.typecode).max
    fp_arr = fp_arr.reshape(-1)
    # return fp_arr, 16000
    return wav.raw_data, 16000

def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

#         sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
#                 sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
#                 sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                start_frame = voiced_frames[0].timestamp
                end_frame = voiced_frames[-1].timestamp + voiced_frames[-1].duration
                # yield b''.join([f.bytes for f in voiced_frames])
                yield b''.join([f.bytes for f in voiced_frames]), (start_frame, end_frame)
                ring_buffer.clear()
                voiced_frames = []
#     if triggered:
#         sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
#     sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        start_frame = voiced_frames[0].timestamp
        end_frame = voiced_frames[-1].timestamp + voiced_frames[-1].duration        
        yield b''.join([f.bytes for f in voiced_frames]), (start_frame, end_frame)

def vad_file(fpath,vad_level):
    audio, sample_rate = read_wave2(fpath)
    vad = webrtcvad.Vad(vad_level)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    print(len(audio),len(frames))
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    dir_path, fname = os.path.split(fpath)
    fname = fname.split('.')[0]+".wav"

    # print(len(segments))
    for i, (segment, (start_frame, end_frame)) in enumerate(segments):
        # arr = np.frombuffer(segment,dtype=np.float32)
        # s=(io.BytesIO(audio))
        # song=AudioSegment.from_raw(s, sample_width=2, frame_rate=16000, channels=1)
        # samples = song.get_array_of_samples()
        # samples = np.array(samples)
        # print(samples[100:120])
        # print(arr[100:120],'\n',read_wave2(fpath)[0][100:120])
        print(f'[{i}] Start frame: {start_frame},\t End frame: {end_frame}\t|\t{end_frame-start_frame}')
        # break
        # o_fname = 'vad_%002d_' % (i,) +fname
        # write_wave(write_path+"/"+o_fname, segment, sample_rate)

# data_path = sys.argv[1]
# write_path = sys.argv[2]


# folder = sys.argv[3]
# print(folder)
# if not os.path.exists(write_path):
#     os.mkdir(write_path)

# f_all = glob.glob(data_path+'/'+folder+"/**/*.wav",recursive=True)

# Parallel(n_jobs=-1)(
#     delayed(vad_file)(file,write_path+'/'+folder+'/',2) for file in tqdm.tqdm(f_all)
# )
#for file in tqdm.tqdm(f_all):
#    vad_file(file,write_path+'/'+folder+'/',2)

# vad_file("test.wav",3)

