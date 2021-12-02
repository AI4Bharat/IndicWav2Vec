from pydub import AudioSegment
from pydub.utils import make_chunks
from pathlib import Path
from joblib import Parallel, delayed
import sys,os,tqdm

#Usage
#python <script>.py <path to dir to be chunked>

# crop size in milli secs
size = 20000
def chunk(file):
    audio = AudioSegment.from_file(file)
    if int(audio.duration_seconds) > 25:
        chunks = make_chunks(audio, size)


        str_path = file.absolute()
        str_path = str_path.as_posix()
        dir, file_name = str_path.split("/"), str_path.split("/")[-1].split(".")[0]
        dir = "/".join(dir[:-1])

        for i, chunk in enumerate(chunks):
            chunk_name = f"{dir}/broken-{file_name}-{i}.wav"
            chunk.export(chunk_name, format="wav")
        os.remove(str_path)

target_folder = sys.argv[1]

audio_paths = sorted(list(Path(target_folder).glob("**/*.wav")))


Parallel(n_jobs=-1)(delayed(chunk)(file) for file in tqdm.tqdm(audio_paths))
