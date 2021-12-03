import sys
import soundfile as sf
import glob
import os,tqdm

p2root = sys.argv[1]

manifest = p2root+"/manifest/"

if not os.path.exists(manifest):
    os.makedirs(manifest)

charset = set()
for folder in tqdm.tqdm(os.listdir(p2root)):
    if 'manifest' == folder:
        continue
    wavs = glob.glob(p2root+'/'+folder+'/**/*.wav',recursive=True)
    samples = [len(sf.read(w)[0]) for w in wavs]
    #print(wavs)
    root = os.path.abspath(os.path.split(wavs[0])[0])	
    wavs = [os.path.split(x)[-1] for x in wavs]

    wav2trans = dict()

    with open(p2root+'/'+folder+'/transcription.txt','r') as transcrip:
        lines = transcrip.read().strip().split('\n')
    for line in lines:
        if '\t' in line:
            file, trans = line.split("\t")
        else:
            splitted_line = line.split(" ")
            file, trans = splitted_line[0], " ".join(splitted_line[1:])
        wav2trans[file] = trans
        charset.update(trans.replace(" ","|"))
    

    with open(manifest+folder+".tsv",'w') as tsv, \
        open(manifest+folder+".wrd","w") as wrd, \
        open(manifest+folder+".ltr",'w') as ltr:
        print(root,file=tsv)
        for n,d in zip(wavs,samples):
            print(n,d,sep='\t',file=tsv)
            print(wav2trans[n[:-4]],file=wrd)
            print(" ".join(list(wav2trans[n[:-4]].replace(" ", "|"))) + " |", file=ltr)


with open(manifest+"dict.ltr.txt",'w') as dct:
    for e,c in enumerate(charset):
        print(c,e,file=dct)
