basepath = "<path to dataset>"

with open(basepath+"/**/train_bg/wav.scp") as f:
    lines = f.read().strip().split('\n')
for line in tqdm.tqdm(lines):
#     name, _ = line.strip().split('\t')
    name = line.strip().split(' ')[0]
    
    shutil.copy(basepath+"/audio/"+name+".flac", basepath+"/**/train_wav/"+name+".flac")

with open(basepath+"/**/dev_bg/wav.scp") as f:
    lines = f.read().strip().split('\n')
for line in tqdm.tqdm(lines):
#     name, _ = line.strip().split('\t')
    name = line.strip().split(' ')[0]
    
    shutil.copy(basepath+"/audio/"+name+".flac", basepath+"/**/valid_wav/"+name+".flac")
