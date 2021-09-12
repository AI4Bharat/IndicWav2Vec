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

manifest = "path_to_manifest_containing_tsvs"
    
with open(manifest+"/train.tsv",'r') as train_tsv, \
open(manifest+"/valid.tsv",'r') as val_tsv,\
open(manifest+"/dict.ltr.txt",'w') as dict_ltr,\
open(manifest+"/valid.wrd",'w') as wrd_val_out,\
open(manifest+"/train.wrd",'w') as wrd_train_out,\
open(manifest+"/valid.ltr",'w') as ltr_val_out,\
open(manifest+"/train.ltr",'w') as ltr_train_out:
    root_train = next(train_tsv).strip()
    root_valid = next(val_tsv).strip()

    transcription_train = "path/to/train/transcriptions/train_bg/text"

    transcription_valid = "path/to/valid/transcriptions/dev_bg/text"
    
    trans_dict = {}

    with open(transcription_train,'r') as transcrip:
        lines = transcrip.read().strip().split('\n')
    for line in lines:
        if '\t' in line:
            file, trans = line.split("\t")
        else:
            splitted_line = line.split(" ")
            file, trans = splitted_line[0], " ".join(splitted_line[1:])
        trans_dict[file] = trans

    with open(transcription_valid,'r') as transcrip:
        lines = transcrip.read().strip().split('\n')
    for line in lines:
        if '\t' in line:
            file, trans = line.split("\t")
        else:
            splitted_line = line.split(" ")
            file, trans = splitted_line[0], " ".join(splitted_line[1:])
        trans_dict[file] = trans
        
    for line in train_tsv:
        line = line.strip().split('\t')

        print(trans_dict[line[0][:-5]], file=wrd_train_out)
        print(
            " ".join(list(trans_dict[line[0][:-5]].replace(" ", "|"))) + " |",
            file=ltr_train_out,
        )
        
    for line in val_tsv:
        line = line.strip().split('\t')

        print(trans_dict[line[0][:-5]], file=wrd_val_out)
        print(
            " ".join(list(trans_dict[line[0][:-5]].replace(" ", "|"))) + " |",
            file=ltr_val_out,
        )
        
    chars = set()
    for word in list(trans_dict.values()):
        word = word.replace(" ","|")
        chars = chars.union(set(list(word)))
    char_dict = {}
    for v, k in enumerate(list(chars)):
        char_dict[k] = v

    for k,v in char_dict.items(): 
        print(k,v,file=dict_ltr)

#     print(os.path.dirname(t.read().split('\n')[0]))

