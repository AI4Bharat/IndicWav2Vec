manifest = "<mucs_lang_manifest>"
trans_train = "<mucs_lang_path>/train/transcription.txt"
trans_val = "<mucs_lang_path>/test/transcription.txt"

trans_dict = {}
with open(trans_train, 'r') as f:
    lines = f.read().strip().split("\n")
    for line in tqdm.tqdm(lines):
        spline = line.split(" ")
        fname, trs = spline[0], " ".join(spline[1:])
        trans_dict[fname+".wav"] = trs

with open(trans_val, 'r') as f:
    lines = f.read().strip().split("\n")
    for line in tqdm.tqdm(lines):
        spline = line.split(" ")
        fname, trs = spline[0], " ".join(spline[1:])
        trans_dict[fname+".wav"] = trs
  
  
with open(manifest+"/train.tsv",'r') as train_tsv, \
open(manifest+"/valid.tsv",'r') as val_tsv,\
open(manifest+"/dict.ltr.txt",'w') as dict_ltr,\
open(manifest+"/valid.wrd",'w') as wrd_val_out,\
open(manifest+"/train.wrd",'w') as wrd_train_out,\
open(manifest+"/valid.ltr",'w') as ltr_val_out,\
open(manifest+"/train.ltr",'w') as ltr_train_out:
    root_train = next(train_tsv).strip()
    root_valid = next(val_tsv).strip()
    
    for line in tqdm.tqdm(train_tsv):
        line = line.strip().split('\t')

        print(trans_dict[line[0]], file=wrd_train_out)
        print(
            " ".join(list(trans_dict[line[0]].replace(" ", "|"))) + " |",
            file=ltr_train_out,
        )
        
    for line in tqdm.tqdm(val_tsv):
        line = line.strip().split('\t')

        print(trans_dict[line[0]], file=wrd_val_out)
        print(
            " ".join(list(trans_dict[line[0]].replace(" ", "|"))) + " |",
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

