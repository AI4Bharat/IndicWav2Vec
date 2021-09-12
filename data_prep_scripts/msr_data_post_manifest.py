manifest = "path/to/manifest/containing/tsv_files"

with open(manifest+"/train.tsv",'r') as train_tsv, \
    open(manifest+"/valid.tsv",'r') as val_tsv,\
    open(manifest+"/dict.ltr.txt",'w') as dict_ltr,\
    open(manifest+"/valid.wrd",'w') as wrd_val_out,\
    open(manifest+"/train.wrd",'w') as wrd_train_out,\
    open(manifest+"/valid.ltr",'w') as ltr_val_out,\
    open(manifest+"/train.ltr",'w') as ltr_train_out:
        root_train = next(train_tsv).strip()
        root_valid = next(val_tsv).strip()

        transcription_train = os.path.dirname(root_train)+"/transcription.txt" 
        transcription_val = os.path.dirname(root_valid)+"/transcription.txt"
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


        with open(transcription_val,'r') as transcrip:
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

            print(trans_dict[line[0][:-4]], file=wrd_train_out)
            print(
                " ".join(list(trans_dict[line[0][:-4]].replace(" ", "|"))) + " |",
                file=ltr_train_out,
            )
        for line in val_tsv:
            line = line.strip().split('\t')

            print(trans_dict[line[0][:-4]], file=wrd_val_out)
            print(
                " ".join(list(trans_dict[line[0][:-4]].replace(" ", "|"))) + " |",
                file=ltr_val_out,
            )
        chars = set()
        for word in list(trans_dict.values()):
            word=word.replace(" ","|")
            chars = chars.union(set(list(word)))
        char_dict = {}
        for v, k in enumerate(list(chars)):
            char_dict[k] = v

        for k,v in char_dict.items(): 
            print(k,v,file=dict_ltr)
