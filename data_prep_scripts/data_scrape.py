from __future__ import unicode_literals
from youtube_dl import YoutubeDL
import pandas as pd
import tqdm,os,sys


#Usage
#python <script>.py <xlsx file> <Sheet name from xlsx> <download folder name>

new_excel_path = sys.argv[1]

sheet_name = sys.argv[2]
lang=sheet_name.strip().lower()
version = sys.argv[3]

sheet_names_new = pd.ExcelFile(new_excel_path).sheet_names
print(sheet_names_new)

assert (sheet_name in sheet_names_new)

save_path = version+"/"+lang

if not os.path.exists(save_path):
    os.makedirs(save_path+"/audio")
    

ydl_opts = {
    'format': 'bestaudio/best',
    'noplaylist':True,
    'outtmpl': save_path+'/audio/%(id)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192'
    }],
    'postprocessor_args': [
        '-ar', '16000',
        '-ac', '1'
    ],
    'prefer_ffmpeg': True,
    'keepvideo': False
}

df_new = pd.read_excel(new_excel_path,header=1,sheet_name=sheet_name)


urls_new = set(df_new['URL\'s'].dropna())

   
urls = urls_new

if os.path.exists(save_path+'/completed.txt'): 
    done_urls = []
    with open(save_path+'/completed.txt','r') as c_r:
        done_urls += c_r.read().strip().split('\n')
    while '' in done_urls:
        done_urls.remove('') 

    done_urls = set(done_urls)
    urls = urls - done_urls
    
    with open(save_path+'/completed.txt', 'w') as cc:
        print('\n'.join(list(done_urls)).strip(),file=cc)


faulty = open(save_path+'/faulty.txt','w')
faulty.close()

with YoutubeDL(ydl_opts) as ydl:
    for url in tqdm.tqdm(list(urls)):
        try:
            ydl.download([url.strip()])

        except:
            faulty = open(save_path+'/faulty.txt','a')
            faulty.write(url+'\n')
            faulty.close()
            continue

        comp = open(save_path+'/completed.txt','a')
        comp.write(url+"\n")
        comp.close()

