from flask import Flask, request
from flask import jsonify
from flask_sockets import Sockets
from flask_cors import CORS, cross_origin
from omegaconf import OmegaConf
import numpy as np, math,torch,json
from support import load_model,W2lKenLMDecoder,W2lViterbiDecoder,load_data
import time,os
import soundfile as sf
import io
from pydub import AudioSegment
import webrtcvad
import re
from vad_old import frame_generator, vad_collector
from nltk import sent_tokenize
from punctuate import RestorePuncts

punct_model = RestorePuncts()

with open('models_info.json','r') as j:
    config = json.load(j)

name2model_dict = dict()
for k,m in config.items():
    if eval(m['lm_usage']):
        lmarg = OmegaConf.create(m['lm_details'])
        lmarg.unk_weight = -math.inf
        model,dictionary = load_model(m['model_path'])
        generator = W2lKenLMDecoder(lmarg, dictionary)
    else:
        lmarg = OmegaConf.create({'nbest':1})
        model,dictionary = load_model(m['model_path'])
        generator = W2lViterbiDecoder(lmarg, dictionary)
    name2model_dict[k] = [model,generator,dictionary]

def align(fp_arr,model,generator,dictionary,cuda='cpu'):
    feature = torch.from_numpy(fp_arr).float()
    if cuda != 'cpu' and torch.cuda.is_available():
        feature = feature.to(cuda)
    sample = {"net_input":{"source":None,"padding_mask":None}}
    sample["net_input"]["source"] = feature.unsqueeze(0)
    if cuda != 'cpu' and torch.cuda.is_available():
        sample["net_input"]["padding_mask"] = torch.BoolTensor(sample["net_input"]["source"].size(1)).fill_(False).unsqueeze(0).to(cuda)
    else:
        sample["net_input"]["padding_mask"] = torch.BoolTensor(sample["net_input"]["source"].size(1)).fill_(False).unsqueeze(0)
        
    with torch.no_grad():
        hypo = generator.generate([model], sample, prefix_tokens=None)
    hyp_pieces = dictionary.string(hypo[0][0]["tokens"].int().cpu())
    tr = hyp_pieces.replace(' ','').replace('|',' ').strip()
    #print(tr)
    return tr

def predict_from_sample(fp_arr,model,generator,dictionary,vad_val,chunk_size,la):
    sample_rate = 16000
    vad = webrtcvad.Vad(vad_val) #2
    frames = frame_generator(30, fp_arr, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    
    op = ''

    for i, (segment, (start_frame, end_frame)) in enumerate(segments):
        song=AudioSegment.from_raw(io.BytesIO(segment), sample_width=2, frame_rate=16000, channels=1)
        samples = song.get_array_of_samples()
        arr = np.array(samples).T.astype(np.float64)
        arr /= np.iinfo(samples.typecode).max
        arr = arr.reshape(-1)
        for e,frame in enumerate(range(0,len(arr),int(chunk_size))):
            if end_frame-frame-start_frame <= chunk_size + 0.1:
                op_pred = align(arr[int((frame)*16000):int((end_frame)*16000)],model,generator,dictionary) + ' ' 
                if len(op_pred.strip()) >2:
                    op+= op_pred
                break
            else:
                op_pred = align(arr[int((frame)*16000):int((frame+chunk_size+0.1)*16000)],model,generator,dictionary)
                if len(op_pred.strip()) > 2:
                  op+= op_pred + ' '
    if la == 'en':
        sent = op.lower()
        sent = re.sub(r'[^\w\s]', '', sent)
		
        punctuated = punct_model.punctuate(sent,lang=la)
        return punctuated
    return op
    #tokenised = sent_tokenize(punctuated)

app = Flask(__name__)
sockets = Sockets(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def hello_world():
    return "<p>Hi</p>"

@app.route("/infer_en",methods=['POST'])
@cross_origin()
def infer_en():
    print('hit')
    lang = "en" 
    if not os.path.exists('../downloaded/'+lang):
        os.makedirs('../downloaded/'+lang)
    start_time = time.time()
    stp = '../downloaded/'+lang+'/'+request.files['file'].filename
    request.files['file'].save(stp)  
    vad_val = 2
    chunk_size = float(5.0)
    fp_arr = load_data(stp,of='raw')
    model,generator,dictionary = name2model_dict[lang]
    res = predict_from_sample(fp_arr,model,generator,dictionary,vad_val,chunk_size,lang)
    end_time = time.time()
    return jsonify({'text': res, 'duration':round(end_time-start_time,2)})

@sockets.route('/stream')
@cross_origin()
def start_streaming(ws):
    #app.logger.info("Connection accepted")
    while not ws.closed:
        message = ws.receive()
        if message is None:
            #app.logger.info("No message received...")
            continue
        
        ws.send("hello")
        


@app.route("/infer_ulca_en",methods=['POST'])
@cross_origin()
def infer_ulca_en():
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    vad_val = req_data.get('vad_level',2)
    chunk_size = float(req_data.get('chunk_size',5.0))
    lang = 'en'
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = str(round(time.time() * 1000))
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
            print('data loaded')
        except:
            status = 'ERROR' 
            continue
            
        model,generator,dictionary = name2model_dict[lang]
        res = predict_from_sample(fp_arr,model,generator,dictionary,vad_val,chunk_size,lang)
        preds.append({'source':res})
    return jsonify({"status":status, "output":preds})



@app.route("/infer_ulca_hi",methods=['POST'])
@cross_origin()
def infer_ulca_hi():
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    vad_val = req_data.get('vad_level',2)
    chunk_size = float(req_data.get('chunk_size',5.0))
    lang = 'hi'
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = str(round(time.time() * 1000))
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
            print('data loaded')
        except:
            status = 'ERROR' 
            continue
            
        model,generator,dictionary = name2model_dict[lang]
        res = predict_from_sample(fp_arr,model,generator,dictionary,vad_val,chunk_size,lang)
        preds.append({'source':res})
    return jsonify({"status":status, "output":preds})


@app.route("/infer_ulca_ta",methods=['POST'])
@cross_origin()
def infer_ulca_ta():
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    vad_val = req_data.get('vad_level',2)
    chunk_size = float(req_data.get('chunk_size',5.0))
    lang = 'ta'
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = str(round(time.time() * 1000))
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
            print('data loaded')
        except:
            status = 'ERROR' 
            continue
            
        model,generator,dictionary = name2model_dict[lang]
        res = predict_from_sample(fp_arr,model,generator,dictionary,vad_val,chunk_size,lang)
        preds.append({'source':res})
    return jsonify({"status":status, "output":preds})


@app.route("/infer_ulca_te",methods=['POST'])
@cross_origin()
def infer_ulca_te():
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    vad_val = req_data.get('vad_level',2)
    chunk_size = float(req_data.get('chunk_size',5.0))
    lang = 'te'
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = str(round(time.time() * 1000))
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
            print('data loaded')
        except:
            status = 'ERROR' 
            continue
            
        model,generator,dictionary = name2model_dict[lang]
        res = predict_from_sample(fp_arr,model,generator,dictionary,vad_val,chunk_size,lang)
        preds.append({'source':res})
    return jsonify({"status":status, "output":preds})


@app.route("/infer_ulca_gu",methods=['POST'])
@cross_origin()
def infer_ulca_gu():
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    vad_val = req_data.get('vad_level',2)
    chunk_size = float(req_data.get('chunk_size',5.0))
    lang = 'gu'
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = str(round(time.time() * 1000))
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
            print('data loaded')
        except:
            status = 'ERROR' 
            continue
            
        model,generator,dictionary = name2model_dict[lang]
        res = predict_from_sample(fp_arr,model,generator,dictionary,vad_val,chunk_size,lang)
        preds.append({'source':res})
    return jsonify({"status":status, "output":preds})


@app.route("/infer_ulca_bn",methods=['POST'])
@cross_origin()
def infer_ulca_bn():
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    vad_val = req_data.get('vad_level',2)
    chunk_size = float(req_data.get('chunk_size',5.0))
    lang = 'bn'
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = str(round(time.time() * 1000))
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
            print('data loaded')
        except:
            status = 'ERROR' 
            continue
            
        model,generator,dictionary = name2model_dict[lang]
        res = predict_from_sample(fp_arr,model,generator,dictionary,vad_val,chunk_size,lang)
        preds.append({'source':res})
    return jsonify({"status":status, "output":preds})


@app.route("/infer_ulca_ne",methods=['POST'])
@cross_origin()
def infer_ulca_ne():
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    vad_val = req_data.get('vad_level',2)
    chunk_size = float(req_data.get('chunk_size',5.0))
    lang = 'ne'
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = str(round(time.time() * 1000))
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
            print('data loaded')
        except:
            status = 'ERROR' 
            continue
            
        model,generator,dictionary = name2model_dict[lang]
        res = predict_from_sample(fp_arr,model,generator,dictionary,vad_val,chunk_size,lang)
        preds.append({'source':res})
    return jsonify({"status":status, "output":preds})



@app.route("/infer_ulca_si",methods=['POST'])
@cross_origin()
def infer_ulca_si():
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    vad_val = req_data.get('vad_level',2)
    chunk_size = float(req_data.get('chunk_size',5.0))
    lang = 'si'
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = str(round(time.time() * 1000))
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
            print('data loaded')
        except:
            status = 'ERROR' 
            continue
            
        model,generator,dictionary = name2model_dict[lang]
        res = predict_from_sample(fp_arr,model,generator,dictionary,vad_val,chunk_size,lang)
        preds.append({'source':res})
    return jsonify({"status":status, "output":preds})



@app.route("/infer_ulca_or",methods=['POST'])
@cross_origin()
def infer_ulca_or():
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    vad_val = req_data.get('vad_level',2)
    chunk_size = float(req_data.get('chunk_size',5.0))
    lang = 'or'
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = str(round(time.time() * 1000))
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
            print('data loaded')
        except:
            status = 'ERROR' 
            continue
            
        model,generator,dictionary = name2model_dict[lang]
        res = predict_from_sample(fp_arr,model,generator,dictionary,vad_val,chunk_size,lang)
        preds.append({'source':res})
    return jsonify({"status":status, "output":preds})



@app.route("/infer_ulca_mr",methods=['POST'])
@cross_origin()
def infer_ulca_mr():
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    vad_val = req_data.get('vad_level',2)
    chunk_size = float(req_data.get('chunk_size',5.0))
    lang = 'mr'
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = str(round(time.time() * 1000))
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
            print('data loaded')
        except:
            status = 'ERROR' 
            continue
            
        model,generator,dictionary = name2model_dict[lang]
        res = predict_from_sample(fp_arr,model,generator,dictionary,vad_val,chunk_size,lang)
        preds.append({'source':res})
    return jsonify({"status":status, "output":preds})



if __name__ == "__main__":
   # app.logger.setLevel(logging.DEBUG)
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    server = pywsgi.WSGIServer(('', 5000), app, keyfile="key.pem", certfile="cert.pem", handler_class=WebSocketHandler)
    print("Server listening on: http://localhost:" + str(5000))
    server.serve_forever()
