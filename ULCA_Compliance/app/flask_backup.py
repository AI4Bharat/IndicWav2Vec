from flask import Flask, request
from flask import jsonify
from flask_cors import CORS, cross_origin
from omegaconf import OmegaConf
import numpy as np, math,torch,json
from support import load_model,W2lKenLMDecoder,W2lViterbiDecoder,load_data
import time,os

with open('test.json','r') as j:
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

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def hello_world():
    return "<p>Hi</p>"

@app.route("/infer_en",methods=['POST'])
@cross_origin()
def infer_en():
    lang = "en" 
    if not os.path.exists('../downloaded/'+lang):
        os.makedirs('../downloaded/'+lang)
    start_time = time.time()
    stp = '../downloaded/'+lang+'/'+request.files['file'].filename
    request.files['file'].save(stp)  
    fp_arr = load_data(stp,of='raw')
    feature = torch.from_numpy(fp_arr).float()
    sample = {"net_input":{"source":None,"padding_mask":None}}
    sample["net_input"]["source"] = feature.unsqueeze(0)
    sample["net_input"]["padding_mask"] = torch.BoolTensor(sample["net_input"]["source"].size(1)).fill_(False).unsqueeze(0)
    model,generator,dictionary = name2model_dict[lang]

    with torch.no_grad():
        hypo = generator.generate([model], sample, prefix_tokens=None)
    hyp_pieces = dictionary.string(hypo[0][0]["tokens"].int().cpu())
    tr = hyp_pieces.replace(' ','').replace('|',' ').strip()
    end_time = time.time()
    return jsonify({'text': tr, 'duration':round(end_time-start_time,2)})


@app.route("/infer_ulca_en",methods=['POST'])
@cross_origin()
def infer_ulca_en():
#    print(request.data)
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            #preds += str({'status': 'error'})
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = str(round(time.time() * 1000))
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
        except:
            status = 'ERROR' 
            #preds+= str({'status': 'error'})
            continue
        feature = torch.from_numpy(fp_arr).float()
        sample = {"net_input":{"source":None,"padding_mask":None}}
        sample["net_input"]["source"] = feature.unsqueeze(0)
        sample["net_input"]["padding_mask"] = torch.BoolTensor(sample["net_input"]["source"].size(1)).fill_(False).unsqueeze(0)

        model,generator,dictionary = name2model_dict['en']

        with torch.no_grad():
            hypo = generator.generate([model], sample, prefix_tokens=None)
        hyp_pieces = dictionary.string(hypo[0][0]["tokens"].int().cpu())
        tr = hyp_pieces.replace(' ','').replace('|',' ').strip()
        #preds += str({'transcript':tr,'status': 'success'})
        preds.append({'source':tr})
    return jsonify({"status":status, "output":preds})
    #return jsonify("{"+preds+"}")


@app.route("/infer_ulca_hi",methods=['POST'])
@cross_origin()
def infer_ulca_hi():
#    print(request.data)
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            #preds += str({'status': 'error'})
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = round(time.time() * 1000)
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
        except:
            status = 'ERROR' 
            #preds+= str({'status': 'error'})
            continue
        feature = torch.from_numpy(fp_arr).float()
        sample = {"net_input":{"source":None,"padding_mask":None}}
        sample["net_input"]["source"] = feature.unsqueeze(0)
        sample["net_input"]["padding_mask"] = torch.BoolTensor(sample["net_input"]["source"].size(1)).fill_(False).unsqueeze(0)

        model,generator,dictionary = name2model_dict['hi']

        with torch.no_grad():
            hypo = generator.generate([model], sample, prefix_tokens=None)
        hyp_pieces = dictionary.string(hypo[0][0]["tokens"].int().cpu())
        tr = hyp_pieces.replace(' ','').replace('|',' ').strip()
        #preds += str({'transcript':tr,'status': 'success'})
        preds.append({'source':tr})
    return jsonify({"status":status, "output":preds})
    #return jsonify("{"+preds+"}")


@app.route("/infer_ulca_ta",methods=['POST'])
@cross_origin()
def infer_ulca_ta():
#    print(request.data)
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            #preds += str({'status': 'error'})
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = round(time.time() * 1000)
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
        except:
            status = 'ERROR' 
            #preds+= str({'status': 'error'})
            continue
        feature = torch.from_numpy(fp_arr).float()
        sample = {"net_input":{"source":None,"padding_mask":None}}
        sample["net_input"]["source"] = feature.unsqueeze(0)
        sample["net_input"]["padding_mask"] = torch.BoolTensor(sample["net_input"]["source"].size(1)).fill_(False).unsqueeze(0)

        model,generator,dictionary = name2model_dict['ta']

        with torch.no_grad():
            hypo = generator.generate([model], sample, prefix_tokens=None)
        hyp_pieces = dictionary.string(hypo[0][0]["tokens"].int().cpu())
        tr = hyp_pieces.replace(' ','').replace('|',' ').strip()
        #preds += str({'transcript':tr,'status': 'success'})
        preds.append({'source':tr})
    return jsonify({"status":status, "output":preds})
    #return jsonify("{"+preds+"}")


@app.route("/infer_ulca_te",methods=['POST'])
@cross_origin()
def infer_ulca_te():
#    print(request.data)
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            #preds += str({'status': 'error'})
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = round(time.time() * 1000)
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
        except:
            status = 'ERROR' 
            #preds+= str({'status': 'error'})
            continue
        feature = torch.from_numpy(fp_arr).float()
        sample = {"net_input":{"source":None,"padding_mask":None}}
        sample["net_input"]["source"] = feature.unsqueeze(0)
        sample["net_input"]["padding_mask"] = torch.BoolTensor(sample["net_input"]["source"].size(1)).fill_(False).unsqueeze(0)

        model,generator,dictionary = name2model_dict['te']

        with torch.no_grad():
            hypo = generator.generate([model], sample, prefix_tokens=None)
        hyp_pieces = dictionary.string(hypo[0][0]["tokens"].int().cpu())
        tr = hyp_pieces.replace(' ','').replace('|',' ').strip()
        #preds += str({'transcript':tr,'status': 'success'})
        preds.append({'source':tr})
    return jsonify({"status":status, "output":preds})
    #return jsonify("{"+preds+"}")


@app.route("/infer_ulca_gu",methods=['POST'])
@cross_origin()
def infer_ulca_gu():
#    print(request.data)
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            #preds += str({'status': 'error'})
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = round(time.time() * 1000)
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
        except:
            status = 'ERROR' 
            #preds+= str({'status': 'error'})
            continue
        feature = torch.from_numpy(fp_arr).float()
        sample = {"net_input":{"source":None,"padding_mask":None}}
        sample["net_input"]["source"] = feature.unsqueeze(0)
        sample["net_input"]["padding_mask"] = torch.BoolTensor(sample["net_input"]["source"].size(1)).fill_(False).unsqueeze(0)

        model,generator,dictionary = name2model_dict['gu']

        with torch.no_grad():
            hypo = generator.generate([model], sample, prefix_tokens=None)
        hyp_pieces = dictionary.string(hypo[0][0]["tokens"].int().cpu())
        tr = hyp_pieces.replace(' ','').replace('|',' ').strip()
        #preds += str({'transcript':tr,'status': 'success'})
        preds.append({'source':tr})
    return jsonify({"status":status, "output":preds})
    #return jsonify("{"+preds+"}")


@app.route("/infer_ulca_bn",methods=['POST'])
@cross_origin()
def infer_ulca_bn():
#    print(request.data)
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            #preds += str({'status': 'error'})
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = round(time.time() * 1000)
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
        except:
            status = 'ERROR' 
            #preds+= str({'status': 'error'})
            continue
        feature = torch.from_numpy(fp_arr).float()
        sample = {"net_input":{"source":None,"padding_mask":None}}
        sample["net_input"]["source"] = feature.unsqueeze(0)
        sample["net_input"]["padding_mask"] = torch.BoolTensor(sample["net_input"]["source"].size(1)).fill_(False).unsqueeze(0)

        model,generator,dictionary = name2model_dict['bn']

        with torch.no_grad():
            hypo = generator.generate([model], sample, prefix_tokens=None)
        hyp_pieces = dictionary.string(hypo[0][0]["tokens"].int().cpu())
        tr = hyp_pieces.replace(' ','').replace('|',' ').strip()
        #preds += str({'transcript':tr,'status': 'success'})
        preds.append({'source':tr})
    return jsonify({"status":status, "output":preds})
    #return jsonify("{"+preds+"}")


@app.route("/infer_ulca_ne",methods=['POST'])
@cross_origin()
def infer_ulca_ne():
#    print(request.data)
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            #preds += str({'status': 'error'})
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = round(time.time() * 1000)
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
        except:
            status = 'ERROR' 
            #preds+= str({'status': 'error'})
            continue
        feature = torch.from_numpy(fp_arr).float()
        sample = {"net_input":{"source":None,"padding_mask":None}}
        sample["net_input"]["source"] = feature.unsqueeze(0)
        sample["net_input"]["padding_mask"] = torch.BoolTensor(sample["net_input"]["source"].size(1)).fill_(False).unsqueeze(0)

        model,generator,dictionary = name2model_dict['ne']

        with torch.no_grad():
            hypo = generator.generate([model], sample, prefix_tokens=None)
        hyp_pieces = dictionary.string(hypo[0][0]["tokens"].int().cpu())
        tr = hyp_pieces.replace(' ','').replace('|',' ').strip()
        #preds += str({'transcript':tr,'status': 'success'})
        preds.append({'source':tr})
    return jsonify({"status":status, "output":preds})
    #return jsonify("{"+preds+"}")


@app.route("/infer_ulca_si",methods=['POST'])
@cross_origin()
def infer_ulca_si():
#    print(request.data)
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            #preds += str({'status': 'error'})
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = round(time.time() * 1000)
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
        except:
            status = 'ERROR' 
            #preds+= str({'status': 'error'})
            continue
        feature = torch.from_numpy(fp_arr).float()
        sample = {"net_input":{"source":None,"padding_mask":None}}
        sample["net_input"]["source"] = feature.unsqueeze(0)
        sample["net_input"]["padding_mask"] = torch.BoolTensor(sample["net_input"]["source"].size(1)).fill_(False).unsqueeze(0)

        model,generator,dictionary = name2model_dict['si']

        with torch.no_grad():
            hypo = generator.generate([model], sample, prefix_tokens=None)
        hyp_pieces = dictionary.string(hypo[0][0]["tokens"].int().cpu())
        tr = hyp_pieces.replace(' ','').replace('|',' ').strip()
        #preds += str({'transcript':tr,'status': 'success'})
        preds.append({'source':tr})
    return jsonify({"status":status, "output":preds})
    #return jsonify("{"+preds+"}")


@app.route("/infer_ulca_or",methods=['POST'])
@cross_origin()
def infer_ulca_or():
#    print(request.data)
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            #preds += str({'status': 'error'})
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = round(time.time() * 1000)
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
        except:
            status = 'ERROR' 
            #preds+= str({'status': 'error'})
            continue
        feature = torch.from_numpy(fp_arr).float()
        sample = {"net_input":{"source":None,"padding_mask":None}}
        sample["net_input"]["source"] = feature.unsqueeze(0)
        sample["net_input"]["padding_mask"] = torch.BoolTensor(sample["net_input"]["source"].size(1)).fill_(False).unsqueeze(0)

        model,generator,dictionary = name2model_dict['or']

        with torch.no_grad():
            hypo = generator.generate([model], sample, prefix_tokens=None)
        hyp_pieces = dictionary.string(hypo[0][0]["tokens"].int().cpu())
        tr = hyp_pieces.replace(' ','').replace('|',' ').strip()
        #preds += str({'transcript':tr,'status': 'success'})
        preds.append({'source':tr})
    return jsonify({"status":status, "output":preds})
    #return jsonify("{"+preds+"}")


@app.route("/infer_ulca_mr",methods=['POST'])
@cross_origin()
def infer_ulca_mr():
#    print(request.data)
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            #preds += str({'status': 'error'})
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = round(time.time() * 1000)
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
        except:
            status = 'ERROR' 
            #preds+= str({'status': 'error'})
            continue
        feature = torch.from_numpy(fp_arr).float()
        sample = {"net_input":{"source":None,"padding_mask":None}}
        sample["net_input"]["source"] = feature.unsqueeze(0)
        sample["net_input"]["padding_mask"] = torch.BoolTensor(sample["net_input"]["source"].size(1)).fill_(False).unsqueeze(0)

        model,generator,dictionary = name2model_dict['mr']

        with torch.no_grad():
            hypo = generator.generate([model], sample, prefix_tokens=None)
        hyp_pieces = dictionary.string(hypo[0][0]["tokens"].int().cpu())
        tr = hyp_pieces.replace(' ','').replace('|',' ').strip()
        #preds += str({'transcript':tr,'status': 'success'})
        preds.append({'source':tr})
    return jsonify({"status":status, "output":preds})
    #return jsonify("{"+preds+"}")

if __name__ == "__main__":
   # app.logger.setLevel(logging.DEBUG)
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    server = pywsgi.WSGIServer(('', 5000), app)
    print("Server listening on: http://localhost:" + str(5000))
    server.serve_forever()
