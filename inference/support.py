import time
import gc
import itertools as it
import os,math
import pydub
import os.path as osp
from typing import List
import warnings
from collections import deque, namedtuple
from fairseq.data import Dictionary
import fairseq
import soundfile as sf,wave
import numpy as np
import torch
import urllib.response, requests
from examples.speech_recognition.data.replabels import unpack_replabels
from fairseq import tasks
from fairseq.utils import apply_to_sample
from omegaconf import open_dict, OmegaConf
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import base64
import uuid
#from vad_old import read_wave
import subprocess
try:
    from flashlight.lib.text.dictionary import create_word_dict, load_words
    from flashlight.lib.sequence.criterion import CpuViterbiPath, get_data_ptr_as_bytes
    from flashlight.lib.text.decoder import (
        CriterionType,
        LexiconDecoderOptions,
        KenLM,
        LM,
        LMState,
        SmearingMode,
        Trie,
        LexiconDecoder,
    )
except:
    warnings.warn(
        "flashlight python bindings are required to use this functionality. Please install from https://github.com/facebookresearch/flashlight/tree/master/bindings/python"
    )
    LM = object
    LMState = object


class W2lDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest

        # criterion-specific init
        self.criterion_type = CriterionType.CTC
        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        if "<sep>" in tgt_dict.indices:
            self.silence = tgt_dict.index("<sep>")
        elif "|" in tgt_dict.indices:
            self.silence = tgt_dict.index("|")
        else:
            self.silence = tgt_dict.eos()
        self.asg_transitions = None

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        model = models[0]
        encoder_out = model(**encoder_input)
        if hasattr(model, "get_logits"):
            emissions = model.get_logits(encoder_out) # no need to normalize emissions
        else:
            emissions = model.get_normalized_probs(encoder_out, log_probs=True)
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))

class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        if self.asg_transitions is None:
            transitions = torch.FloatTensor(N, N).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view(N, N)
        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
        CpuViterbiPath.compute(
            B,
            T,
            N,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [
            [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}]
            for b in range(B)
        ]
    
class W2lKenLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        self.unit_lm = getattr(args, "unit_lm", False)

        if args.lexicon:
            self.lexicon = load_words(args.lexicon)
            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index("<unk>")

            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence)

            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)
                for spelling in spellings:
                    spelling_idxs = [tgt_dict.index(token) for token in spelling]
                    # assert (
                    #     tgt_dict.unk() not in spelling_idxs
                    # ), f"{spelling} {spelling_idxs}"
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)
            
            self.decoder_opts = LexiconDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                word_score=args.word_score,
                unk_score=args.unk_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )

            if self.asg_transitions is None:
                N = 768
                # self.asg_transitions = torch.FloatTensor(N, N).zero_()
                self.asg_transitions = []

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                self.asg_transitions,
                self.unit_lm,
            )
        else:
            assert args.unit_lm, "lexicon free decoding can only be done with a unit language model"
            from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions

            d = {w: [[w]] for w in tgt_dict.symbols}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(args.kenlm_model, self.word_dict)
            
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence, self.blank, []
            )
            
    def get_timesteps(self, token_idxs: List[int]) -> List[int]:
        """Returns frame numbers corresponding to every non-blank token.
        Parameters
        ----------
        token_idxs : List[int]
            IDs of decoded tokens.
        Returns
        -------
        List[int]
            Frame numbers corresponding to every non-blank token.
        """
        timesteps = []
        for i, token_idx in enumerate(token_idxs):
            if token_idx == self.blank:
                continue
            if i == 0 or token_idx != token_idxs[i-1]:
                timesteps.append(i)
        return timesteps

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append(
                [
                    {
                        "tokens": self.get_tokens(result.tokens),
                        "score": result.score,
                        "timesteps": self.get_timesteps(result.tokens),
                        "words": [
                            self.word_dict.get_entry(x) for x in result.words if x >= 0
                        ],
                    }
                    for result in nbest_results
                ]
            )
        return hypos

def load_model(mdl_path):
    while True:
        try:
            print("Loading model..")
            mdl, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([mdl_path])
            print("Successfully loaded model "+mdl_path)
            break

        except:
            print(f"Model loading failed for path: {mdl_path}. Retrying..")
            m = torch.load(mdl_path)
            m['cfg']['task']['_name'] = 'audio_finetuning'
            torch.save(m,mdl_path)
    return mdl[0].eval(), task.target_dictionary

import soundfile as sf
DOWNLOAD_FOLDER = 'media/'
HEADERS = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"}
def load_data(wavpath,of='raw',**extra):
    print("Wavpath", wavpath)
    print("of", of)
    if of == 'raw':    
        # wav, _ = read_wave(wavpath)
        # wav = pydub.AudioSegment.from_file(wavpath).set_frame_rate(16000).set_sample_width(2).set_channels(1)
        #if os.path.exists("test.wav"):
        #    os.remove("test.wav")
        subprocess.call(['ffmpeg','-y', '-i', wavpath,'-ar', '16000', '-ac', '1', '-hide_banner', '-loglevel', 'error', wavpath+'_new.wav'])
        
        #os.remove(wavpath)
        #wavpath = wavpath+'_new.wav'
        wav= pydub.AudioSegment.from_file(wavpath+'_new.wav', sample_width=2, frame_rate=16000, channels=1)
    elif of == 'url': 
        if not os.path.exists(DOWNLOAD_FOLDER):
            os.makedirs(DOWNLOAD_FOLDER)
        file_id = uuid.uuid4().hex[:6].upper()
        # urllib.request.urlretrieve(wavpath, DOWNLOAD_FOLDER+file_id)
        try:
            print("Downloading file..")
            resp = requests.get(wavpath, headers=HEADERS).content
            with open(DOWNLOAD_FOLDER+file_id, "wb") as f:
                f.write(resp)
            print("Audio is saved")
        except Exception as e:
            print(e)
        print("wavpath", wavpath)
        print("downloads", DOWNLOAD_FOLDER+file_id)
        subprocess.call(['ffmpeg', '-i', DOWNLOAD_FOLDER+file_id,'-ar', '16k', '-ac', '1', '-hide_banner', '-loglevel', 'error', DOWNLOAD_FOLDER+file_id+'new.wav'])
        if os.path.exists(DOWNLOAD_FOLDER+file_id):
            os.remove(DOWNLOAD_FOLDER+file_id)
        return load_data(DOWNLOAD_FOLDER+file_id+'new.wav')
        # return load_data(DOWNLOAD_FOLDER+file_id)
    elif of == 'bytes':
        lang = extra['lang']
        name = extra['bytes_name']
        if not os.path.exists(DOWNLOAD_FOLDER):
            os.makedirs(DOWNLOAD_FOLDER)
        with wave.open(DOWNLOAD_FOLDER+name, 'wb') as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(16000)
            file.writeframes(base64.b64decode(wavpath))
        return load_data(DOWNLOAD_FOLDER+name)
    
    # sarray = wav.get_array_of_samples()
    # fp_arr = np.array(sarray).T.astype(np.float64)
    # fp_arr /= np.iinfo(sarray.typecode).max
    # fp_arr = fp_arr.reshape(-1)
    # return fp_arr

    return wav.raw_data
