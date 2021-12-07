import gc
import itertools as it
import os.path as osp
from typing import List
import warnings
from collections import deque, namedtuple
from fairseq.data import Dictionary
import fairseq
import soundfile as sf
import numpy as np
import torch
from examples.speech_recognition.data.replabels import unpack_replabels
from fairseq import tasks
from fairseq.utils import apply_to_sample
from omegaconf import open_dict,OmegaConf
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import math, pydub, argparse

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
#                     assert (
#                         tgt_dict.unk() not in spelling_idxs
#                     ), f"{spelling} {spelling_idxs}"
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
    
def align(wavpath,target_dict):
    
    wav = pydub.AudioSegment.from_file(wavpath).set_frame_rate(16000).set_channels(1)
    sarray = wav.get_array_of_samples()
    fp_arr = np.array(sarray).T.astype(np.float64)
    fp_arr /= np.iinfo(sarray.typecode).max
    fp_arr = fp_arr.reshape(-1)

    sample = dict()
    net_input = dict()
    feature = torch.from_numpy(fp_arr).float()
    mdl[0].eval()

    net_input["source"] = feature.unsqueeze(0)

    padding_mask = torch.BoolTensor(net_input["source"].size(1)).fill_(False).unsqueeze(0)

    net_input["padding_mask"] = padding_mask
    sample["net_input"] = net_input


    with torch.no_grad():
        hypo = generator.generate(mdl, sample, prefix_tokens=None)
    hyp_pieces = target_dict.string(hypo[0][0]["tokens"].int().cpu())
    tr = hyp_pieces.replace(' ','').replace('|',' ').strip()
    return tr

def load_model_and_update(mdl_path):
    while True:
        try:
            mdl, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([mdl_path])
            print("Successfully loaded the model, processing predictions")
            break
        except:
            m = torch.load(mdl_path)
            m['cfg']['task']['_name'] = 'audio_finetuning'       
            torch.save(m,mdl_path)
    return mdl,cfg,task

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single file inference')
    parser.add_argument("--audio-file", default=None, help="path to audio file")
    parser.add_argument("--ft-model", default=None, help="path to finetuned model")
    parser.add_argument(
        "--w2l-decoder",
        choices=["viterbi", "kenlm"],
        help="use a w2l decoder",
    )
    parser.add_argument("--lexicon", help="lexicon for w2l decoder (required for KENLM)")
    parser.add_argument("--kenlm-model", help="lm model for w2l decoder (required for KENLM)")
    parser.add_argument("--beam-threshold", type=float, default=25.0)
    parser.add_argument("--beam-size-token", type=float, default=100)
    parser.add_argument("--beam", type=int, default=64)
    parser.add_argument("--word-score", type=float, default=1.0)
    parser.add_argument("--lm-weight", type=int, default=1)    
    parser.add_argument("--unk-weight", type=float, default=-math.inf)
    parser.add_argument("--sil-weight", type=float, default=0.0)
    parser.add_argument("--nbest", type=int, default=1)    
    
    args = parser.parse_args()
    print(args)
    
    mdl,cfg,task = load_model_and_update(args.ft_model)
    dargs = OmegaConf.create({'nbest':args.nbest, \
                               'lexicon':args.lexicon, \
                                'kenlm_model':args.kenlm_model, \
                                'beam_size_token': args.beam_size_token, \
                                'beam':args.beam, \
                                'beam_threshold':args.beam_threshold,\
                                'lm_weight':args.lm_weight, \
                                'word_score':args.word_score, \
                                'unk_weight':args.unk_weight, \
                                'sil_weight':args.sil_weight \
                                })
    
    if args.w2l_decoder == 'kenlm':
        generator = W2lKenLMDecoder(dargs, task.target_dictionary)
    else:
        generator = W2lViterbiDecoder(dargs, task.target_dictionary)
    print(align(args.audio_file,task.target_dictionary))
