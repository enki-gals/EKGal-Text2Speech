from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
from utils import HParams, load_checkpoint, get_hparams_from_file
import commons
import sys
import re
import yaml
from torch import no_grad, LongTensor
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

class TTS_Engine():
    def __init__(self, yaml_path: str):
        __data: dict = self.__open_yaml(yaml_path)
        # replace existing input course to yaml file
        self.vits_model_path = __data['VITSModelPath']
        self.config_file_path = __data['configFilePath']
        self.speaker_id = __data['speakerID']
        self.hparams_ms = get_hparams_from_file(self.config_file_path)

        def init_moegoe(vits_model_path:str, config_file_path:str, hparams_ms:HParams)->SynthesizerTrn:
            n_speakers = hparams_ms.data.n_speakers if 'n_speakers' in hparams_ms.data.keys() else 0
            n_symbols = len(hparams_ms.symbols) if 'symbols' in hparams_ms.keys() else 0
            speakers = hparams_ms.speakers if 'speakers' in hparams_ms.keys() else ['0']
            use_f0 = hparams_ms.data.use_f0 if 'use_f0' in hparams_ms.data.keys() else False
            emotion_embedding = hparams_ms.data.emotion_embedding if 'emotion_embedding' in self.hparams_ms.data.keys() else False

            net_g_ms = SynthesizerTrn(
                n_symbols,
                hparams_ms.data.filter_length // 2 + 1,
                hparams_ms.train.segment_size,
                n_speakers,
                emotion_embedding=emotion_embedding,
                **hparams_ms.model)
            _ = net_g_ms.eval()
            return net_g_ms
        
        self.net_g_ms:SynthesizerTrn = init_moegoe(self.vits_model_path, self.config_file_path)
        load_checkpoint(self.vits_model_path)

        

    def __open_yaml(self, yaml_path: str) -> dict:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return data
    
    def __get_label_value(self, text:str, label:str, default, warning_name:str='value')->tuple[float, float]:
        value = re.search(rf'\[{label}=(.+?)\]', text)
        if value:
            try:
                text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
                value = float(value.group(1))
            except:
                print(f'Invalid {warning_name}!')
                sys.exit(1)
        else:
            value = default
        return value, text

    def __get_label(self, text:str, label:str)->tuple[bool, str]:
        if f'[{label}]' in text:
            return True, text.replace(f'[{label}]', '')
        else:
            return False, text

    def __get_text(self, text:str, hps:HParams, cleaned:bool=False)->LongTensor:
        if cleaned:
            text_normalized = text_to_sequence(text, hps.symbols, [])
        else:
            text_normalized = text_to_sequence(_clean_text(text), hps.symbols, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_normalized = commons.intersperse(text_normalized, 0)
        text_normalized = LongTensor(text_normalized)
        return text_normalized
    
    def text_to_audio(self, text: str):
        length_scale, text = self.__get_label_value(text, 'LENGTH', 1, 'length scale')
        noise_scale, text = self.__get_label_value(text, 'NOISE', 0.667, 'noise scale')
        noise_scale_w, text = self.__get_label_value(text, 'NOISEW', 0.8, 'deviation of noise')
        cleaned, text = self.__get_label(text, 'CLEANED')

        sequence_text_normalized = self.__get_text(text, self.hparams_ms, cleaned)
        with no_grad():
            x_tst = sequence_text_normalized.unsqueeze(0)
            x_tst_lengths = LongTensor([x_tst.size(0)])
            sid = LongTensor([self.speaker_id])
            audio = self.net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                        noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
        write('audio.wav', self.hparams_ms.data.sampling_rate, audio)
    
    


    


