from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils 
import commons
import sys
import re
import yaml
from torch import no_grad, LongTensor
import logging

import simpleaudio

logging.getLogger('numba').setLevel(logging.WARNING)

class TTS_Engine():
    def __init__(self, yaml_path: str):
        try:
            __data: dict = self.__open_yaml(yaml_path)
            # replace existing input course to yaml file
            self.vits_model_path = __data['VITSModelPath']
            print(f"model:{self.vits_model_path}")
            self.config_file_path = __data['configFilePath']
            print(f"config:{self.config_file_path}")
            self.speaker_id = __data['speakerID']
            print(f"speakerid: {self.speaker_id}")
            self.hparams_ms = utils.get_hparams_from_file(self.config_file_path)

            n_speakers = self.hparams_ms.data.n_speakers if 'n_speakers' in self.hparams_ms.data.keys() else 0
            n_symbols = len(self.hparams_ms.symbols) if 'symbols' in self.hparams_ms.keys() else 0
            print(f"n_symbols: {n_symbols}")
            speakers = self.hparams_ms.speakers if 'speakers' in self.hparams_ms.keys() else ['0']
            print(f"speakers: {speakers}")
            use_f0 = self.hparams_ms.data.use_f0 if 'use_f0' in self.hparams_ms.data.keys() else False
            print(f"use_f0: {use_f0}")
            emotion_embedding = self.hparams_ms.data.emotion_embedding if 'emotion_embedding' in self.hparams_ms.data.keys() else False

            self.net_g_ms = SynthesizerTrn(
                n_symbols,
                self.hparams_ms.data.filter_length // 2 + 1,
                self.hparams_ms.train.segment_size,
                n_speakers=n_speakers,
                emotion_embedding=emotion_embedding,
                **self.hparams_ms.model)
            _ = self.net_g_ms.eval() 
            utils.load_checkpoint(self.vits_model_path, self.net_g_ms)
        except Exception as e:
            print(e)
            sys.exit(1)

        

    def __open_yaml(self, yaml_path: str) -> dict:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return data
    
    def __get_label_value(self, text:str, label:str, default, warning_name:str='value'):
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

    def __get_label(self, text:str, label:str):
        if f'[{label}]' in text:
            return True, text.replace(f'[{label}]', '')
        else:
            return False, text

    def __get_text(self, text:str, hps:utils.HParams, cleaned=False):
        if cleaned:
            text_normalized = text_to_sequence(text, hps.symbols, [])
        else:
            text_normalized = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_normalized = commons.intersperse(text_normalized, 0)
        text_normalized = LongTensor(text_normalized)
        return text_normalized
    
    def text_to_audio(self, text: str):
        length_scale, text = self.__get_label_value(text, 'LENGTH', 1, 'length scale')
        print(f"length_scale: {length_scale}")
        noise_scale, text = self.__get_label_value(text, 'NOISE', 0.667, 'noise scale')
        print(f"noise_scale: {noise_scale}")
        noise_scale_w, text = self.__get_label_value(text, 'NOISEW', 0.8, 'deviation of noise')
        print(f"noise_scale_w: {noise_scale_w}")
        cleaned, text = self.__get_label(text, 'CLEANED')
        print(f"cleaned: {cleaned}")

        sequence_text_normalized = self.__get_text(text, self.hparams_ms, cleaned=cleaned)
        print(f"sequence_text_normalized: {sequence_text_normalized}")  
        with no_grad():
            x_tst = sequence_text_normalized.unsqueeze(0)
            x_tst_lengths = LongTensor([sequence_text_normalized.size(0)])
            sid = LongTensor([self.speaker_id])
            audio = self.net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                        noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
        write('audio.wav', self.hparams_ms.data.sampling_rate, audio)
        

if __name__ == '__main__':
    try:
        tts_engine = TTS_Engine('config.yaml')
        tts_engine.text_to_audio("こんにちは")
    except Exception as e:
        print(e)
        sys.exit(1)
    
    


    


