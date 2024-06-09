from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils 
import commons
import sys
import io
import re
import yaml
from torch import no_grad, LongTensor
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

class TTS_Engine():
    """
    Read the voice model and settings from the config file and perform the tts function of moegoe.
    ...

    Attributes
    ----------
    vits_model_path : str
        Path of the Model file
    config_file_path : str
        Path of the Config file
    speaker_id : int
        voice actor id
    hparams_ms : HParams
        ...
    net_g_ms : SynthesizerTrn
        ...
    Methods
    -------
    text_to_audio(self, text:str)
        Convert text to wav byte stream
    """
    def __init__(self, yaml_path: str):
        """
        Parameters
        ----------
        yaml_path : str
            Path of the yaml file
        
        yaml file should have the following format:
        ```
        VITSModelPath: 'path/to/model.pth'
        configFilePath: 'path/to/config.yaml'
        speakerID: 0 'speaker id'
        ```
        
        Raises
        ------
        Exception
            If there is an error in the process
        """
        try:
            __data: dict = self.__open_yaml(yaml_path)
            # replace existing input course to yaml file
            self.vits_model_path = __data['VITSModelPath']
            self.config_file_path = __data['configFilePath']
            self.speaker_id = __data['speakerID']
            self.hparams_ms = utils.get_hparams_from_file(self.config_file_path)
            
            # init Moegoe sequence
            self.__init_moegoe() 
            _ = self.net_g_ms.eval() 
            utils.load_checkpoint(self.vits_model_path, self.net_g_ms)
        except Exception as e:
            raise Exception(f'Error in TTS_Engine __init__: {e}')


    def __open_yaml(self, yaml_path: str) -> dict:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return data
    
    def __init_moegoe(self):
        n_speakers = self.hparams_ms.data.n_speakers if 'n_speakers' in self.hparams_ms.data.keys() else 0
        n_symbols = len(self.hparams_ms.symbols) if 'symbols' in self.hparams_ms.keys() else 0
        speakers = self.hparams_ms.speakers if 'speakers' in self.hparams_ms.keys() else ['0']
        use_f0 = self.hparams_ms.data.use_f0 if 'use_f0' in self.hparams_ms.data.keys() else False
        emotion_embedding = self.hparams_ms.data.emotion_embedding if 'emotion_embedding' in self.hparams_ms.data.keys() else False
        #init object
        self.net_g_ms = SynthesizerTrn(
            n_symbols,
            self.hparams_ms.data.filter_length // 2 + 1,
            self.hparams_ms.train.segment_size,
            n_speakers=n_speakers,
            emotion_embedding=emotion_embedding,
            **self.hparams_ms.model)
    
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
        """
        Convert text to wav byte stream.
        
        Parameters
        ----------
        text : str
            Text to be converted to audio
        
        Raises
        ------
        Exception
            If there is an error in the process
            
        Returns
        -------
        bytes_io.read() : bytes
            Wav byte stream

        """
        try:
            length_scale, text = self.__get_label_value(text, 'LENGTH', 1, 'length scale')
            noise_scale, text = self.__get_label_value(text, 'NOISE', 0.667, 'noise scale')
            noise_scale_w, text = self.__get_label_value(text, 'NOISEW', 0.8, 'deviation of noise')
            cleaned, text = self.__get_label(text, 'CLEANED')

            sequence_text_normalized = self.__get_text(text, self.hparams_ms, cleaned=cleaned)
            with no_grad():
                x_tst = sequence_text_normalized.unsqueeze(0)
                x_tst_lengths = LongTensor([sequence_text_normalized.size(0)])
                sid = LongTensor([self.speaker_id])
                audio = self.net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                            noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
            bytes_wav = bytes()
            bytes_io = io.BytesIO(bytes_wav)
            write(bytes_io, self.hparams_ms.data.sampling_rate, audio)
            return bytes_io.read()

        except Exception as e:
            raise Exception(f'Error in text_to_audio: {e}')

if __name__ == '__main__':
    try:
        tts_engine = TTS_Engine('config.yaml')
        stream = tts_engine.text_to_audio('私は今日の状況を見なければなりません。')
        with open('output.wav', 'wb') as f:
            f.write(stream)
    except Exception as e:
        print(e)
        sys.exit(1)
    
    


    


