from datasets import load_dataset
from datasets import load_dataset
import torch
import torchaudio
import math
from whisper.audio import pad_or_trim, log_mel_spectrogram
from whisper.__init__ import load_model
from whisper.decoding import DecodingOptions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dataset_name = "DynamicSuperb/SpeechDetection_LibriSpeech-TestClean"

class HuggingfaceDataset(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """

    def __init__(self, dataset_name, train_split=['test[:80%]'], eval_split=['test[80%:]'], device=DEVICE, n_example = 0):
        self.train_dataset = load_dataset(dataset_name, split = train_split)[0]
        self.eval_dataset = load_dataset(dataset_name, split = eval_split)[0]
        set_list = set(self.train_dataset['label'])
        self.labels = list(set_list)
        self.device = device
        self.n_example = n_example

    def __len__(self):
        return len(self.train_dataset) + len(self.eval_dataset)

    def __getitem__(self, item):
        train_instances = self.train_dataset[4*item:4*item+self.n_example]
        eval_instance = self.eval_dataset[item]
        audio = []
        for t in train_instances['audio']:
            audio.extend(t['array'].tolist()+[0.0]*100)
        audio.extend(eval_instance['audio']['array'].tolist())
        
        audio = pad_or_trim(torch.tensor(audio)).to(self.device)
        mel = log_mel_spectrogram(audio)

        prompt = ('Analyze the audio and determine whether it consists of real speech or not. The answer could be yes or no. Answer:' + ' '.join(train_instances['label']))

        return (mel, prompt, eval_instance['label'])
