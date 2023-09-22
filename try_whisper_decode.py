import os
import numpy as np
import csv
from tqdm import tqdm
try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
import pandas as pd
from whisper.audio import pad_or_trim, log_mel_spectrogram
from whisper.__init__ import load_model
from whisper.decoding import DecodingOptions
from whisper.tokenizer import get_tokenizer, get_encoding
import torchaudio
from datasetloader import HuggingfaceDataset

from pydub import AudioSegment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = HuggingfaceDataset("DynamicSuperb/SpeechDetection_LibriSpeech-TestClean", n_example=2)
# loader = torch.utils.data.DataLoader(dataset, batch_size=16)


def exclude_tokens_from_max_token_id(tokens, max_token_id):
    ans = list(range(tokens[0]))
    for i in range(len(tokens)):
        if i == len(tokens)-1:
            ans.extend(list(range(tokens[i]+1, max_token_id+1)))
            return ans
        ans.extend(list(range(tokens[i]+1, tokens[i+1])))

tokenizer = get_tokenizer(False) # monolingual tokenizer
labels = dataset.labels
encoded_tokens = [tokenizer.encode(label) for label in labels]

label_map_token = { encoded_tokens[i][0] : labels[i] for i in range(len(labels))}

tokens = []
for encodeds in encoded_tokens:
    tokens = list(set(tokens) | set(encodeds))
tokens.sort()
max_token_id = get_encoding().max_token_value
suppress_tokens = exclude_tokens_from_max_token_id(tokens, max_token_id)
ans=[]
correct = 0
tot = 0
with torch.no_grad():
    for idx, (mel, prompt, eval_label) in tqdm(enumerate(dataset)):
        tot += 1
        model = load_model("small.en")
        options = DecodingOptions(language = "en", 
                                without_timestamps = True, 
                                suppress_tokens = suppress_tokens,
                                prompt=prompt)
        result = model.decode(mel, options)
        ans.append([idx, result.text, eval_label])

        if eval_label == label_map_token[result.tokens[0]] :
            correct += 1

with open('output_speech_detection_2_shot.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['index', 'result_text', 'label'])
    for a in ans:
        writer.writerow(a)
    

print(correct/tot)