from transformers import Speech2TextFeatureExtractor, Speech2TextModel
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from optimal_input.opt_input_with_betas import GetOptInput


model = Speech2TextModel.from_pretrained("facebook/s2t-small-librispeech-asr")
feature_extractor = Speech2TextFeatureExtractor.from_pretrained("facebook/s2t-small-librispeech-asr")
# reg = transformer_regression(dir, subject)

for param in model.parameters():
    param.requires_grad = False

layers = ["encoder.conv.conv_layers.0","encoder.conv.conv_layers.1","encoder.layers.0.fc2",
			"encoder.layers.1.fc2","encoder.layers.2.fc2","encoder.layers.3.fc2",
			"encoder.layers.4.fc2","encoder.layers.5.fc2","encoder.layers.6.fc2",
			"encoder.layers.7.fc2","encoder.layers.8.fc2","encoder.layers.9.fc2"]

a = np.load('/depot/jgmakin/data/auditory_cortex/betas/layer_0.npy', allow_pickle=True)
b = np.load('/depot/jgmakin/data/auditory_cortex/betas/layer_1.npy', allow_pickle=True)

# sr, aud = read("/depot/jgmakin/data/audio_data/sent_0.wav")
get_opt_input = GetOptInput(model, feature_extractor)

loss_l0 = []
spect_l0 = []

for i in range(64):
    get_opt_input.get_opt_input(layers[1], b[:, i].reshape(-1, 1).astype('float32'), 30, iterations=100)
    loss_l0.append(get_opt_input.loss_list)
    spect_l0.append(get_opt_input.spect[0].detach().numpy().T)
    
np.save('/depot/jgmakin/data/auditory_cortex/opt_inputs/layer_1_64_channels.npy', spect_l0 ,allow_pickle=True)