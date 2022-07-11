import wav2letter
from wav2letter.models import Wav2LetterRF

from optimal_input.opt_inp_w2l import GetOptInput

from scipy.io.wavfile import read
import yaml
import os
import numpy as np
import torch
from tqdm import tqdm

print("started")

dir = os.getcwd()
conf_file = 'config_rf.yaml'
manifest_file = os.path.join(os.path.dirname(wav2letter.__file__),"conf",conf_file)
with open(manifest_file, 'r') as f:
    model_param = yaml.load(f, Loader=yaml.FullLoader)


checkpoint_file = "Wav2letter-epoch=024-val_loss=0.37.ckpt"
checkpoint = os.path.join(model_param["results_dir"],checkpoint_file)
model = Wav2LetterRF.load_from_checkpoint(checkpoint, manifest=model_param)

for param in model.parameters():
    param.requires_grad = False
print("loaded model")

layers = ["conv" + str(x) for x in range(1, 16)]

get_opt_input = GetOptInput(model)

beta = np.load("/depot/jgmakin/data/auditory_cortex/betas/w2l_l5.npy", allow_pickle=True)

l = 5
print("starting")
for i in [31, 32]:
    _ , aud = read("/depot/jgmakin/data/audio_data/sent_0.wav")
    aud = aud[:7500]
    get_opt_input.get_opt_input(torch.tensor(aud.astype('float32')).unsqueeze(0).T, layers[l], beta[:, i].reshape(-1, 1).astype('float32'), iterations=100)
    np.save(f'/depot/jgmakin/data/auditory_cortex/opt_inputs/w2l/loss_layer_{l}_channel_{i}.npy', get_opt_input.loss_list, allow_pickle=True)
    np.save(f'/depot/jgmakin/data/auditory_cortex/opt_inputs/w2l/layer_{l}_channel_{i}.npy', get_opt_input.aud.detach().numpy(), allow_pickle=True)