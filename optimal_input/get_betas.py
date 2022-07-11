from auditory_cortex.feature_extractors import FeatureExtractorW2L
from auditory_cortex.regression import transformer_regression

from scipy.io.wavfile import read
from scipy import signal
import numpy as np


class GetBetas():
    def __init__(self, data_dir, sub, model) -> None:
        self.reg = transformer_regression(data_dir, sub, load_features=False)
        self.extractor = FeatureExtractorW2L(model)
    
    def lin_reg(self, z, n):
        return np.linalg.solve(z.T.dot(z), (z.T).dot(n))

    def betas(self, sent, bin_width, layer):
        spikes = self.reg.all_channel_spikes(bin_width = bin_width, sents = [sent])
        _, aud = read(f"/depot/jgmakin/data/audio_data/sent_{sent}.wav")
        self.extractor.translate(aud)
        feat = self.extractor.get_features(layer)
        s = np.array([spikes[k] for k in range(len(spikes.keys()))])
        resampled_features = signal.resample(feat, spikes[0].shape[0], axis=0)

        return self.lin_reg(resampled_features, s.T)





