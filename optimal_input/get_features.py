from gradient_extraction.model import TransformerModel
from auditory_cortex.dataset import Neural_Data


def down_sample_features(self, feats, k):
    out = np.zeros((int(np.ceil(feats.shape[0]/k)),feats.shape[1]))
    # print(out.shape)
    for i in range(out.shape[0]):
    #Just add the remaining samples at the end...!
        if (i == out.shape[0] -1):
            out[i] = feats[k*i:, :].sum(axis=0)
        else:  
            out[i] = feats[k*i:k*(i+1), :].sum(axis=0)
    return out


def down_sample_spikes(self, spks, k):
    # out = np.zeros(int(np.ceil(spks.shape[0]/k)))
    out = np.zeros((int(np.ceil(spks.shape[0]/k)), spks.shape[1]))
    for i in range(out.shape[0]):
    #Just add the remaining samples at the end...!
        if (i == out.shape[0] -1):
            out[i] = spks[k*i:, :].sum(axis=0)
        else:  
            out[i] = spks[k*i:k*(i+1), :].sum(axis=0)
    return out
    
class GetFeatures:
    def __init__(self, dir, subject, delay=0):
        self.model = TransformerModel()
        self.neural_data = Neural_Data(dir, subject)

        self.get_transformer_features()


    def translate(self, aud, fs = 16000):
        inputs_features = self.model.processor(aud,padding=True, sampling_rate=fs, return_tensors="pt").input_features
        generated_ids = self.model.model_extractor(inputs_features)

    def get_transformer_features(self):
        features = [{} for _ in range(len(self.model.layers))]
        for i in range(1,499):
            self.translate(self.neural_data.audio(i))

            for j, l in enumerate(self.model.layers):
                features[j][i] = self.model.model_extractor.features[l]

        self.all_hidden_features = features
        
    def get_spikes(self, delay=0):
        spikes = [{}, {}]
        for i in range(1,499):
            spikes[0][i] = self.neural_data.retrieve_spike_counts(sent=i, win=20, delay=delay, 
                                                                early_spikes=False,
                                                                model=self.model_name, 
                                                                offset=-0.25)
            spikes[1][i] = self.neural_data.retrieve_spike_counts(sent=i, win=40, delay=delay, 
                                                                early_spikes=False,
                                                                model=self.model_name,
                                                                offset=0.39)

        self.all_spikes_dict = spikes