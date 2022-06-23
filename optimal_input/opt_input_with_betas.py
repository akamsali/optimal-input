import torch
from optimal_input.hook import Hook


class GetOptInput:
    def __init__(self, model, feature_extractor):
        self.model = model 
        self.feature_extractor = feature_extractor
        self.decoder_input_ids = torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id
        
    def back_prop_step(self):   
        n = torch.matmul(self.beta, self.hook.output_f)
        loss = -n.mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_opt_input(self, layer, beta, spect_width, iterations=2):
        self.hook = Hook(self.model, layer, backward=True)
        self.beta = torch.tensor(beta.T)

        # self.spect = self.
        torch.manual_seed(0)
        self.spect = torch.rand(size=(1, spect_width, 80))

        self.spect.requires_grad = True

        self.optimizer = torch.optim.Adam([self.spect], lr=0.01)

        for param in self.model.parameters():
            assert param.requires_grad == False, "Model parameters not frozen"

        self.loss_list = []

        self.optimizer.zero_grad()
        for i in range(iterations):
            net_out = self.model(self.spect, decoder_input_ids=self.decoder_input_ids)
            l = self.back_prop_step()
            self.loss_list.append(l)