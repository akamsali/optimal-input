import torch
from optimal_input.hook import Hook
class GetOptInput:
    def __init__(self, model):
        self.model = model 

    def back_prop_step(self):   
        n = torch.matmul(self.beta, self.hook.output_f)
        loss = -n.mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_opt_input(self, aud, layer, beta, iterations=2):
        self.hook = Hook(self.model, layer, backward=True)
        self.beta = torch.tensor(beta.T)

        self.aud = aud

        self.aud.requires_grad = True

        self.optimizer = torch.optim.Adam([self.aud], lr=10)

        for param in self.model.parameters():
            assert param.requires_grad == False, "Model parameters not frozen"

        self.loss_list = []

        self.optimizer.zero_grad()
        for i in range(iterations):
            net_out = self.model(self.aud)
            l = self.back_prop_step()
            self.loss_list.append(l)