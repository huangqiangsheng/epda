import torch
class Predictor(object):
    def __init__(self,model):
        self.model=model
        self.model.eval()
        
    def predict(self,input):
        with torch.no_grad():
            output=self.model(input)
        return output
