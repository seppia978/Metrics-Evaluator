import math
import torch
import torch.nn.functional as F

class BASECAM:
    def __init__(self,model):
        if not hasattr(model.arch, model.layer):
            raise ValueError(f"Unable to find submodule {model.layer} in the model")
        self.model=model

        self.model.arch.eval()
        if torch.cuda.is_available():
            self.model.arch.cuda()
        self.gradients = dict()
        self.activations = dict()



        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)


    def backward_hook(self,module, grad_input, grad_output):
        if torch.cuda.is_available():
            self.gradients['value'] = grad_output[0].cuda()
        else:
            self.gradients['value'] = grad_output[0]
        return None

    def forward_hook(self,module, input, output):
        if torch.cuda.is_available():
            self.activations['value'] = output.cuda()
        else:
            self.activations['value'] = output
        return None