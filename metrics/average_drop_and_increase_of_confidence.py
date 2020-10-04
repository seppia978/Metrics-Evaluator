# EXAMPLE OF FILE CONTAINIG METRICS ON ALL DATASET

import torch
from MetricEvaluator import evaluate_metrics as EVMET
import torch.nn.functional as FF

class AverageDrop(EVMET.MetricOnAllDataset):
    def __init__(self,name,arch,result=0):
        super().__init__(name,result)
        self.arch=arch
        if torch.cuda.is_available():
            self.arch.set_arch(self.arch.get_arch().cuda())

    def update(self,inp,out,saliency_map):
        Y_i_c,class_idx=out.max(1)[0].item(),out.max(1)[-1].item()
        if torch.cuda.is_available():
            inp = inp.cuda()
        out_sal = FF.softmax(self.arch.get_arch()(inp * saliency_map), dim=1)
        O_i_c = out_sal[:, class_idx][0].item()
        print(self.arch.get_name())
        self.result += (max(0.0, Y_i_c - O_i_c) / Y_i_c)

    def final_step(self,num_imgs):
        self.result=self.result * 100 / num_imgs

class IncreaseInConfidence(EVMET.MetricOnAllDataset):
    def __init__(self,name,arch,result=0):
        super().__init__(name,result)
        self.arch=arch
        if torch.cuda.is_available():
            self.arch.set_arch(self.arch.get_arch().cuda())

    def one(M, m):
       return int(M > m)

    def update(self,inp,out,saliency_map):
        Y_i_c,class_idx=out.max(1)[0].item(),out.max(1)[-1].item()
        if torch.cuda.is_available():
            inp = inp.cuda()
        out_sal = FF.softmax(self.arch.get_arch()(inp * saliency_map), dim=1)
        O_i_c = out_sal[:, class_idx][0].item()
        self.result += self.one(O_i_c, Y_i_c)

    def final_step(self,num_imgs):
        self.result=self.result * 100 / num_imgs

