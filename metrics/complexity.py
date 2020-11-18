import torch
from MetricEvaluator import evaluate_metrics as EVMET
import torch.nn.functional as FF

class Complexity(EVMET.MetricOnAllDataset):
    def __init__(self,name,arch,result=0.):
        super().__init__(name,result)
        self.arch,self.tot_px=arch,0.
        if torch.cuda.is_available():
            self.arch.set_arch(self.arch.get_arch().cuda())

    def update(self,*args):
        saliency_map=args[-1]

        self.result+=(saliency_map>0).sum()
        self.tot_px+=(saliency_map.shape[-1]*saliency_map.shape[-2])


    def final_step(self,**kwargs):

        self.result=float(self.result * 100 / self.tot_px)

    def clear(self):
        super().clear()
        self.tot_px=0.