import torch
import scipy.stats as STS
from MetricEvaluator import evaluate_metrics as EVMET
import torch.nn.functional as FF

class Complexity(EVMET.MetricOnAllDataset):
    def __init__(self,name,arch,result=0.,hbins=100):
        super().__init__(name,result)
        self.arch,self.tot,self.hbins=arch,0.,hbins
        if torch.cuda.is_available():
            self.arch.set_arch(self.arch.get_arch().cuda())

    def update(self,*args):
        saliency_map=args[-2].cpu()
        #saliency_map=torch.zeros(saliency_map.shape)
        #saliency_map+=1

        # count non zero vals
        '''
        self.result+=(saliency_map>0).sum()
        self.tot+=(saliency_map.shape[-1]*saliency_map.shape[-2])
        #'''

        # l1 norm
        #'''
        self.result+=abs(saliency_map).sum()
        print(self.result)
        self.tot+=(saliency_map.shape[-1]*saliency_map.shape[-2])
        #'''


    def final_step(self,**kwargs):
        self.result=float(self.result * 100/ self.tot)

    def clear(self):
        super().clear()
        self.tot=0.

    def print(self):
        print(f'Complexity: {self.result}%')
