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
        '''

        # entropy
        S=saliency_map.clone().view(1,-1).squeeze(0)
        S/=S.sum()
        self.result+=STS.entropy(S)
        self.tot+=1

    def final_step(self,**kwargs):

        self.result=float(self.result / self.tot)

    def clear(self):
        super().clear()
        self.tot=0.