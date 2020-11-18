import torch
import torch.nn.functional as FF
from MetricEvaluator import evaluate_metrics as EVMET
from sklearn import metrics as SKM
import matplotlib.pyplot as plt

class SaliencyMapExtractor:
    def __init__(self,name,method):
        self.name,self.method=name,method

    def __call__(self,arch,img,target,out=None):
        return self.method(self.name,arch=arch,img=img,out=out,target=target)

class Coherency(EVMET.MetricOnAllDataset):
    def __init__(self,
                 name,
                 arch,
                 saliency_map_extractor:SaliencyMapExtractor,
                 result=0.
                 ):
        super().__init__(name,result)
        self.arch,self.saliency_map_extractor,self.res_list=arch,saliency_map_extractor,[]
        if torch.cuda.is_available():
            self.arch.set_arch(self.arch.get_arch().cuda())

    def update(self,*args):
        inp,_,target,salmap=args

        salmap2=self.saliency_map_extractor(self.arch,inp*salmap,target)
        self.res_list.append(torch.exp(-torch.norm(salmap-salmap2)))

    def final_step(self,**kwargs):
        print(self.res_list)
        self.result = SKM.auc(torch.arange(0, 1, 1/len(self.res_list)).numpy(),
                                    torch.tensor(self.res_list).numpy())