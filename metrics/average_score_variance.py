import torch
from MetricEvaluator import evaluate_metrics as EVMET
import torch.nn.functional as FF

def one(X,Y):
    if X>Y:
        return 1
    elif X==Y:
        return 0
    else:
        return -1

class AverageScoreVariance(EVMET.MetricOnAllDataset):
    def __init__(self,name,arch,result=0.):
        super().__init__(name,result)
        self.arch,self.num_imgs=arch,0.
        if torch.cuda.is_available():
            self.arch.set_arch(self.arch.get_arch().cuda())

    def update(self,*args):
        inp,Y_i_c,class_idx,saliency_map,_=args
        if torch.cuda.is_available():
            inp = inp.cuda()
            self.arch.arch = self.arch.arch.cuda()
        with torch.no_grad():
            out_sal = FF.softmax(self.arch.get_arch()(inp * saliency_map), dim=1)
        O_i_c = out_sal[:, class_idx][0].item()

        #print(self.result,one(Y_i_c, O_i_c))
        self.result += one(O_i_c, Y_i_c)
        self.num_imgs+=1

    def final_step(self,**kwargs):
        self.result=self.result * 100 / self.num_imgs
    def clear(self):
        super().clear()
        self.num_imgs=0.

    def print(self):
        print(f'Average Score Variance: {self.result}%')