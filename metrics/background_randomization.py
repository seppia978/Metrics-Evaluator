import torch
from MetricEvaluator import evaluate_metrics as EVMET
import torch.nn.functional as FF
from sklearn import metrics as SKM

class BackgroundRandomization(EVMET.MetricOnSingleExample):
    def __init__(self,name,arch,result=0,max_iter=100):
        super().__init__(name, result, [])
        self.arch, self.st = arch, max_iter
        if torch.cuda.is_available():
            self.arch.set_arch(self.arch.get_arch().cuda())


    def update(self,*args):
        img, Y_i_c, class_idx, em = args
        for _ in range(self.max_iter):
            idx = (img.view(*img.shape[:-2], 1, -1).squeeze(0).squeeze(0).squeeze(0)==0).nonzero()


    def final_step(self,**kwargs):
        self.res_list = self.res_list[:int(1 / self.st)]
        self.result = round(SKM.auc(torch.arange(0, 1, self.st).numpy(),
                                    torch.tensor(self.res_list).numpy()), 3)