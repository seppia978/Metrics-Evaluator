import torch
import torch.nn.functional as FF
import torchvision.transforms.functional as F
from MetricEvaluator import evaluate_metrics as EVMET
from sklearn import metrics as SKM
from scipy import stats as STS
from images_utils import images_utils as IMUT
import matplotlib.pyplot as plt

class SaliencyMapExtractor:
    def __init__(self,name=None,method=None):
        self.name,self.method=name,method

    def __call__(self,arch,img,target,out=None):
        return self.method(arch=arch,img=img,out=out,target=target,extractor=self.name)

class Coherency(EVMET.MetricOnAllDataset):
    def __init__(self,
                 name,
                 arch,
                 saliency_map_extractor:SaliencyMapExtractor=SaliencyMapExtractor(),
                 result=0.,
                 l=100,
                 outpath=None
                 ):
        super().__init__(name,result)
        self.arch,self.saliency_map_extractor,self.res_list,self.l=arch,saliency_map_extractor,[],l
        self.outpath=outpath
        if torch.cuda.is_available():
            self.arch.set_arch(self.arch.get_arch().cuda())

    def update_preliminary_checks(self,*args):
        if self.saliency_map_extractor.method is None or self.saliency_map_extractor.name is None:
            raise ValueError(f'saliency_map_extractor.name and saliency_map_extractor.method must not be None. Found {self.saliency_map_extractor.name} and {self.saliency_map_extractor.method}')
    def final_step_preliminary_checks(self,*args):
        assert len(self.res_list)>1, f'AUC must be computed on more than one point. Found {len(self.res_list)}'

    def update(self,*args):
        self.update_preliminary_checks()
        inp,Y,target,A,img=args

        B=self.saliency_map_extractor(arch=self.arch,img=inp*A,target=target)
        A,B=A.detach(),B.detach()
        if self.outpath is not None:
            IM=IMUT.IMG_list()
            F.to_pil_image(B.squeeze(0).cpu()).save(f'{self.outpath}{IM.get_num_img(img)-1}_{img}/B.png')
            F.to_pil_image(A.squeeze(0).cpu()).save(f'{self.outpath}{IM.get_num_img(img)-1}_{img}/A.png')

        # Linear coherency
        '''
        x=torch.norm(A - B)
        m=-1/inp.shape[-1]
        y=m*x+1.
        '''
        # Pearson correlation coefficient
        #'''
        Asq,Bsq=A.view(1,-1).squeeze(0).cpu(),B.view(1,-1).squeeze(0).cpu()
        y,_=STS.pearsonr(Asq,Bsq)
        y=abs(y)
        #'''

        # Cross correlation
        '''
        y=
        '''
        if torch.tensor(y).isnan():
            y=sum(self.res_list)/len(self.res_list)

        self.res_list.append(y)

    def final_step(self,**kwargs):
        self.final_step_preliminary_checks()

        self.result=sum(self.res_list) * 100 / len(self.res_list)
        #self.result = SKM.auc(torch.arange(0, 1, 1/len(self.res_list)).numpy(),
        #                            torch.tensor(self.res_list).numpy())

    def clear(self):
        super().clear()
        self.res_list=[]

    def print(self):
        print(f'Coherency: {self.result}%')