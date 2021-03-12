# EXAMPLE OF FILE CONTAINIG METRICS ON SINGLE EXAMPLE

import torch
import torch.nn.functional as FF
from MetricEvaluator import evaluate_metrics as EVMET
from sklearn import metrics as SKM
from images_utils.images_utils import *


class Deletion(EVMET.MetricOnSingleExample):
    def __init__(self,name,arch,result=0.,st=0.01):
        super().__init__(name,result,[])
        self.arch,self.st=arch,st
        if torch.cuda.is_available():
            self.arch.set_arch(self.arch.get_arch().cuda())

    def get_res_list(self):
        return self.res_list

    def update(self,*args):
        img, Y_i_c, class_idx, em,_=args
        exp_map = em.clone()
        inp = img.clone()
        self.res_list.append(Y_i_c)

        for _ in range(int(1/self.st)):
            inp, exp_map = remove_more_important_px(inp, exp_map, step=self.st)
            if torch.cuda.is_available():
                inp = inp.cuda()
                self.arch.arch=self.arch.arch.cuda()

            with torch.no_grad():
                out = FF.softmax(self.arch.get_arch()(inp), dim=1)
            Y_i_c = out[:, class_idx][0].item()
            self.res_list.append(Y_i_c)
    def final_step(self,**kwargs):
        self.res_list=self.res_list[:int(1/self.st)]
        self.result=round(SKM.auc(torch.arange(0, 1, self.st).numpy(),
                      torch.tensor(self.res_list).numpy()), 3)

class Insertion(EVMET.MetricOnSingleExample):
    def __init__(self,name,arch,result=0.,st=0.01):
        super().__init__(name,result,[])
        self.arch,self.st=arch,st
        if torch.cuda.is_available():
            self.arch.set_arch(self.arch.get_arch().cuda())
            
    def get_res_list(self):
        return self.res_list

    def update(self, *args):
        img, Y_i_c, class_idx, em,_=args
        exp_map = em.clone()
        inp = self.arch.apply_transform(torch.zeros(img.squeeze(0).shape))
        self.res_list.append(Y_i_c)
        # print(img.mean(),inp.norm())

        for _ in range(int(1/self.st)):
            # print(exp_map.norm(),(inp-img).norm())
            inp, exp_map = insert_more_important_px(img, inp, exp_map, step=self.st)
            if torch.cuda.is_available():
                inp=inp.cuda()
                self.arch.arch = self.arch.arch.cuda()
            with torch.no_grad():
                out = FF.softmax(self.arch.get_arch()(inp), dim=1)
            Y_i_c = out[:, class_idx][0].item()
            self.res_list.append(Y_i_c)

    def final_step(self,**kwargs):
        self.res_list=self.res_list[-int(1/self.st):]
        self.result=SKM.auc(torch.arange(0, 1, self.st).numpy(),
                      torch.tensor(self.res_list).numpy())



def remove_more_important_px(img,exp_map,step=0.01):
    exp_map=exp_map.squeeze(0).squeeze(0)
    max_iter=int((img.shape[2]*img.shape[3]*step))
    #print(exp_map)
    #print(max_iter)
    #print(exp_map.view(1,-1).topk(max_iter)[1])

     #= exp_map.view(1, -1)
    argmax=exp_map.view(1, -1).topk(max_iter)[1]
    #print(argmax.shape)
    zero = trans(torch.zeros(3).unsqueeze(1).unsqueeze(1)).cuda()
    zero=zero.repeat(1,1,1,max_iter)
    #im1=img.view(-1,1,3).clone()
    #print(zero)
    #print(argmax)

    i = (argmax // exp_map.shape[1]).type(torch.LongTensor)
    j = (argmax % exp_map.shape[1]).type(torch.LongTensor)
    img[:, :, i, j]=zero.view(img[:, :, i, j].shape)
    #print(i,j)
    exp_map[i,j]=0

    #img=im1.view(img.shape).clone()
    #for idx in argmax[0]:
    #    print(denormalize(img.view(-1,1,3)[idx]))
    #print(denormalize(zero), denormalize(img).min(), denormalize(img).max())
    #plt.figure()
    #plt.imshow(denormalize(img).squeeze(0).cpu().detach().permute(1,2,0).numpy())
    #plt.show()
    #plt.savefig(f'out/fig{time.time()}.png')
    return img,exp_map

def insert_more_important_px(img,inp,exp_map,step=0.01):
    exp_map = exp_map.squeeze(0).squeeze(0)
    max_iter = int((inp.shape[2] * inp.shape[3] * step))
    #print(max_iter)
    im=inp.clone()
    if torch.cuda.is_available():
        im=im.cuda()
    argmax = exp_map.view(1, -1).topk(max_iter)[1]
    #img1,im1 = img.view(-1, 1, 3),im.view(-1, 1, 3)
    i = argmax // exp_map.shape[1]
    j = argmax % exp_map.shape[1]
    im[:, :, i, j] = img[:, :, i, j]
    exp_map[i, j] = 0

    #plt.figure()
    #plt.imshow(denormalize(im1.view(im.shape)).cpu().squeeze(0).detach().permute(1, 2, 0).numpy())
    #plt.savefig(f'out/fig{time.time()}111.png')
    #im=im1.view(im.shape)

    #plt.figure()
    #plt.imshow(denormalize(im).squeeze(0).cpu().detach().permute(1,2,0).numpy())
    #plt.savefig(f'out/fig{time.time()}222.png')
    return im,exp_map

