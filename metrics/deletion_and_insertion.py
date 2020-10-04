# EXAMPLE OF FILE CONTAINIG METRICS ON SINGLE EXAMPLE

import torch
import torch.nn.functional as FF
from MetricEvaluator import evaluate_metrics as EVMET
from sklearn import metrics as SKM
from images_utils.images_utils import *


class Deletion(EVMET.MetricOnSingleExample):
    def __init__(self,name,result,arch,st=0.01):
        super().__init__(name,result,[])
        self.arch,self.st=arch,st
        if torch.cuda.is_available():
            self.arch.set_arch(self.arch.get_arch().cuda())

    def get_res_list(self):
        return self.res_list

    def update(self,*args):
        img, out, em=args
        exp_map = em.clone()
        inp = img.clone()
        Y_i_c = out.max(1)[0].item()
        self.res_list.append(Y_i_c)
        class_idx = out.max(1)[-1].item()
        while not exp_map.norm() == 0:
            inp, exp_map = remove_more_important_px(inp, exp_map, step=self.st)
            if torch.cuda.is_available():
                inp = inp.cuda()
            out = FF.softmax(self.arch.get_arch()(inp), dim=1)
            Y_i_c = out[:, class_idx][0].item()
            self.res_list.append(Y_i_c)
    def final_step(self,**kwargs):
        self.res_list=self.res_list[:int(1/self.st)]
        self.result=round(SKM.auc(torch.arange(0, 1, self.st).numpy(),
                      torch.tensor(self.res_list).numpy()), 3)

class Insertion(EVMET.MetricOnSingleExample):
    def __init__(self,name,result,arch,st=0.01):
        super().__init__(name,result,[])
        self.arch,self.st=arch,st
        if torch.cuda.is_available():
            self.arch.set_arch(self.arch.get_arch().cuda())
            
    def get_res_list(self):
        return self.res_list

    def update(self, *args):
        img, out, em = args
        exp_map = em.clone()
        # print(exp_map.min())
        inp = self.arch.apply_transform(torch.zeros(img.squeeze(0).shape))

        class_idx = out.max(1)[-1].item()
        # print(img.mean(),inp.norm())
        while not exp_map.norm() == 0:
            # print(exp_map.norm(),(inp-img).norm())
            inp, exp_map = insert_more_important_px(img, inp, exp_map, step=self.st)
            if torch.cuda.is_available():
                inp=inp.cuda()
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
    if not img.norm() == 0:
        #print(max_iter)
        #print(exp_map.view(1,-1).topk(max_iter)[1])

        argmax = exp_map.view(1, -1).topk(max_iter)[1]
        zero = trans(torch.zeros(3).unsqueeze(1).unsqueeze(1)).cuda()
        zero=zero.repeat(1,1,1,max_iter)
        #im1=img.view(-1,1,3).clone()
        #print(zero)
        #print(argmax)

        i = argmax // exp_map.shape[1]
        j = argmax % exp_map.shape[1]
        img[:, :, i, j]=zero.view(img[:, :, i, j].shape)
        for idx in argmax[0]:
            exp_map.view(-1,1)[idx]=0
        #img=im1.view(img.shape).clone()
        #for idx in argmax[0]:
        #    print(denormalize(img.view(-1,1,3)[idx]))
        '''
        
        print(img.view(1,-1,3).shape)
        for _ in range(max_iter):
            #print(max_iter)
            argmax=exp_map.argmax()
            #print(argmax in exp_map.view(1,-1).topk(max_iter)[1])
            i = argmax // exp_map.shape[1]
            j = argmax % exp_map.shape[1]
            zero=trans(torch.zeros(3).unsqueeze(1).unsqueeze(1)).cuda()
            img[:,:,i,j]=zero.view(1,3)
            exp_map[i,j]=0
            #print(i,j,zero)

            #if (iii+1) % 1000 == 0:
            #    print(iii,max_iter)
        '''
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
    if not exp_map.norm()==0:
        argmax = exp_map.view(1, -1).topk(max_iter)[1]
        #img1,im1 = img.view(-1, 1, 3),im.view(-1, 1, 3)
        i = argmax // exp_map.shape[1]
        j = argmax % exp_map.shape[1]
        im[:, :, i, j] = img[:, :, i, j]
        for idx in argmax[0]:
            #im1[idx]=img1[idx]
            exp_map.view(-1,1)[idx]=0

        #plt.figure()
        #plt.imshow(denormalize(im1.view(im.shape)).cpu().squeeze(0).detach().permute(1, 2, 0).numpy())
        #plt.savefig(f'out/fig{time.time()}111.png')
        #im=im1.view(im.shape)
        '''
        for iii in range(max_iter):
            argmax = exp_map.argmax()
            i = argmax // exp_map.shape[1]
            j = argmax % exp_map.shape[1]
            im[:,:,i,j]=img[:,:,i,j]
            exp_map[i,j]=0
            #print(exp_map.norm(),(im-img).norm())
            #print(f'{(i,j)}')
            #if (iii+1) % 1000 == 0:
            #    print(iii,max_iter)
            #    plt.figure()
            #    plt.imshow(im.cpu().squeeze(0).detach().permute(1, 2, 0).numpy())
            #    plt.savefig(f'out/{exp_map.norm()}.png')
        '''
        #plt.figure()
        #plt.imshow(denormalize(im).squeeze(0).cpu().detach().permute(1,2,0).numpy())
        #plt.savefig(f'out/fig{time.time()}222.png')
        return im,exp_map
