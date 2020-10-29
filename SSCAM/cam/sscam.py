import torch
import numpy as np
import math
import torch.nn.functional as F
from SSCAM.cam.basecam import *
from torch.autograd import Variable
import time

class SSCAM1(BaseCAM):

    """
        SSCAM1, inherit from BaseCAM
    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, param_n=35, mean=0, sigma=2, retain_graph=False):
        b, c, h, w = input.size()
        
        # prediction on raw input
        logit = self.model_arch(input)
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()
        
        logit = F.softmax(logit,dim=1)

        if torch.cuda.is_available():
          predicted_class= predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b1, k, u, v = activations.size()
        
        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        #HYPERPARAMETERS (can be modified for better/faster explanations)
        mean = 0
        param_n = 35
        param_sigma_multiplier = 2
        

        with torch.no_grad():
          for i in range(k):

              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
 
              if saliency_map.max() == saliency_map.min():
                continue

              x = saliency_map               

              if (torch.max(x) - torch.min(x)).item() == 0:
                continue
              else:
                sigma = param_sigma_multiplier / (torch.max(x) - torch.min(x)).item()
              
              score_list = []
              noisy_list = []
              
              # Adding noise to the upsampled activation map `x`
              for _ in range(param_n):
                #noise=torch.normal(mean,sigma**2,x.size()).cuda()
                noise = Variable(x.data.new(x.size()).normal_(mean, sigma**2))
                
                noisy_img = x + noise

                noisy_list.append(noisy_img)

                print((noisy_img * input).shape)
               
                output = self.model_arch(noisy_img * input)
                output = F.softmax(output,dim=1)
                score = output[0][predicted_class]
                score_list.append(score)
              
              # Averaging the scores to introduce smoothing
              score = sum(score_list) / len(score_list)
              score_saliency_map +=  score * saliency_map
                
        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class SSCAM2(BaseCAM):

    """
        SSCAM2, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, param_n=35, mean=0, sigma=2, retain_graph=False):
        b, c, h, w = input.size()
        
        # prediction on raw input
        logit = self.model_arch(input)
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()
        
        logit = F.softmax(logit,dim=1)

        if torch.cuda.is_available():
            predicted_class= predicted_class.cuda()
            score = score.cuda()
            logit = logit.cuda()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b1, k, u, v = activations.size()
        
        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        #HYPERPARAMETERS (can be modified for better/faster explanations)
        mean = 0
        param_n = 35
        param_sigma_multiplier = 2

        now = time.time()

        size = list(input.shape)
        size[0] = param_n
        noisy_list = torch.zeros(size).cuda()
        for i in range(param_n):
            noisy_list[i, :] = Variable(input.data.new(input.size()).normal_(mean, sigma ** 2))

        bs=32
        with torch.no_grad():
            for i in range(math.ceil(k/bs)):
                selection_slice = slice(i * bs, min((i + 1) * bs, k))
                print('0', time.time() - now)
                now=time.time()
                start=now

                # upsampling
                saliency_map = activations.permute(1,0,2,3)[selection_slice, :, :, :]
                print('1', time.time() - now)
                now = time.time()


                print(activations.shape,saliency_map.shape,selection_slice)
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
                now-=time.time()
                print('2', now)
                now = time.time()

                m, M = saliency_map.min(), saliency_map.max()
                if m == M:
                    print('FUCKKKK')
                    continue

                # Normalization
                norm_saliency_map = (saliency_map - m)/(M - m)

                now = time.time()-now
                print('3', now)
                now = time.time()

                print(input.shape,norm_saliency_map.shape)
                x = input * norm_saliency_map
                now -= time.time()
                print('4', now)
                now = time.time()


                if (torch.max(x) - torch.min(x)).item() == 0:
                    continue
                else:
                    sigma = param_sigma_multiplier / (torch.max(x) - torch.min(x)).item()
                now -= time.time()
                print('5', now)
                now = time.time()

                size=list(x.shape)
                size[0]=param_n
                score_list = torch.zeros(param_n).cuda()

                # Adding noise to the normalized input mask `x`
                noisy_list,x=noisy_list.unsqueeze(1),x.unsqueeze(0)
                print(noisy_list.shape,x.shape)
                noisy_input=noisy_list+x
                noisy_input=noisy_input.cuda()

                output = F.softmax(self.model_arch(noisy_input),dim=1)

                score_list = output[:,predicted_class]

                now -= time.time()
                print('6', now)
                now = time.time()
                #score_list[i]=score

                #print(noisy_list,noisy_list.shape)
                '''
                for i in range(param_n):
                    noise = Variable(x.data.new(x.size()).normal_(mean, sigma**2))

                    noisy_img = x + noise 

                    noisy_list.append(noisy_img)

                    noisy_img = noisy_img.cuda()

                    output = self.model_arch(noisy_img)

                    output = F.softmax(output,dim=1)

                    score = output[0][predicted_class]

                    score_list.append(score)
                '''
                # Averaging the scores to introduce smoothing
                #score = sum(score_list) / len(score_list)
                score=score_list.max()
                now -= time.time()
                print('6.5', now)
                now = time.time()
                score_saliency_map +=  score * saliency_map
                print('7', time.time() - now, time.time()-start)
                now = time.time()
                
            score_saliency_map = F.relu(score_saliency_map)
            score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
