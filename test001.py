import torch
import torch.nn.functional as FF
import torchvision.transforms.functional as F
import torchvision.models as models
from sklearn import metrics as SKM
import os
import time
import get_explanation_map as expmap
from utils import *
import metrics.average_drop_and_increase_of_confidence as ADIC
import metrics.deletion_and_insertion as DAI
from utils_functions import plot

def evaluate_metrics(p,img_dict, model='resnet', times=1,metrics=[],outpath_root='.',labs_vs_gt=None):
    GT=labs_vs_gt[0]
    labs=labs_vs_gt[1]
    num_imgs = len(img_dict)
    model = model
    if model == 'resnet':
        arch = models.resnet18(pretrained=True).eval()
    elif model == 'vgg':
        arch = models.vgg16(pretrained=True).eval()
    elif model == 'alexnet':
        arch = models.alexnet(pretrained=True).eval()

    if torch.cuda.is_available():
        arch = arch.cuda()

    start = time.time()
    now = start
    times = times
    average_drop, increase_in_confidence = 0.0, 0.0
    deletion, insertion=[],[]
    if metrics is not []:
        for _ in range(times):
            for i, (k,img) in enumerate(img_dict.items()):
                outpath=outpath_root+f'{k}_{img}/'
                inp_0=load_image(p + '/' + img)
                os.mkdir(outpath)
                inp_0.save(f'{outpath}{img}')
                inp = apply_transforms(inp_0)
                if torch.cuda.is_available():
                    inp = inp.cuda()
                #print(f'Before test.run: {round(time.time() - now, 0)}s')
                now = time.time()
                out, scorecam_map = expmap.get_explanation_map(arch=model, img=p + '/' + img)
                F.to_pil_image(scorecam_map.squeeze(0)).save(f'{outpath}/exp_map.png')
                #print(f'After test.run: {round(time.time() - now, 0)}s')
                now = time.time()
                if torch.cuda.is_available():
                    scorecam_map = scorecam_map.cuda()
                #print(f'Before arch: {round(time.time() - now, 0)}s')
                now = time.time()
                out_sal = FF.softmax(arch(inp * scorecam_map), dim=1)
                #print(f'After arch: {round(time.time() - now, 0)}s')
                now = time.time()
                # print(type(out_sal),out_sal.shape)
                Y_i_c = out.max(1)[0].item()
                class_idx = out.max(1)[-1].item()
                class_name=labs[class_idx]
                gt_name=GT[str(img[-13:-5])][0].split()[1]
                O_i_c = out_sal[:, class_idx][0].item()
                # print(f'#-------------------------------------------------------------------#')
                # print(f'{Y_i_c},{out.max(1)[-1].item()},\n{O_i_c},{out_sal.max(1)[-1].item()}\n')
                # print(f'{Y_i_c},{O_i_c},{max(0.0,Y_i_c-O_i_c)},{max(0,Y_i_c-O_i_c)/Y_i_c}')
                # print('#-------------------------------------------------------------------#')
                if 'average_drop' in metrics and 'increase_in_confidence' in metrics:
                    average_drop,increase_in_confidence=ADIC.average_drop_and_increase_of_confidence(average_drop,increase_in_confidence,Y_i_c,O_i_c)
                if 'deletion' in metrics and 'insertion' in metrics:
                    precision=100
                    deletion,insertion=DAI.deletion_and_insertion(deletion,insertion,inp,scorecam_map,arch,step=1/precision)
                    #print(deletion, insertion)

                    #deletion_score = round(torch.tensor(deletion).sum().item() / precision,3)
                    #insertion_score = round(torch.tensor(insertion).sum().item() / precision,3)
                    deletion_score = round(SKM.auc(torch.arange(0, 1, 1 / precision).numpy(),
                                             torch.tensor(deletion).numpy()),3)
                    insertion_score = round(SKM.auc(torch.arange(0, 1, 1 / precision).numpy(),
                                              torch.tensor(insertion).numpy()),3)
                    plot(torch.arange(0,1,1/precision),[deletion,insertion],label=[f'deletion={deletion_score}',f'insertion={insertion_score}'],path=f'{outpath}plot_{k}.png',title=f'label={class_name}, GT={gt_name}')

                    print(f'The final deletion is: {deletion_score}')
                    print(f'The final insertion is: {insertion_score}')
                    deletion,insertion=[],[]
                print(f'After one img: {int(time.time() - now)}s')
                now = time.time()

            print(f'In {num_imgs} images')
            if 'average_drop' in metrics and 'increase_in_confidence' in metrics:
                average_drop *= 100 / num_imgs
                increase_in_confidence *= 100 / num_imgs
                print(f'The final AVG drop is: {round(average_drop, 2)}%')
                print(f'The final Increase in Confidence is: {round(increase_in_confidence, 2)}%')

        print(f'Execution time: {int(time.time() - start)}s')