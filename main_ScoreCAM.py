from images_utils import images_utils as IMUT
from MetricEvaluator import evaluate_metrics as EVMET
import metrics.average_drop_and_increase_of_confidence as ADIC
import metrics.deletion_and_insertion as DAI
import torchvision.models as models
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from ScoreCAM import test
import sys
import time
import os
from xdeep.xlocal.gradient.explainers import *

import matplotlib.pyplot as plt
import torch.nn.functional as FF
torch.set_num_threads(1)
CAMS={'GradCAM':GradCAM,'GradCAM++':GradCAMpp}



def run(*params,arch, img, out,target):
    #st=time.time()
    #now=st
    CAM=params[0]
    input = img
    model = arch

    #print('-----first in run', time.time()-now,'\n')
    md = {'arch': model.get_arch(), 'layer_name': model.layer}
    cam = CAM(md)
    #print('-----after creating object in run', time.time() - now,'\n')
    salmap = cam(input, class_idx=target)
    #print('-----after generating salmap in run', time.time() - now,'\n')
    ##plt.figure()
    #plt.imshow(salmap.squeeze(0).squeeze(0))
    #plt.savefig(f'result{str(cam)}.png')

    return salmap



def get_name_images(s):
    return str(s)[-13:-5]

def get_num_img(s):
    return int(s[-13:-5])
def get_n_imgs(l,pattern):
    ret=[]
    for i in range(len(l)):
        s = ''.join(map(str, ['0' for _ in range(13 - 5 - len(str(l[i])))]))+str(l[i])
        p=pattern.replace('********', s)
        ret.append(p)
    return {k: v for k, v in zip(range(l[0],l[-1]+1), ret)}

#------ MAIN -------#
i=0
chunk_id=-1
chunk_dim=0
params=[]
for arg in sys.argv[1:]:
    if not(i%2==0):
        params.append(arg)
    i+=1

chunk_id,chunk_dim=[int(x) for x in params]
num_imgs = chunk_dim

displacement=0

print(num_imgs)
p = ''
root = './'  # '/tirocinio_tesi/Score-CAM000/Score-CAM'
outpath_root = root + 'out/filter/'
data_root = root + 'ILSVRC2012_devkit_t12/data/'

with open(data_root + 'IMAGENET_path.txt', 'r') as f:
    p = f.read().strip()
p+='/'
labs_w = []
with open(data_root + 'labels.txt', 'r') as f:
    labs_w = [x[9:] for x in f.read().strip().split('\n')]

labs = []
with open(data_root + 'imagenet_classes.txt', 'r') as f:
    labs = [x for x in f.read().strip().split('\n')]

labs1 = {}
with open(data_root + 'labels.txt', 'r') as f:
    labs1 = {str(get_name_images(x.split()[0])): x.split()[2] for x in f.read().split('\n')}
    # labs1=[x[2] for x in labs1.split()]

GT = {}
with open(data_root + 'imagenet_val_gts.txt', 'r') as f:
    GT = {get_name_images(x.split()[0]): [get_name_images(x.split()[0]) + ' ' + x.split()[2]] for x in f.read().strip().split('\n')}


#img_dict=IMUT.IMG_list(path=p,GT=GT,labs=labs).generate_random(num_imgs)
base,window=chunk_id*chunk_dim+displacement,chunk_dim
pattern='ILSVRC2012_val_********.JPEG'
#img_list=get_n_imgs(range(base+1,base+window+1),pattern)

with open('filter.txt','r') as f:
    txt=f.read()
img_list=[p.split()[0] for p in txt.strip().split('\n')[base:base+window]]
img_list={get_num_img(p.split()[0]):p.split()[0] for p in img_list}

#arch=EVMET.Architecture(models.resnet18(pretrained=True).eval(),'resnet18','layer4')
arch=EVMET.Architecture(models.vgg16(pretrained=True).eval(),'vgg16','features_29')

avg_drop=ADIC.AverageDrop('average_drop',arch)
inc_conf=ADIC.IncreaseInConfidence('increase_in_confidence',arch)
deletion=DAI.Deletion('deletion',arch)
insertion=DAI.Insertion('insertion',arch)

img_dict = IMUT.IMG_list(path=p,outpath_root='out/filter/', GT=GT, labs=labs).select_imgs(img_list)

em = EVMET.MetricsEvaluator(img_dict, saliency_map_extractor=run, model=arch,
                                metrics=[avg_drop, inc_conf, deletion, insertion])
start = time.time()
now = start
path0=img_dict.get_outpath_root()
try:
    os.mkdir(f'{path0}')
except:
    pass

for c in CAMS.keys():
    try:
        os.mkdir(f'{path0}{str(c)}/')
    except:
        pass
    img_dict.set_outpath_root(f'{path0}{str(c)}/')
    print(img_dict.get_outpath_root())
    print(img_dict.get_img_dict())

    M_res,m_res=em(CAMS[c])

    print(f'Execution time: {int(time.time() - start)}s')
    print(f'In {num_imgs} images')
    for M in M_res:
        print(f'The final {M.get_name()} is: {round(M.get_result(), 2)}%')

    f=open(f'{outpath_root}output.txt','a')
    output={(f'Chunk{chunk_id}',f'{c}'):([m.get_result() for m in M_res],[torch.tensor(m.get_result()).mean().item() for m in m_res])}
    f.write(str(output))
    f.write('\n')
    for m in em.metrics:
        m.clear()

