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
from torchcam.cams import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
import matplotlib.pyplot as plt
import torch.nn.functional as FF

VGG_CONFIG = {_vgg: dict(input_layer='features', conv_layer='features')
              for _vgg in models.vgg.__dict__.keys()}

RESNET_CONFIG = {_resnet: dict(input_layer='conv1', conv_layer='layer4', fc_layer='fc')
                 for _resnet in models.resnet.__dict__.keys()}

DENSENET_CONFIG = {_densenet: dict(input_layer='features', conv_layer='features', fc_layer='classifier')
                   for _densenet in models.densenet.__dict__.keys()}

MODEL_CONFIG = {
    **VGG_CONFIG, **RESNET_CONFIG, **DENSENET_CONFIG,
    'mobilenet_v2': dict(input_layer='features', conv_layer='features')
}



def apply_transform(image,size=224):
    means,stds=[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]
    #if not isinstance(image, Image.Image):
    #    image = F.to_pil_image(image)

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means,stds)
    ])

    # ùprint(image.shape)
    tensor = transform(image).unsqueeze(0)

    tensor.requires_grad = True

    return tensor

def run(arch, img, target):
    input = Image.open(img).convert('RGB')
    input = apply_transform(input)
    model = arch
    conv_layer = MODEL_CONFIG[model.get_name()]['conv_layer']
    input_layer = MODEL_CONFIG[model.get_name()]['input_layer']
    #fc_layer = MODEL_CONFIG[arch]['fc_layer']
    #cam_extractors = [CAM(model, conv_layer, fc_layer), GradCAM(model, conv_layer),
    #                  GradCAMpp(model, conv_layer), SmoothGradCAMpp(model, conv_layer, input_layer),
    #                  ScoreCAM(model, conv_layer, input_layer), SSCAM(model, conv_layer, input_layer),
    #                  ]#ISSCAM(model, conv_layer, input_layer)]
    cam=ScoreCAM(model.get_arch(), conv_layer, input_layer)
    if torch.cuda.is_available():
        input = input.cuda()
    with torch.no_grad(): out = FF.softmax(model.get_arch()(input),dim=1)
    ret=cam(class_idx=target,scores=out).cpu()
    heatmap = to_pil_image(ret, mode='F')
    plt.figure()
    plt.imshow(heatmap)
    plt.savefig(f'heat.png')
    print(model,target,out[:,target],ret,ret.shape)
    return ret



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
outpath_root = root + 'out/'
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
img_list=get_n_imgs(range(base+1,base+window+1),pattern)

img_dict=IMUT.IMG_list(path=p,GT=GT,labs=labs).select_imgs(img_list)
try:
    os.mkdir(f'{img_dict.get_outpath_root()}vgg16/')
except:
    pass

img_dict.set_outpath_root(f'{img_dict.get_outpath_root()}vgg16/')
print(img_dict.get_outpath_root())
print(img_dict.get_img_dict())



arch=EVMET.Architecture(models.resnet18(pretrained=True).eval(),'resnet18')
#arch=EVMET.Architecture(models.vgg16(pretrained=True).eval(),'vgg16')
avg_drop=ADIC.AverageDrop('average_drop',arch)
inc_conf=ADIC.IncreaseInConfidence('increase_in_confidence',arch)
deletion=DAI.Deletion('deletion',arch)
insertion=DAI.Insertion('insertion',arch)

em=EVMET.MetricsEvaluator(img_dict, saliency_map_extractor=run, model=arch, metrics=[avg_drop, inc_conf, deletion, insertion])

start = time.time()
now = start
M_res,m_res=em.evaluate_metrics()
print(f'Execution time: {int(time.time() - start)}s')
print(f'In {num_imgs} images')
for M in M_res:
    M.final_step(num_imgs)
    print(f'The final {M.get_name()} is: {round(M.get_result(), 2)}%')

f=open(f'{outpath_root}output.txt','a')
output={f'Chunk{chunk_id}':([m.get_result() for m in M_res],[torch.tensor(m.get_result()).mean().item() for m in m_res])}
f.write(str(output))
f.write('\n')
((M.clear(),m.clear()) for M,m in zip(M_res,m_res))
