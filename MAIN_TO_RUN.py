from images_utils import images_utils as IMUT

# to use MetricEvaluator
from MetricEvaluator import evaluate_metrics as EVMET

# metrics
import metrics.average_drop_and_increase_of_confidence as ADIC
import metrics.deletion_and_insertion as DAI
import metrics.complexity as COMPLEXITY
import metrics.coherency as COHERENCY
import metrics.average_score_variance as ASV
import metrics.elasped_time as EA

# misc
import torchvision.models as models
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import sys
import time
import os

# backbones
from torchcammaster.torchcam.cams import IntersectionSamCAM,DropCAM, SamCAM3, SamCAM4, SamCAM2, SamCAM, GradCAM,XGradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM, ISSCAM
from captum.attr import IntegratedGradients,Saliency,Occlusion,FeaturePermutation


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

    # print(image.shape)
    tensor = transform(image).unsqueeze(0)

    tensor.requires_grad = True

    return tensor

def run(arch=None, img=None, out=None, target=None,**params):

    key=params['extractor']
    #key='ScoreCAM'
    if key=='GradCAM':
        cam=GradCAM(arch, conv_layer)
    elif key=='XGradCAM':
        cam=XGradCAM(arch,conv_layer)
    elif key=='GradCAM++':
        cam=GradCAMpp(arch, conv_layer)
    elif key=='SmoothGradCAM++':
        cam=SmoothGradCAMpp(arch, conv_layer, input_layer)
    elif key=='ScoreCAM':
        cam=ScoreCAM(arch, conv_layer, input_layer)
    elif key=='IntersectionSamCAM':
        cam=IntersectionSamCAM(arch,conv_layer,input_layer)
    elif key=='SamCAM':
        cam=SamCAM(arch,conv_layer)
    elif key=='SamCAM2':
        cam=SamCAM2(arch,conv_layer,p=0.25)
    elif key=='SamCAM3':
        cam=SamCAM3(arch,conv_layer,p=1.0)
    elif key=='SamCAM4':
        cam=SamCAM4(arch,conv_layer,input_layer)
    elif key=='DropCAM':
        cam=DropCAM(arch,conv_layer,input_layer)
    elif key=='SSCAM':
        cam=SSCAM(arch, conv_layer, input_layer,num_samples=10)
    elif key=='ISSCAM':
        cam=ISSCAM(arch, conv_layer, input_layer)
    elif key=='IntegratedGradients'or key=='IGDown':
        ig=IntegratedGradients(arch.arch)
        cam=ig.attribute
    elif key=='Saliency' or key=='SaliencyDown':
        saliency=Saliency(arch.arch)
        cam=saliency.attribute
    elif key=="FakeCAM":
        cam=None
    elif key=='Occlusion':
        occ=Occlusion(arch.arch)
        cam=occ.attribute

    model = arch

    if type(img)==Image.Image:
        inp=apply_transform(img).cuda()
    else:
        inp=img
    out=F.softmax(model.arch(inp),dim=1)

    if cam is not None:
        if 'GradCAM' in key:
            salmap = cam(inp,target=target,scores=out)
        elif 'Occlusion' in key:
            salmap = cam(inp,sliding_window_shapes=(3,45,45),strides=(3,9,9), target=target)
            salmap = torch.abs(salmap.sum(dim=1))
        else:
            salmap = cam(inp, target=target)
    else:
        salmap=torch.ones((inp.shape[-1],inp.shape[-2]))
        salmap[0,0]=0

    # remove 50% less important pixel
    #salmap.view(1,-1)[0,(1-salmap).view(1,-1).topk(int((salmap.shape[-1]**2)/2))[1]]=0.

    salmap=salmap.to(torch.float32)
    if key=='IntegratedGradients' or key=='Saliency':
        salmap = torch.abs(salmap.sum(dim=1))
        salmap = (salmap - salmap.min()) / (salmap.max() - salmap.min())

        salmap_previous = salmap
        sigma=3.

        # torchvision gaussian
        '''
        trans=transforms.Compose([
            transforms.GaussianBlur(3,sigma=sigma)
        ])
        salmap=trans(salmap)
        '''

        #scipy gaussian
        '''
        from scipy.ndimage import gaussian_filter as GB
        salmap=torch.from_numpy(GB(salmap.cpu().detach().numpy(),sigma))
        #'''

        salmap = torch.abs(salmap)
        salmap = (salmap - salmap.min()) / (salmap.max() - salmap.min())
        salmap=salmap.squeeze(0)
    elif key=='IGDown' or key=='SaliencyDown':
        salmap = torch.abs(salmap.sum(dim=1))
        salmap = (salmap - salmap.min()) / (salmap.max() - salmap.min())
        salmap_previous=salmap
        salmap=F.interpolate(salmap.unsqueeze(0), (7,7), mode='bilinear', align_corners=False)
        salmap=F.interpolate(salmap, salmap_previous.shape[-2:], mode='bilinear', align_corners=False)
        salmap = torch.abs(salmap.sum(dim=1))
        salmap = (salmap - salmap.min()) / (salmap.max() - salmap.min())

    salmap = torch.abs(salmap)
    salmap = (salmap - salmap.min()) / (salmap.max() - salmap.min())
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


#---------------------- MAIN ----------------------#
i=0
chunk_id=-1
chunk_dim=0
params=[]
for arg in sys.argv[1:]:
    if not(i%2==0):
        params.append(arg)
    i+=1

print(len(sys.argv[1:]))
filtered=True if (len(sys.argv[1:]))/2 != 3 else False

chunk_id,chunk_dim=[int(x) for x in params[0:2]]
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



#img_dict=IMUT.IMG_list(path=p,GT=GT,labs=labs).generate_random(num_imgs)
base,window=chunk_id*chunk_dim+displacement,chunk_dim
pattern='ILSVRC2012_val_********.JPEG'
#img_list=get_n_imgs(range(base+1,base+window+1),pattern)
if filtered:
    with open('filter.txt','r') as f:
        txt=f.read()

    img_list=[p.split()[0] for p in txt.strip().split('\n')[base:base+window]]
    #print(*(enumerate(img_list)))
    img_list={get_num_img(p.split()[0]):p.split()[0] for p in img_list}
    try:
        os.mkdir(f'{outpath_root}filter/')
    except:
        pass
    img_dict = IMUT.IMG_list(path=p, outpath_root='out/filter/', GT=GT, labs=labs).select_imgs(img_list)
else:
    img_dict=IMUT.IMG_list(path=p, outpath_root='out/filter/', GT=GT, labs=labs).select_all_imgs_from_path()

#rch=EVMET.Architecture(models.resnet18(pretrained=True).eval(),'resnet18','layer4')
#arch=EVMET.Architecture(models.resnet101(pretrained=True).eval(),'resnet101','layer4')
#arch=EVMET.Architecture(models.resnet152(pretrained=True).eval(),'resnet152','layer4')
arch=EVMET.Architecture(models.vgg16(pretrained=True).eval(),'vgg16','features_29')

print(chunk_id,chunk_dim)
avg_drop=ADIC.AverageDrop('average_drop',arch)
inc_conf=ADIC.IncreaseInConfidence('increase_in_confidence',arch)
deletion=DAI.Deletion('deletion',arch)
insertion=DAI.Insertion('insertion',arch)
complexity=COMPLEXITY.Complexity('Average complexity',arch)
coherency=COHERENCY.Coherency('Average coherency',arch)
avg_score_var=ASV.AverageScoreVariance('Average score variance',arch)
ea=EA.ElapsedTime('Elapsed Time',nimgs=len(img_dict))



em = EVMET.MetricsEvaluator(img_dict, saliency_map_extractor=run, model=arch,
                                metrics=[deletion,insertion])
start = time.time()
now = start
path0=img_dict.get_outpath_root()

conv_layer = arch.layer#MODEL_CONFIG[arch.name]['conv_layer']
input_layer = MODEL_CONFIG[arch.name]['input_layer']
fc_layer = arch.arch.classifier[6]#MODEL_CONFIG[arch.name]['fc_layer'] # for vgg
#fc_layer=MODEL_CONFIG[arch.name]['fc_layer'] # for resnet
cam_extractors = [
                      #'CAM':CAM(arch, conv_layer, fc_layer),
                      #'GradCAM',
                      #'GradCAM++',
                      #'SmoothGradCAM++',
                      #'ScoreCAM',
                      'IntegratedGradients',
                      #'IGDown',
                      #'SaliencyDown',
                      #'Saliency',
                      'FakeCAM',
                      'Occlusion'
                      #'XGradCAM',
                      #'DropCAM'
                      #'IntersectionSamCAM',
                      #'SamCAM',
                      #'SamCAM3',
                      #'SamCAM4'
                      #'SSCAM',
                      #'ISSCAM'
                 ]
for idx,c in enumerate(cam_extractors):
    coherency.saliency_map_extractor=COHERENCY.SaliencyMapExtractor(c, run)
    try:
        os.mkdir(f'{path0}{str(c)}/')
    except:
        pass
    img_dict.set_outpath_root(f'{path0}{str(c)}/')
    print(img_dict.get_outpath_root())
    print(img_dict.get_img_dict())
    print(f'{path0}output.txt')

    coherency.outpath=img_dict.outpath_root
    M_res,m_res=em(extractor=c,print_metrics=True)

    #print(f'Execution time: {int(time.time() - start)}s')
    #print(f'In {num_imgs} images')
    #for M in M_res:
    #    print(f'The final {M.get_name()} is: {M.get_result()}%')

    time.sleep(chunk_id)
    f=open(f'{path0}output.txt','a')
    output={(f'Chunk{chunk_id}',f'{c}'):([m.get_result() for m in M_res],[torch.tensor(m.get_result()).mean().item() for m in m_res])}
    try:
        f.write(str(output))
        f.write('\n')
    except:
        print('no')
    for m in em.metrics:
        m.clear()