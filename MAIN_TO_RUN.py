import argparse
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
import pathlib

# backbones
from torchcammaster.torchcam.cams import CAM,IntersectionSamCAM,DropCAM, SamCAM3, SamCAM4, SamCAM2, SamCAM, GradCAM,XGradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM, ISSCAM
from captum.attr import IntegratedGradients,Saliency,Occlusion


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
    if key=='GradCAM'.lower():
        cam=GradCAM(arch, conv_layer)
    elif key=='CAM'.lower():
        cam=CAM(arch, conv_layer, fc_layer)
    elif key=='XGradCAM'.lower():
        cam=XGradCAM(arch,conv_layer)
    elif key=='GradCAM++'.lower():
        cam=GradCAMpp(arch, conv_layer)
    elif key=='SmoothGradCAM++'.lower():
        cam=SmoothGradCAMpp(arch, conv_layer, input_layer)
    elif key=='ScoreCAM'.lower():
        cam=ScoreCAM(arch, conv_layer, input_layer)
    elif key=='IntersectionSamCAM'.lower():
        cam=IntersectionSamCAM(arch,conv_layer,input_layer)
    elif key=='SamCAM'.lower():
        cam=SamCAM(arch,conv_layer)
    elif key=='SamCAM2'.lower():
        cam=SamCAM2(arch,conv_layer,p=0.25)
    elif key=='SamCAM3'.lower():
        cam=SamCAM3(arch,conv_layer,p=1.0)
    elif key=='SamCAM4'.lower():
        cam=SamCAM4(arch,conv_layer,input_layer)
    elif key=='DropCAM'.lower():
        cam=DropCAM(arch,conv_layer,input_layer)
    elif key=='SSCAM'.lower():
        cam=SSCAM(arch, conv_layer, input_layer,num_samples=10)
    elif key=='ISSCAM'.lower():
        cam=ISSCAM(arch, conv_layer, input_layer)
    elif 'IntegratedGradients'.lower() in key or key=='IGDown'.lower():
        ig=IntegratedGradients(arch.arch)
        cam=ig.attribute
    elif key=='Saliency'.lower() or key=='SaliencyDown'.lower():
        saliency=Saliency(arch.arch)
        cam=saliency.attribute
    elif key=="FakeCAM".lower():
        cam=None
    elif key=='Occlusion'.lower():
        occ=Occlusion(arch.arch)
        cam=occ.attribute

    model = arch

    if type(img)==Image.Image:
        inp=apply_transform(img).cuda()
    else:
        inp=img
    out=F.softmax(model.arch(inp),dim=1)

    if cam is not None:
        if 'GradCAM'.lower() in key:
            salmap = cam(inp,target=target,scores=out)
        elif 'Occlusion'.lower() in key:
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
    if 'IntegratedGradients'.lower() in key or key=='Saliency'.lower():
        salmap = torch.abs(salmap.sum(dim=1))
        salmap = (salmap - salmap.min()) / (salmap.max() - salmap.min())

        salmap_previous = salmap
        if '20' in key:
            sigma=20
        elif '5' in key:
            sigma=5
        else:
            sigma=3

        # torchvision gaussian
        '''
        trans=transforms.Compose([
            transforms.GaussianBlur(3,sigma=sigma)
        ])
        salmap=trans(salmap)
        '''

        #scipy gaussian
        #'''
        from scipy.ndimage import gaussian_filter as GB
        salmap=torch.from_numpy(GB(salmap.cpu().detach().numpy(),sigma))
        #'''

        salmap = torch.abs(salmap)
        salmap = (salmap - salmap.min()) / (salmap.max() - salmap.min())
        salmap=salmap.squeeze(0)
    elif key=='IGDown'.lower() or key=='SaliencyDown'.lower():
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
    return {k: v for k, v in zip(range(len(l)), ret)}


#---------------------- MAIN ----------------------#

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-rp',"--res_path", type=str, help='results path',required=True)
    parser.add_argument('-cid',"--chunk_id", type=int, help='Job chunk id',required=True)
    parser.add_argument('-cdim',"--chunk_dim", type=int, help='Job chunk dimension',required=True)
    parser.add_argument('-cnn',"--cnn", type=str, help='Backbone to use',required=True)
    parser.add_argument('-m',"--metrics", type=str, help='Metrics to evaluate',required=True,nargs='+')
    parser.add_argument('-am',"--attr_methods", type=str, help='Attribution methods to evaluate',required=True,nargs='+')


    args = parser.parse_args()

    res_path=args.res_path
    chunk_id=args.chunk_id
    chunk_dim=args.chunk_dim
    arch_name=args.cnn.lower()
    metric_names=[el.lower() for el in args.metrics]
    cam_extractors=[el.lower() for el in args.attr_methods]

    if res_path[-1] is not '/':
        res_path=res_path+'/'

    pathlib.Path(res_path).mkdir(parents=True, exist_ok=True)
    i=0

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
    displacement=0
    base,window=chunk_id*chunk_dim+displacement,chunk_dim
    pattern='ILSVRC2012_val_********.JPEG'
    #img_list=get_n_imgs(range(base+1,base+window+1),pattern)
    #p='''/homes/spoppi/tirocinio_tesi/Score-CAM000/Score-CAM/out/filter'''
    # filtered=True
    # if filtered:
    #     #with open('filter.txt','r') as f:
    #     #    txt=f.read()
    #
    #     #img_list=[p.split()[0] for p in txt.strip().split('\n')[base:base+window]]
    #     #print(*(enumerate(img_list)))
    #     img_list =['1','1011','24','49','719','118','1096','1312','567','26','865']
    #     #img_list=['1965']
    #     img_list=get_n_imgs(img_list,pattern)
    #     img_list={get_num_img(p.split()[0]):p.split()[0] for p in img_list.values()}
    #     print(img_list)
    #     try:
    #         os.mkdir(f'{outpath_root}filter/')
    #     except:
    #         pass
    #     img_dict = IMUT.IMG_list(path=p, outpath_root='out/filter/', GT=GT, labs=labs).select_imgs(img_list)
    # else:
    img_list={k:v for k,v in enumerate(sorted(os.listdir(p))[base:base+window])}
    print(img_list)
    img_dict=IMUT.IMG_list(path=p, outpath_root=res_path, GT=GT, labs=labs).select_imgs(img_list)
    #img_dict=IMUT.IMG_list(path=p,outpath_root='out/filter/', GT=GT, labs=labs).select_particular_img('gr.jpg')


    arch_dict={
        'resnet18': EVMET.Architecture(models.resnet18(pretrained=True).eval(),'resnet18','layer4'),
        'resnet50': EVMET.Architecture(models.resnet50(pretrained=True).eval(),'resnet50','layer4'),
        'vgg16': EVMET.Architecture(models.vgg16(pretrained=True).eval(),'vgg16','features_29')
    }
    arch=arch_dict[arch_name]
    #arch=EVMET.Architecture(models.resnet101(pretrained=True).eval(),'resnet101','layer4')
    #arch=EVMET.Architecture(models.resnet152(pretrained=True).eval(),'resnet152','layer4')
    #arch=EVMET.Architecture(models.vgg16(pretrained=True).eval(),'vgg16','features_29')

    metrics_dict={
        'average_drop': ADIC.AverageDrop('average_drop',arch),
        'average_increase' : ADIC.IncreaseInConfidence('average_increase',arch),
        'increase_in_confidence' : ADIC.IncreaseInConfidence('increase_in_confidence',arch),
        'deletion': DAI.Deletion('deletion',arch),
        'insertion': DAI.Insertion('insertion',arch),
        'average_complexity': COMPLEXITY.Complexity('Average complexity',arch),
        'average_coherency': COHERENCY.Coherency('Average coherency',arch),
        'average_score_variance': ASV.AverageScoreVariance('Average score variance',arch)
    }

    metrics={m:metrics_dict[m] for m in metric_names}

    # avg_drop=ADIC.AverageDrop('average_drop',arch)
    # inc_conf=ADIC.IncreaseInConfidence('increase_in_confidence',arch)
    # deletion=DAI.Deletion('deletion',arch)
    # insertion=DAI.Insertion('insertion',arch)
    # complexity=COMPLEXITY.Complexity('Average complexity',arch)
    # coherency=COHERENCY.Coherency('Average coherency',arch)
    # avg_score_var=ASV.AverageScoreVariance('Average score variance',arch)
    # ea=EA.ElapsedTime('Elapsed Time',nimgs=len(img_dict))



    em = EVMET.MetricsEvaluator(img_dict, saliency_map_extractor=run, model=arch,
                                    metrics=[metrics[m] for m in metrics])
    start = time.time()
    now = start

    conv_layer = arch.layer#MODEL_CONFIG[arch.name]['conv_layer']
    input_layer = MODEL_CONFIG[arch.name]['input_layer']
    #fc_layer = arch.arch.classifier[6]#MODEL_CONFIG[arch.name]['fc_layer'] # for vgg
    fc_layer=MODEL_CONFIG[arch.name]['fc_layer'] # for resnet

    # cam_extractors = [
    #                       #'CAM'
    #                       #'GradCAM',
    #                       #'GradCAM++',
    #                       #'SmoothGradCAM++',
    #                       #'ScoreCAM',
    #                       'IntegratedGradients20',
    #                       'IntegratedGradients5',
    #                       'IntegratedGradients3',
    #                       #'IGDown',
    #                       #'SaliencyDown',
    #                       #'Saliency',
    #                       #'FakeCAM',
    #                       #'Occlusion'
    #                       #'XGradCAM',
    #                       #'DropCAM'
    #                       #'IntersectionSamCAM',
    #                       #'SamCAM',
    #                       #'SamCAM3',
    #                       #'SamCAM4'
    #                       #'SSCAM',
    #                       #'ISSCAM'
    #                  ]
    for idx,c in enumerate(cam_extractors):
        if 'average_coherency' in metric_names:
            metrics['average_coherency'].saliency_map_extractor=COHERENCY.SaliencyMapExtractor(c, run)
        try:
            os.mkdir(f'{res_path}{str(c)}/')
        except:
            pass
        img_dict.set_outpath_root(f'{res_path}{str(c)}/')
        print(img_dict.get_outpath_root())
        #print(img_dict.get_img_dict())
        print(f'{res_path}output.txt')

        if 'average_coherency' in metric_names:
            metrics['average_coherency'].outpath=img_dict.outpath_root
        if '20' in c:
            sigma = 20
        elif '5' in c:
            sigma = 5
        else:
            sigma = 3
        M_res,m_res=em(extractor=c,print_metrics=True,sigma=sigma)

        #print(f'Execution time: {int(time.time() - start)}s')
        #print(f'In {num_imgs} images')
        #for M in M_res:
        #    print(f'The final {M.get_name()} is: {M.get_result()}%')

        time.sleep(chunk_id)
        f=open(f'{res_path}output.txt','a')
        output={(f'Chunk{chunk_id}',f'{c}'):([m.get_result() for m in M_res],[torch.tensor(m.get_result()).mean().item() for m in m_res])}
        try:
            f.write(str(output))
            f.write('\n')
        except:
            print('no')
        for m in em.metrics:
            m.clear()