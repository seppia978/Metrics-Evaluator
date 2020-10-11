import torch
import torchvision.models as models
import torch.nn.functional as FF
import os

from ScoreCAM.utils import *


from utils_functions import plot

def list_metrics(em):
    return em.__list_metrics__()

class Architecture:
    def __init__(self,arch, name,T=None,DeT=None,means=[0.485, 0.456, 0.406],stds=[0.229, 0.224, 0.225]):
        self.arch,self.name,self.T,self.DeT,self.means,self.stds=arch,name,T,DeT,means,stds

    def get_name(self):
        return self.name
    def get_arch(self):
        return self.arch
    def get_custom_transformation(self):
        return self.T
    def get_custom_detransformation(self):
        return self.DeT
    def get_means(self):
        return self.means
    def get_stds(self):
        return self.stds

    def set_name(self,name):
        self.name=name
    def set_arch(self,arch):
        self.arch=arch
    def set_custom_transformation(self,T):
        self.T=T
    def set_custom_detransformation(self,DeT):
        self.DeT=DeT
    def set_means(self,m):
        self.means=m
    def set_stds(self,s):
        self.stds=s

    def apply_transform(self,image,size=224):
        if not isinstance(image, Image.Image):
            image = F.to_pil_image(image)

        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(self.means, self.stds)
        ])

        tensor = transform(image).unsqueeze(0)

        tensor.requires_grad = True

        return tensor

    def detransform(self, tensor):
        denormalized = tensor.clone()

        for channel, mean, std in zip(denormalized[0], self.means, self.stds):
            channel.mul_(std).add_(mean)

        return denormalized

class Metric:

    # CONSTRUCTOR
    def __init__(self,name,alldataset):
        self.name,self.alldataset=name,alldataset

    # UTILS
    def help(self):
        return  '''Metric helper: You simply must override the two methods update(.) and final_step(.)
                '''
    def is_all_dataset(self):
        return self.alldataset

    # GETTERS
    def get_name(self):
        return self.name

    # SETTERS
    def set_name(self,name):
        self.name=name
    def set_alldataset(self,alldataset):
        self.alldataset=alldataset


    # TO OVERRIDE
    def update(self,*args):
        raise NotImplementedError("method needs to be defined by sub-class")

    def final_step(self,**kwargs):
        raise NotImplementedError("method needs to be defined by sub-class")

    def clear(self):
        raise NotImplementedError("method needs to be defined by sub-class")

class MetricOnSingleExample(Metric):

    # CONSTRUCTOR
    def __init__(self,name,result,res_list=[],alldataset=False):
        super().__init__(name,alldataset)
        self.result,self.res_list=result,res_list

    # UTILS
    def clear_list(self):
        self.res_list = []

    def clear(self):
        self.result = 0

    def get_result(self):
        return self.result
    def get_res_list(self):
        return self.res_list


class MetricOnAllDataset(Metric):
    # CONSTRUCTOR
    def __init__(self,name,result,alldataset=True):
        super().__init__(name,alldataset)
        self.result=result

    # UTILS
    def clear(self):
        self.result=0

    def get_result(self):
        return self.result

class MetricsEvaluator:

    # LIST OF METRICS AVAILABLE
    available_metrics=[]

    # CONSTRUCTOR
    def __init__(self,img_dict, saliency_map_extractor=None, model=Architecture(models.resnet18(pretrained=True).eval(),'resnet18'),metrics=[],times=1):
        self.img_dict,self.saliency_map_extractor,self.model,self.metrics,self.times=img_dict, saliency_map_extractor, model,[m for m in metrics],times
        for x in self.metrics:
            self.available_metrics.append(x.get_name())

    # UTILS
    def __list_metrics__(self):
        return self.available_metrics

    #GETTERS
    def get_imgs(self):
        return self.img_dict
    def get_saliency_map_extractor(self):
        return self.saliency_map_extractor
    def get_model(self):
        return self.model
    def get_metrics(self):
        return self.metrics
    def get_metric_by_name(self,name):
        return (x for x in self.metrics if x.get_name()==name)
    def get_times(self):
        return self.times

    # SETTERS
    def set_imgs(self,img_dict):
        self.img_dict=img_dict
    def set_saliency_map_extractor(self, saliency_map_extractor):
        self.saliency_map_extractor=saliency_map_extractor
    def set_model(self,model):
        self.model=model
    def set_metrics(self,metrics):
        self.metrics=metrics
    def set_times(self,times):
        self.times=times

    # UTILS
    def clear_metrics(self):
        self.metrics=[]
    def delete_metric_by_name(self,name):
        for i in range(len(self.metrics)):
            x=self.metrics[i]
            if x.get_name()==name:
                self.metrics.pop(x)
    def append_metric(self,name,func):
        self.metrics.append(Metric(name,func))

    # EVALUATE METRICS
    def get_explanation_map(self,img=None,target=None):
        return self.saliency_map_extractor(arch=self.model, img=img,target=target)

    def evaluate_metrics(self):#,**kwargs):
        img_dict, saliency_map_extractor, model, metrics, times=self.img_dict,self.saliency_map_extractor,self.model,self.metrics,self.times
        #for k in kwargs:
        #    k=kwargs[k]
        GT=img_dict.get_GT()
        labs=img_dict.get_labels()
        num_imgs = len(img_dict)
        '''
        if model == 'resnet18':
            arch_obj = Architecture(models.resnet18(pretrained=True).eval(),model)
        elif model == 'vgg16':
            arch_obj = Architecture(models.vgg16(pretrained=True).eval(),model)
        elif model == 'alexnet':
            arch_obj = Architecture(models.alexnet(pretrained=True).eval(),model)
        '''

        if torch.cuda.is_available():
            arch = self.model.get_arch().cuda()
        precision = 100


        if metrics is not []:
            for _ in range(times):
                m_res=[m for m in metrics if m.is_all_dataset() is not True]
                M_res = [m for m in metrics if m.is_all_dataset()]

                # For each image
                for i, (k,img) in enumerate(img_dict.get_items()):
                    print(f'image {i}')

                    # Preliminary steps
                    outpath=img_dict.get_outpath_root()+f'{k}_{img}/'
                    inp_0=load_image(img_dict.get_path() + '/' + img)
                    try:
                        os.mkdir(outpath)
                    except:
                        pass
                    inp_0.save(f'{outpath}{img}')

                    # Apply transformation
                    inp = self.model.apply_transform(inp_0)
                    if torch.cuda.is_available():
                        inp = inp.cuda()
                    #print(f'Before test.run: {round(time.time() - now, 0)}s')

                    with torch.no_grad():
                        out = FF.softmax(arch(inp), dim=1)

                    # Get class idx for this img
                    class_idx = out.max(1)[-1].item()
                    class_name = labs[str(class_idx)]
                    gt_name = GT[str(img[-13:-5])][0].split()[1]

                    # Get explanation map using the explanation method defined when creating the object
                    saliency_map=self.get_explanation_map(img=img_dict.get_path() + '/' + img,target=class_idx)
                    out,saliency_map=out.detach(),saliency_map.detach()
                    F.to_pil_image(saliency_map.squeeze(0)).save(f'{outpath}/exp_map.png')
                    #print(f'After test.run: {round(time.time() - now, 0)}s')
                    if torch.cuda.is_available():
                        saliency_map = saliency_map.cuda()
                    #print(f'Before arch: {round(time.time() - now, 0)}s')

                    #out_sal = FF.softmax(arch(inp * saliency_map), dim=1)
                    #print(f'After arch: {round(time.time() - now, 0)}s')

                    # print(type(out_sal),out_sal.shape)
                    #Y_i_c = out.max(1)[0].item()


                    #O_i_c = out_sal[:, class_idx][0].item()

                    # Updates ad plots of the metrics on a single example
                    Y=[]
                    L=[]

                    plt.figure()
                    plt.imshow(denormalize(inp*saliency_map).squeeze(0).cpu().detach().permute(1,2,0).numpy())
                    plt.savefig(f'{outpath}/inp*sal.png')
                    for c,m in enumerate(m_res):
                        m.update(inp,out,saliency_map)
                        m.final_step()
                        print(f'The final {m.get_name()} score is {m.get_result()}')
                        #print(m.get_res_list())
                        Y.append(m.get_res_list())
                        L.append((m.get_name(),m.get_result()))
                        m.clear_list()
                    plot(torch.arange(0, 1, 1 / precision), Y,
                         label=L,
                         path=f'{outpath}plot_{k}.png',
                         title=f'label={class_name}, GT={gt_name}')
                    for c,M in enumerate(M_res):
                        M.update(inp,out,saliency_map)
                    #print(f'After one img: {int(time.time() - now)}s')
                    #now = time.time()

        return M_res,m_res



