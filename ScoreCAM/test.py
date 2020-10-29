# pip install importlib_resources

import torch
import torch.nn.functional as F
import torchvision.models as models

from ScoreCAM.utils import *
from ScoreCAM.cam.scorecam import *

def run(*params, arch=None, img=None, out=None, target=None, path='out/',):
  outpath=path
  if img is not None:
    if arch.get_name() == 'alexnet':
      # alexnet
      alexnet = models.alexnet(pretrained=True).eval()
      alexnet_model_dict = dict(type='alexnet', arch=alexnet, layer_name='features_10',input_size=(224, 224))
      alexnet_scorecam = ScoreCAM(alexnet_model_dict)

      #input_image = load_image(img)
      input_ = img#apply_transforms(input_image)
      if torch.cuda.is_available():
        input_ = input_.cuda()

      out=alexnet(input_)
      predicted_class = out.max(1)[-1]

      scorecam_map = alexnet_scorecam(input_,class_idx=target)
      name='alex'+str(0)+'.png'
      outpath+=name
      #basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path=outpath)
    elif arch.get_name() == 'vgg16':
      # vgg

      vgg = models.vgg16(pretrained=True).eval()
      vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name='features_29',input_size=(224, 224))

      vgg_scorecam = ScoreCAM(vgg_model_dict)

      # input_image = load_image(img)
      input_ = img  # apply_transforms(input_image)

      if torch.cuda.is_available():
        input_ = input_.cuda()

      out=vgg(input_)

      #predicted_class = out.max(1)[-1]

      #print(torch.cuda.memory_summary())
      scorecam_map = vgg_scorecam(input_,class_idx=target)

      name='vgg16'+str(0)+'.png'
      outpath+=name
      #basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path=outpath)

    elif arch.get_name() == 'resnet18':
      # resnet
      resnet = models.resnet18(pretrained=True).eval()
      resnet_model_dict = dict(type='resnet18', arch=resnet, layer_name='layer4',input_size=(224, 224))
      resnet_scorecam = ScoreCAM(resnet_model_dict)

      # input_image = load_image(img)
      input_ = img  # apply_transforms(input_image)

      if torch.cuda.is_available():
        input_ = input_.cuda()
      out=resnet(input_)
      #score_c,predicted_class = out,out.max(1)[-1]

      scorecam_map = resnet_scorecam(input_,class_idx=target)
      name='resnet'+str(0)+'.png'
      outpath+=name
      #basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path=outpath)

    return scorecam_map.type(torch.FloatTensor).cpu()

