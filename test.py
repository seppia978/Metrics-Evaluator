# pip install importlib_resources

import torch
import torch.nn.functional as F
import torchvision.models as models

from utils import *
from cam.scorecam import *

def run(i=0,arch='resnet',path='out/', img=None):
  outpath=path
  if img is not None:
    if arch == 'alexnet':
      # alexnet
      alexnet = models.alexnet(pretrained=True).eval()
      alexnet_model_dict = dict(type='alexnet', arch=alexnet, layer_name='features_10',input_size=(224, 224))
      alexnet_scorecam = ScoreCAM(alexnet_model_dict)

      input_image = load_image(img)
      input_ = apply_transforms(input_image)
      if torch.cuda.is_available():
        input_ = input_.cuda()

      out=alexnet(input_)
      predicted_class = out.max(1)[-1]

      scorecam_map = alexnet_scorecam(input_)
      name='alex'+str(i)+'.png'
      outpath+=name
      #basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path=outpath)
    elif arch == 'vgg16':
      # vgg
      vgg = models.vgg16(pretrained=True).eval()
      vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name='features_29',input_size=(224, 224))
      vgg_scorecam = ScoreCAM(vgg_model_dict)

      input_image = load_image(img)
      input_ = apply_transforms(input_image)
      if torch.cuda.is_available():
        input_ = input_.cuda()
      out=vgg(input_)
      predicted_class = out.max(1)[-1]

      scorecam_map = vgg_scorecam(input_)
      name='vgg16'+str(i)+'.png'
      outpath+=name
      #basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path=outpath)

    elif arch == 'resnet':
      # resnet
      resnet = models.resnet18(pretrained=True).eval()
      resnet_model_dict = dict(type='resnet18', arch=resnet, layer_name='layer4',input_size=(224, 224))
      resnet_scorecam = ScoreCAM(resnet_model_dict)

      input_image = load_image(img)
      input_ = apply_transforms(input_image)
      if torch.cuda.is_available():
        input_ = input_.cuda()
      out=resnet(input_)
      #score_c,predicted_class = out,out.max(1)[-1]

      scorecam_map = resnet_scorecam(input_)
      name='resnet'+str(i)+'.png'
      outpath+=name
      #basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path=outpath)

    return F.softmax(out, dim=1), scorecam_map.type(torch.FloatTensor).cpu()

