import os
import random
import torch
import torchvision.transforms as transforms

def trans(img):
  means = [0.485, 0.456, 0.406]
  stds = [0.229, 0.224, 0.225]

  transform = transforms.Compose([
    transforms.Normalize(means, stds)
  ])

  tensor = transform(img).unsqueeze(0)

  return tensor

def denormalize(tensor):
    means, stds = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    denormalized=transforms.Normalize(-1*means/stds,1.0/stds)(tensor)

    return denormalized

class IMG_list:
    def __init__(self,path=None,outpath_root='out/',img_dict=None,GT=None,labs=None):
        if type(GT) is not dict and GT is not None:
            GT={str(i):x for i,x in enumerate(GT)}
        if type(labs) is not dict and labs is not None:
            labs={str(i):x for i,x in enumerate(labs)}

        self.path,self.outpath_root,self.img_dict,self.GT,self.labs=path,outpath_root,img_dict,GT,labs


    def __len__(self):
        return len(self.img_dict)

    def get_path(self):
        return self.path
    def get_outpath_root(self):
        return self.outpath_root
    def get_img_dict(self):
        return self.img_dict
    def get_nth_img(self, n):
        return list(self.img_dict.values())[n]
    def get_idx_img(self, idx):
        return self.img_dict[idx]
    def get_list(self):
        return list(self.img_dict.values())
    def get_keys(self):
        return list(self.img_dict.keys())
    def get_items(self):
        return [x for x in self.img_dict.items()]
    def get_GT(self):
        return self.GT
    def get_labels(self):
        return self.labs

    def get_name_images(self,s):
        return str(s)[-13:-5]

    def get_num_img(self,s):
        return int(s[-13:-5])

    def set_path(self,path):
        self.path=path
    def set_outpath_root(self,outpath_root):
        self.outpath_root=outpath_root
    def set_img_dict(self,img_dict):
        self.img_dict=img_dict
    def set_GT(self,GT):
        if type(GT) is not dict:
            GT={str(i):x for i,x in enumerate(GT)}
        self.GT=GT
    def set_labels(self,labs):
        if type(labs) is not dict:
            labs={str(i):x for i,x in enumerate(labs)}
        self.labs=labs

    def add_img(self,idx,img):
        self.img_dict[idx]=img

    def select_all_imgs_from_path(self):
        images_list = os.listdir(self.path)
        self.img_dict={k:v for k,v in enumerate(images_list)}
        return self

    def select_imgs(self, id):
        i,l=list(id.keys()),list(id.values())
        images_list=[im for im in l]
        img_dict = {k: v for k, v in zip(i, images_list)}
        self.img_dict=img_dict
        return self
    def select_idx(self,images_id):
        if type(images_id) is not list:
            try:
                images_id=list(images_id)
            except:
                images_id=[images_id]
        images_list = [os.listdir(self.path)[int(i)] for i in images_id]
        img_dict = {k: v for k, v in zip(images_id, images_list)}
        self.img_dict = img_dict
        return self
    def generate_random(self,num_imgs=0):
        try:
            random.seed()
            images_id = random.sample(range(len(os.listdir(self.path))), num_imgs)
            images_list = [os.listdir(self.path)[int(i)] for i in images_id]
            img_dict = {k: v for k, v in zip(images_id, images_list)}
            self.img_dict=img_dict
            return self
        except:
            raise ValueError('Path must not be None and num_imgs must be greater than 0')
