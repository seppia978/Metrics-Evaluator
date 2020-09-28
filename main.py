from images_utils import images_utils as IMUT
import evaluate_metrics as EVMET
import metrics.average_drop_and_increase_of_confidence as ADIC
import metrics.deletion_and_insertion as DAI
import torchvision.models as models
import test

num_imgs = 1
p = ''
root = './'  # '/tirocinio_tesi/Score-CAM000/Score-CAM'
outpath_root = root + 'out/'
data_root = root + 'ILSVRC2012_devkit_t12/data/'

def get_name_images(s):
    return str(s)[-13:-5]

#------ MAIN -------#
with open(data_root + 'IMAGENET_path.txt', 'r') as f:
    p = f.read().strip()

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


img_dict=IMUT.IMG_list(path=p,GT=GT,labs=labs).generate_random(num_imgs)
#img_dict=IMUT.IMG_list(path=p,GT=GT,labs=labs).select_idx(1)

print(list(img_dict.get_keys()))
resnet=EVMET.Architecture(models.resnet18(pretrained=True).eval(),'resnet')
avg_drop=ADIC.AverageDrop('average_drop',resnet)
inc_conf=ADIC.IncreaseInConfidence('increase_in_confidence',resnet)
deletion=DAI.Deletion('deletion',0,resnet)
insertion=DAI.Insertion('insertion',0,resnet)
em=EVMET.MetricsEvaluator(img_dict,saliency_map_extractor=test.run,metrics=[avg_drop,inc_conf,deletion,insertion])
print(EVMET.list_metrics(em))
em.evaluate_metrics()




