import ast
import sys
import pandas as pd

path=sys.argv[1] if len(sys.argv)>1 else None


CAMS={
    # 'CAM':CAM(arch, conv_layer, fc_layer),
    'GradCAM':0,
    'GradCAM++':0,
    #'SmoothGradCAM++':0,
    'ScoreCAM':0,
    #'XGradCAM':0,
    # 'DropCAM'
    #'IntersectionSamCAM':0,
    #'SamCAM':0,
    'IntegratedGradients':0,
    'Saliency':0
    # 'SamCAM3',
    # 'SamCAM4'
    # 'SSCAM',
    # 'ISSCAM'
    }

if path is None:
    path='out/filter/output.txt'
else:
    path=f'out/filter/{path}/output.txt'
with open(path,'r') as f:
    txt=f.read().split('\n')

lst=[]
for row in txt[:-1]:
    #print(row)
    lst.append(ast.literal_eval(row))

et,avgdrop,incinconf,comp,coh,asv=[[0 for _ in CAMS] for _ in range(6)]

l=[]
#lst=lst[4:]
#visu=2
#lst=lst[int(len(lst)/2):] if visu == 2 else lst[:int(len(lst)/2)]
#print(lst)
for i,d in enumerate(lst):
  #print(d)
  val=list(d.values())
  cam=list(d.keys())[0][1]
  CAMS[cam]+=1
  l.append([float(val[0][0][0]),float(val[0][0][1]),float(val[0][0][2])])
  et[list(CAMS.keys()).index(cam)] += float(val[0][0][5])
  avgdrop[list(CAMS.keys()).index(cam)] += float(val[0][0][0])
  incinconf[list(CAMS.keys()).index(cam)] += float(val[0][0][1])
  comp[list(CAMS.keys()).index(cam)] += float(val[0][0][2])
  coh[list(CAMS.keys()).index(cam)] += float(val[0][0][3])
  asv[list(CAMS.keys()).index(cam)] += float(val[0][0][4])
  #inst[list(CAMS.keys()).index(cam)] += float(val[0][1][1])

headers=['Elapsed Time','Average Drop','Increase In Confidence','Complexity','Coherency','Average Score Variance']

vals=[et,avgdrop,incinconf,comp,coh,asv]
df_dict={k:v for k,v in zip(headers,vals)}
print('\n--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print('\n\t\t\t|  Elapsed Time\t\t|  Average Drop\t\t|  Increase In Confidence\t| Complexity\t\t|  Coherency\t\t|  Average Score Variance')
for c in CAMS:
    key=c
    if len(key)<5:
        key+='\t\t'
    elif len(key)<12:
        key+='\t'
    index_cam=list(CAMS.keys()).index(c)

    print(f'''
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n
    {key}\t|  {round(et[index_cam]/CAMS[c],2):.2f}s\t\t|  {round(avgdrop[index_cam]/CAMS[c],2):.2f}%\t\t|  {round(incinconf[index_cam]/CAMS[c],2):.2f}%\t\t\t| {round(comp[index_cam]/CAMS[c],3):.3f}%\t\t|  {round(coh[index_cam]/CAMS[c],3):.3f}%\t\t|  {round(asv[index_cam]/CAMS[c],3):.3f}%''')
print('\n--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
#for x in l:
#    print(x[1])
#    input()
