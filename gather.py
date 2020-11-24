import ast

CAMS={
    # 'CAM':CAM(arch, conv_layer, fc_layer),
    'GradCAM':0,
    'GradCAM++':0,
    #'SmoothGradCAM++':0,
    'ScoreCAM':0,
    #'XGradCAM':0,
    # 'DropCAM'
    #'IntersectionSamCAM':0,
    'SamCAM':0,
    # 'SamCAM3',
    # 'SamCAM4'
    # 'SSCAM',
    # 'ISSCAM'
    }

with open('out/filter/output.txt','r') as f:
    txt=f.read().split('\n')

lst=[]
for row in txt[:-1]:
    print(row)
    lst.append(ast.literal_eval(row))

avgdrop,incinconf,comp,coh=[[0 for _ in CAMS] for _ in range(4)]

l=[]
#lst=lst[4:]
#visu=2
#lst=lst[int(len(lst)/2):] if visu == 2 else lst[:int(len(lst)/2)]
print(lst)
for i,d in enumerate(lst):
  #print(d)
  val=list(d.values())
  cam=list(d.keys())[0][1]
  CAMS[cam]+=1
  l.append([float(val[0][0][0]),float(val[0][0][1]),float(val[0][0][2])])
  avgdrop[list(CAMS.keys()).index(cam)] += float(val[0][0][0])
  incinconf[list(CAMS.keys()).index(cam)] += float(val[0][0][1])
  comp[list(CAMS.keys()).index(cam)] += float(val[0][0][2])
  coh[list(CAMS.keys()).index(cam)] = coh[list(CAMS.keys()).index(cam)] +float(val[0][0][3]) if float(val[0][0][3])>=0 else coh[list(CAMS.keys()).index(cam)] + coh[list(CAMS.keys()).index(cam)]/(i+1)
  #inst[list(CAMS.keys()).index(cam)] += float(val[0][1][1])

print('\n----------------------------------------------------------------------------------------------------------------------------')
print('\n\t\t\t|  Average Drop\t\t|  Increase In Confidence\t| Complexity\t\t|  Coherency')
for c in CAMS:
    key=c
    if len(key)<5:
        key+='\t\t'
    elif len(key)<12:
        key+='\t'
    print(f'''
    ------------------------------------------------------------------------------------------------------------------------\n
    {key}\t|  {round(avgdrop[list(CAMS.keys()).index(c)]/CAMS[c],2):.2f}%\t\t|  {round(incinconf[list(CAMS.keys()).index(c)]/CAMS[c],2):.2f}%\t\t\t| {round(comp[list(CAMS.keys()).index(c)]/CAMS[c],3):.3f}%\t\t|  {round(coh[list(CAMS.keys()).index(c)]/CAMS[c],3):.9f}''')
print('\n----------------------------------------------------------------------------------------------------------------------------')
#for x in l:
#    print(x[1])
#    input()