import ast

CAMS={'GradCAM':0,'GradCAM++':0,'SmoothGradCAM++':0,'ScoreCAM':0,'SSCAM':0,'ISSCAM':0}

with open('out/filter/output.txt','r') as f:
    txt=f.read().split('\n')

lst=[]
for row in txt[:-1]:
    lst.append(ast.literal_eval(row))

avgdrop,incinconf,delt,inst=[[0 for _ in CAMS] for _ in range(4)]

l=[]
#lst=lst[4:]
#visu=2
#lst=lst[int(len(lst)/2):] if visu == 2 else lst[:int(len(lst)/2)]
print(len(lst))
for d in lst[287-12:]:
  val=list(d.values())
  cam=list(d.keys())[0][1]
  CAMS[cam]+=1
  l.append([float(val[0][0][0]),float(val[0][0][1]),float(val[0][1][0]),float(val[0][1][1])])
  avgdrop[list(CAMS.keys()).index(cam)] += float(val[0][0][0])
  incinconf[list(CAMS.keys()).index(cam)] += float(val[0][0][1])
  delt[list(CAMS.keys()).index(cam)] += float(val[0][1][0])
  inst[list(CAMS.keys()).index(cam)] += float(val[0][1][1])

print('\n----------------------------------------------------------------------------------------------------------------------------')
print('\n\t\t\t| Average Drop\t\t| Increase In Confidence\t| Deletion\t\t| Insertion')
for c in CAMS:
    key=c
    if len(key)<5:
        key+='\t\t'
    elif len(key)<12:
        key+='\t'
    print(f'''
    ------------------------------------------------------------------------------------------------------------------------\n
    {key}\t| {round(avgdrop[list(CAMS.keys()).index(c)]/CAMS[c],2):.2f}%\t\t| {round(incinconf[list(CAMS.keys()).index(c)]/CAMS[c],2):.2f}%\t\t\t| {round(delt[list(CAMS.keys()).index(c)]/CAMS[c],3):.3f}\t\t\t| {round(inst[list(CAMS.keys()).index(c)]/CAMS[c],3):.3f}''')
print('\n----------------------------------------------------------------------------------------------------------------------------')
#for x in l:
#    print(x[1])
#    input()