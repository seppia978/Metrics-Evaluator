import ast

CAMS={'ScoreCAM':0,'GradCAM':0,'GradCAM++':0}

with open('out/output.txt','r') as f:
    txt=f.read().split('\n')

lst=[]
for row in txt[:-1]:
    lst.append(ast.literal_eval(row))

avgdrop,incinconf,delt,inst=[[0 for _ in CAMS] for _ in range(4)]

l=[]
for d in lst:
  val=list(d.values())
  cam=list(d.keys())[0][1]
  CAMS[cam]+=1
  l.append([float(val[0][0][0]),float(val[0][0][1]),float(val[0][1][0]),float(val[0][1][1])])
  avgdrop[list(CAMS.keys()).index(cam)] += float(val[0][0][0])
  incinconf[list(CAMS.keys()).index(cam)] += float(val[0][0][1])
  delt[list(CAMS.keys()).index(cam)] += float(val[0][1][0])
  inst[list(CAMS.keys()).index(cam)] += float(val[0][1][1])

print(avgdrop)
for c in CAMS:
    print(f'''
    \n-----------------------------------------------------------------------------\n
    {c}\t|[ {avgdrop[list(CAMS.keys()).index(c)]/CAMS[c]}\t| {incinconf[list(CAMS.keys()).index(c)]/CAMS[c]}\t| {delt[list(CAMS.keys()).index(c)]/CAMS[c]}\t| {inst[list(CAMS.keys()).index(c)]/CAMS[c]}]
     ''')
#for x in l:
#    print(x[1])
#    input()