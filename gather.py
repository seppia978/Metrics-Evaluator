import ast

with open('out/output.txt','r') as f:
    txt=f.read().split('\n')

lst=[]
for row in txt[:-1]:
    lst.append(ast.literal_eval(row))

avgdrop,incinconf,delt,inst=0,0,0,0
l=[]
for d in lst:
  val=list(d.values())
  #print(val,val[0],val[0][0][0])
  l.append([float(val[0][0][0]),float(val[0][0][1]),float(val[0][1][0]),float(val[0][1][1])])
  avgdrop += float(val[0][0][0])
  incinconf += float(val[0][0][1])
  delt += float(val[0][1][0])
  inst += float(val[0][1][1])

print(f'avg_drop={avgdrop/len(lst)}, increase_in_confidence={incinconf/len(lst)},'
      f'deletion={delt/len(lst)}, insertion={inst/len(lst)}')
#for x in l:
#    print(x[1])
#    input()