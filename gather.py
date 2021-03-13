import ast
import argparse
import sys
import pandas as pd

class metric():
    def __init__(self,name,vals=[]):
        self.name,self.vals=name,vals

class AttMethod():
    def __init__(self,name,count=0,metrics=[]):
        self.name,self.count,self.metrics=name,count,metrics

    def get_names(self):
        return [m.name for m in self.metrics]

    def update_metrics(self,val):
        i=0
        for v in val:
            for idx,m in enumerate(v):
                self.metrics[idx+i].vals.append(m)
            i+=idx



if __name__== '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-ip',"--in_path", type=str, help='Input path',required=True)
    parser.add_argument('-m', "--metrics", type=str, help='Metrics to evaluate', required=True, nargs='+')
    parser.add_argument('-rp',"--out_path", type=str, help='Output path')

    args = parser.parse_args()

    in_path=args.in_path
    metric_names = [el.lower() for el in args.metrics]
    if args.out_path is not None:
        out_path=args.out_path

    if in_path[-1] is not '/':
        in_path=in_path+'/'
    in_path+='output.txt'
    with open(in_path,'r') as f:
        txt=f.read().split('\n')

    lst=[]
    for row in txt[:-1]:
        #print(row)
        lst.append(ast.literal_eval(row))

    l=[]
    #lst=lst[4:]
    #visu=2
    #lst=lst[int(len(lst)/2):] if visu == 2 else lst[:int(len(lst)/2)]
    #print(lst)

    CAMS={}
    for i,d in enumerate(lst):
        CAMS[list(d.keys())[0][1]]=AttMethod(list(d.keys())[0][1],metrics=[metric(m) for m in metric_names])

    metrics_dict={m:[0 for _ in CAMS] for m in metric_names}

    for i,d in enumerate(lst):
        print(d)
        val=list(d.values())[0]
        cam=list(d.keys())[0][1]
        CAMS[cam].count+=1
        CAMS[cam].update_metrics(val)
        #l.append([float(val[0][0][0]),float(val[0][0][1]),float(val[0][0][2])])
        # et[list(CAMS.keys()).index(cam)] += float(val[0][0][5])
        # avgdrop[list(CAMS.keys()).index(cam)] += float(val[0][0][0])
        # incinconf[list(CAMS.keys()).index(cam)] += float(val[0][0][1])
        # comp[list(CAMS.keys()).index(cam)] += float(val[0][0][2])
        # coh[list(CAMS.keys()).index(cam)] += float(val[0][0][3])
        # asv[list(CAMS.keys()).index(cam)] += float(val[0][0][4])
        #inst[list(CAMS.keys()).index(cam)] += float(val[0][1][1])

    for k,v in CAMS.items():
        print(v.name)
        for i in v.metrics:
            print(i.name, i.vals)
    headers=['Elapsed Time','Average Drop','Increase In Confidence','Complexity','Coherency','Average Score Variance']

    #vals=[et,avgdrop,incinconf,comp,coh,asv]
    #df_dict={k:v for k,v in zip(headers,vals)}
    print('\n--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('\n\t\t\t|  Elapsed Time\t\t|  Average Drop\t\t|  Increase In Confidence\t| Complexity\t\t|  Coherency\t\t|  F1 score\t\t|  Average Score Variance')
    for c in CAMS:
        key=c
        if len(key)<5:
            key+='\t\t'
        elif len(key)<12:
            key+='\t'
        index_cam=list(CAMS.keys()).index(c)
        #print(CAMS[c],comp[index_cam])
        #f1_score=3/(1/(100-round(avgdrop[index_cam]/CAMS[c],2))+1/(100-round(comp[index_cam]/CAMS[c],3)+10e-8)+1/(round(coh[index_cam]/CAMS[c],3)))

        # print(f'''
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n
        # {key}\t|  {round(et[index_cam]/CAMS[c],2):.2f}s\t\t|  {round(avgdrop[index_cam]/CAMS[c],2):.2f}%\t\t|  {round(incinconf[index_cam]/CAMS[c],2):.2f}%\t\t\t| {round(comp[index_cam]/CAMS[c],3):.3f}%\t\t|  {round(coh[index_cam]/CAMS[c],3):.3f}%\t\t|  {f1_score:.3f}%\t\t|  {round(asv[index_cam]/CAMS[c],3):.3f}%''')
    print('\n--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    #for x in l:
    #    print(x[1])
    #    input()
