import os

with open('out/output.txt','r') as f:
    txt=f.read()

tx=[t[1:] for t in txt.split('}')]
params=[(t.split(':')[0],t.split(':')[1]) for t in tx]
print(params)