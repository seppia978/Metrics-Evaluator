import matplotlib.pyplot as plt

def one(M, m):
    return int(M > m)

def plot(X,Y,label=None,dpi=300,path='out/plot.png',y_range=(0,1),title='Graph'):
  plt.figure(dpi=dpi)
  plt.title(title)
  axes = plt.gca()
  #axes.set_xlim([xmin, xmax])
  axes.set_ylim(y_range)
  i=0
  if not type(Y)==list:
    Y=[Y]
  for y,l in zip(Y,label):
    if l is not None:
      plt.plot(X,y,label=f'{l[0]}={l[1]}')
      i+=1
      plt.legend(loc='best')
    else:
      plt.plot(X,y)
    plt.plot(X, y, '.')
  plt.xticks(rotation=90)
  plt.grid(True)
  plt.savefig(path)