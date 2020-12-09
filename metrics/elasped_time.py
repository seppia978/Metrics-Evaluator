from MetricEvaluator.evaluate_metrics import MetricOnSaliencyExtractor as MOSE
import time

class ElapsedTime(MOSE):
    def __init__(self,name,result=0.,nimgs=0.):
        super().__init__(name,result)
        self.nimgs=nimgs
        self.step=0

    def update(self,*args):
        if self.step % 2 == 0:
            self.now=time.time()
        else:
            self.result+=(time.time()-self.now)
        self.step += 1

    def final_step(self,**kwargs):
        if self.nimgs:
            if self.nimgs<0:
                raise ValueError('nimgs must be strictly positive ')
            try:
                self.result /= self.nimgs
            except:
                raise ZeroDivisionError('nimgs must not be zero')

    def clear(self):
        super().clear()
        self.step,self.now=0,0

    def print(self):
        if  self.nimgs==0:
            print(f'Elapsed Time: {self.result}')
        else:
            print(f'Elapsed Time: {self.result}%')