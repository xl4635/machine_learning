import numpy as np
import pandas as pd
#用来计算y=1的概率
def prob(mtrA,mtrB):
    ebx=exp(np.dot(mtrA,mtrB))
    return ebx/(1+ebx)
#随机选择训练集或者测试集
def randchos(data,n,num):
    mysub=np.arange(1,n+1,1)
    sub=np.random.choice(mysub,num,replace=False)
    data=np.array(data)
    train=[]
    test=[]
    for i in range(1,n+1):
        if(i in sub):
            test.append(data[i-1,:])
        else:
            train.append(data[i-1,:])
    test=np.array(test)
    train=np.array(train)
    return test,train
#用于对模型的操作
class Runner(object):
    def __init__(self,model,optimizer,loss_fn,metric):
        self.model=model            #模型
        self.optimizer=optimizer    #优化器
        self.loss_fn=loss_fn        #损失函数
        self.metric=metric          #评估指标
    #模型训练
    def train(self,train_dataset,dev_dataset=None,**kwargs):
        pass
    #模型评价
    def evaluate(self,data_set,**kwargs):
        pass
    #模型预测
    def predict(self,x,**kwargs):
        pass
    #模型保存
    def save_model(self,x,**kwargs):
        pass
    #模型加载
    def load_model(self,model_path):
        pass
#对率回归模型
class Ex():
    #input_size:特征向量长度
    def __init__(self,input_size):
        self.input_size=input_size
        self.params={}
        self.params['beita']=np.random.rand(input_size+1)
        self.params['beita'][input_size]=1
    #将模型用于预测:
    def __call__(self,X):
        X=normalize(X)
        return np.matmul(X,self.params['beita'].T)
#规格化
def normalize(x):
    x_0 = np.min(x, axis=0)
    return (x - x_0) / (np.max(x, axis=0) - x_0)
#优化器:牛顿法优化器  step:学习率 n:迭代轮数
def newton(beita,datas,step,n):
    for i in range(n):
        delta1=datas.shape[1]-1
        for exam in datas:
            
#用来将“好瓜”一栏改写成1/0更便于计算的方式，1为好瓜，0为坏瓜
def cgtoeasy(datas):
    for mem in datas:
        if mem[2]=="是":
            mem[2]=1
        else:
            mem[2]=0
if __name__=="__main__":
    data=pd.read_excel('watermalon.xlsx')
    print(data)
    test,train=randchos(data,17,4)
    cgtoeasy(test)
    cgtoeasy(train)
    print(test,train)
    #分xy训练集和测试集
    xtest=test[:,0:2]
    ytest=test[:,2]
    xtrain=train[:,0:2]
    ytrain=train[:,2]
    print(xtest,ytest)
    xtest=normalize(xtest)
    print(xtest)
    
