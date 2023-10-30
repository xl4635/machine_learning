import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator, FixedLocator
#局部加权核心算法
# testPoint:需要预测的点   xArr:训练集输入  yArr:训练集输出
def lwlr_old(testPoint,xArr,yArr,k=1.0):
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    m=np.shape(xMat)[0]
    weights=np.mat(np.eye((m)))
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        #print(diffMat*diffMat.T)
        weights[j,j]=np.exp(diffMat*diffMat.T/(-2.0*k**2))
    #print(weights)
    #print(np.matmul(weights,xMat))      
    xTx=np.matmul(xMat.T,(np.matmul(weights,xMat)))
    if np.linalg.det(xTx)==0.0:
        print("行列式为0，奇异矩阵，不能做逆")
        return 10.4
    ws=xTx.I*(xMat.T*(weights*yMat))
    #print(ws)
    return testPoint*ws
def lwlr(testPoint,xArr,yArr,k=1.0,ridge=0.01):
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    m, n = xMat.shape
    weights=np.mat(np.eye((m)))
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        #print(diffMat*diffMat.T)
        weights[j,j]=np.exp(diffMat*diffMat.T/(-2.0*k**2))
    ''' 改动1： 线性回归需要一个常数项b，直接添加值恒为1的额外维度 '''
    xMat = np.concatenate((np.ones((m, 1)), xMat), axis=1)
    #print(weights)
    #print(np.matmul(weights,xMat))
    ''' 改动2： 引入正则项，以排除输入维度间的线性相关性 '''
    xTx = xMat.T * weights * xMat + ridge * np.eye(n + 1)
    ''' 改动3（重要）： 输入线性相关导致det(xTx)很小，但不会为零，继续计算会导致结果误差大 '''
    if ridge == 0 and np.linalg.det(xTx) < 1e-6:
        print("行列式为0，奇异矩阵，不能做逆")
        return 10.4
    ws=xTx.I*(xMat.T*(weights*yMat))
    #print(ws)
    return testPoint * ws[1: ] + ws[0]
#  testArr:预测合集
def lwlrTest(testArr,xArr,yArr,k=0.5):
    m=np.shape(testArr)[0]
    yHat=np.zeros(m)
    #print(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat
#用以随机选择训练集以及测试集函数
# n:数据总个数  num:测试集个数
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
#求均方方差
def mean_squared_error(y_true,y_pred):
    #输入: y_true 样本真实值
    #      y_pred 样本预测值
    #输出: error  误差值
    assert len(y_true)==len(y_pred)    
    error=np.mean(np.square(y_true-y_pred))
    return error
#规格化
def normalize(x):
    x_0 = np.min(x, axis=0)
    return (x - x_0) / (np.max(x, axis=0) - x_0)
data=pd.read_table('./endresult.txt',sep=' ',header=0)
x=data.values
''' 改动4： 数据规格化，排除不同维度的尺度差异，不过对于这个数据集影响不大 '''
x[: , : 4] = normalize(x[: , : 4])
#print(lwlr(x[0,0:3],x[:,0:3],x[:,8],0.2)); assert False, "调试完成"
#随机选择20个测试数据
testx,trainx=randchos(x,121,20)
X=trainx[:,0:3]
Y=trainx[:,8]
X_test=testx[:,0:3]
y_test=testx[:,8]
#y_test_pred为预测值
y_test_pred=lwlrTest(X_test,X,Y,0.1)
print(y_test_pred)
print(y_test)
print("损失:",mean_squared_error(y_test,y_test_pred))
'''
a=np.array([[1,2],[2,3]])
b=np.array([[1,2],[2,3]])
print(a*b)
print(np.matmul(a,b))
'''

myplt=plt.figure()
base=np.arange(1,21,1)
#plt.scatter(base,y_test,color="blue",label="真实值")
plt.plot(base,y_test,color="blue",label="真实值")
plt.plot(base,y_test_pred,color="red",label="预测值")
#plt.scatter(base,y_test_pred,color="red",label="预测值")
plt.legend(fontsize='x-large')
plt.xticks(base)
plt.xlabel("数据${X_n}$")
plt.ylabel("极限屈服应力/弹性模量*100")
plt.show()

ymax=0
num=0
tp=pd.read_table('./mypreds.txt',sep=' ',header=0)
tps=tp.values
x_pre=tps[:,0:3]
y_pre=lwlrTest(X_test,X,Y,0.1)
print(y_pre)
for ea in y_pre:
    if(ea>ymax):
        xmax=x_pre[num,0:3]
        ymax=ea
    num+=1
print(num)
print(ymax)
print(xmax)
