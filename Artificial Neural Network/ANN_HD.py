import numpy as np
import pandas as pd


data = pd.read_csv("Heart_Disease_Prediction.csv")

inp = data.drop(["Heart Disease"],axis=1)
output = data["Heart Disease"]

inp["Age"]= inp["Age"].apply(lambda x : (x/1000))
inp["Sex"]= inp["Sex"].apply(lambda x : (x/10))
inp["Chest pain type"]= inp["Chest pain type"].apply(lambda x : (x/10))
inp["BP"]= inp["BP"].apply(lambda x : (x/200))
inp["Cholesterol"]= inp["Cholesterol"].apply(lambda x : (x/500))
inp["Max HR"]= inp["Max HR"].apply(lambda x : x/200)
inp["ST depression"]= inp["ST depression"].apply(lambda x : x-2)
inp["Slope of ST"]= inp["Slope of ST"].apply(lambda x : x-2)
inp["Number of vessels fluro"]= inp["Number of vessels fluro"].apply(lambda x : x-3)
inp["Thallium"]= inp["Thallium"].apply(lambda x : x-10)

rows = len(inp.axes[0]) 
cols = len(inp.axes[1])


inp = inp.values
output = output.values

output = np.array(output)
output = output.reshape(rows,1)
np.random.seed(0)
weight1 = np.random.rand(cols,cols)
weight2 = np.random.rand(cols,cols)
weight3 = np.random.rand(cols)
bias1 = np.zeros(cols)
bias2 = np.zeros(cols)
bias3 = 0.0


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

    
def forward(x,w,b):
    return sigmoid(np.dot(np.transpose(x),w)+b)




def train():
    h1 =[]   
    for i in range(rows):
        temp=[]
        for j in range(cols):
            temp.append(forward(inp[i],weight1[j],bias1[j]))
        h1.append(temp)

    h1=np.array(h1)
    h2 =[]   
    for i in range(rows):
        temp2=[]
        for j in range(cols):
            temp2.append(forward(h1[i],weight2[j],bias2[j]))
        h2.append(temp2)

    h2=np.array(h2)

    o=[]   
    for i in range(rows):
        o.append(forward(h2[i],weight3,bias3))


    def feeddforward(h1,h2,o):
        global weight1,weight2,bias1,bias2 , weight3, bias3
        for i in range(rows):
            for j in range(cols):
                h1[i][j]=(forward(inp[i],weight1[j],bias1[j]))

        for i in range(rows):
            for j in range(cols):
                h2[i][j]=(forward(h1[i],weight2[j],bias2[j]))

        for i in range(rows):
            o[i]=forward(h2[i],weight3,bias3)

        o = np.array(o)
        o = o.reshape(rows,1)

        mse = ((o - output)**2).sum()/rows
        
        output_error = o - output
        output_delta = output_error * o * (1 - o)

    
        weight3=weight3 - (0.01*np.dot(h2.T,output_delta)).T
        bias3 = bias3 - (0.01 * (output_delta.sum() ))
    

        hiddenlayer_error2 = np.dot(output_delta , weight3)
        hiddenlayer_delta2 = hiddenlayer_error2 * h2 * (1 - h2)

        weight2 = weight2 - (0.01 * np.dot(h1.T , hiddenlayer_delta2))
        bias2 = bias2 - (0.01 * np.sum(hiddenlayer_delta2.T, axis = 1 ) )

        hiddenlayer_error1 = np.dot(hiddenlayer_delta2 , weight2)
        hiddenlayer_delta1 = hiddenlayer_error1 * h1 * (1 - h1)
    
        weight1 = weight1 - (0.01 * np.dot(inp.T , hiddenlayer_delta1))
        bias1 = bias1 - (0.01 * np.sum(hiddenlayer_delta1.T, axis = 1 ) )
        weight3 = weight3.reshape(cols)
        return mse
    
  
    print("Training on process. Please wait.")
    for i in range(1000):
       mse = feeddforward(h1,h2,o)

    print("Accuracy from Training:",(1-mse)*100)
    fweight = np.concatenate([weight1.flatten(),weight2.flatten(),weight3])

    fbias = np.concatenate([bias1,bias2])

    fbias = np.append(fbias,np.array(bias3))
    fbias = np.append(fbias,np.array(mse))
    np.savetxt(r'D:\AI\Project\ weight.txt', fweight)
    np.savetxt(r'D:\AI\Project\ bias.txt', fbias)



def test():
    w = np.loadtxt(r'D:\AI\Project\ weight.txt')

    w1 = w[0:169]
    w2 = w[169 : 338]
    w3 = w[338 : len(w)]
    w1 = w1.reshape(13,13)
    w2 = w2.reshape(13,13)

    b = np.loadtxt(r'D:\AI\Project\ bias.txt')
    b1 = b[0:13]
    b2 = b[13 : 26]
    b3 = b[len(b)-2]
    mse = b[len(b)-1]

    b = np.loadtxt(r"D:\AI\Project\test_data.csv", delimiter =",")
    a = b[:,0:13]
    c = b[:,13]
    rows,cols = a.shape
    result = []
    for i in range(rows):
        x = a[i]

        x[0] = x[0]/1000
        x[1] = x[1]/10
        x[2] = x[2]/10
        x[3] = x[3]/200
        x[4] = x[4]/500
        x[7] = x[7]/200
        x[9] = x[9]-2
        x[10] = x[10]-2
        x[11] = x[11]-3
        x[12] = x[12]-10

        result_h1 = [0]*13
        for i in range(13):
            result_h1[i]=forward(x,w1[i],b1[i])
        result_h1 = np.array(result_h1)

        result_h2 = [0]*13
        for i in range(13):
            result_h2[i]=forward(result_h1,w2[i],b2[i])
        result_h2 = np.array(result_h2)    
        result.append(forward(result_h2,w3,b3))
    print("Accuracy from Testing: ",(1-((result - c)**2).sum()/rows)*100)
    print("Accuracy from Training:",(1-mse)*100)
    


i=0
while i==0:
    print("1. Train.")
    print("2. Test.")
    print("3. Exit.")
    choice = int(input("Enter Choice: "))
    if(choice==1):
        train()
    if(choice==2):
        test()
    if(choice==3):
        print("Thank You")
        break



