import numpy as np
import random
import scipy.special
def transpose(x):
    temp=x.shape
    swapped=np.zeros(shape=(temp[1],temp[0]))
    for i in range(temp[0]):
        for j in range(temp[1]):
            swapped[j][i]=x[i][j]
    return swapped
def sigmoid (x): 
    return 1/(1 + np.exp(-x)) 


class NeuralNetwork():
    def __init__(self,x,y,z):
        self.input=x
        self.hidden=y
        self.output=z
        self.lr=0.1
        self.wih=np.random.normal(0.0,pow(self.input,-0.5),(self.hidden,self.input))
        self.who=np.random.normal(0.0,pow(self.hidden,-0.5),(self.output,self.hidden))
        self.activation_function = lambda x: scipy.special.expit(x)
    def guess(self,input_array):
        i=np.array(input_array,ndmin=2).T
        
        hidden_input=np.dot(self.wih,i)# Multiply Input and Weights of input and hidden 
        #self.hidden_output = np.ones_like(self.hidden_input)
        '''for i in range(self.hidden_input.shape[0]):
            self.hidden_output[i]=sigmoid(self.hidden_input[i])'''
        hidden_output = self.activation_function(hidden_input)

        #print self.hidden_input

        #WERE DONE WITH THE INPUT HIDDEN LAYER PART

        #Now to start wtih the hidden output layer!
        output=np.dot(self.who,hidden_output)
        '''for i in range(self.output.shape[0]):
            self.output[i]=sigmoid(self.output[i])'''
        output = self.activation_function(output)
        #print self.output
        return output
        
    def feedforward(self,input_array,output_array):
        i=np.array(input_array,ndmin=2).T
        targets=np.array(output_array,ndmin=2).T
        hidden_input=np.dot(self.wih,i)# Multiply Input and Weights of input and hidden 
        #self.hidden_output = np.ones_like(self.hidden_input)
        hidden_output = self.activation_function(hidden_input)
        #print self.hidden_input

        #WERE DONE WITH THE INPUT HIDDEN LAYER PART

        #Now to start wtih the hidden output layer!
        output=np.dot(self.who,hidden_output)
        output = self.activation_function(output)

        #till here
        #Beginf Feeding Bck
        output_error=targets-output
        hidden_errors=np.dot(self.who.T,output_error)
        self.who += self.lr * np.dot((output_error * output* (1.0-output)), np.transpose(hidden_output))

        #Now we've updated weights of the hidden-output layer! Now to propogate this backwards

        #begin fucking with the weights of the input hidden layer

        self.wih += self.lr*np.dot((hidden_errors*hidden_input*(1-hidden_output)),np.transpose(i))



			

a=NeuralNetwork(2,10,1)
for i in range(10000):
	print "Operation"+str(i)
	x=random.randint(0,1)
	y=random.randint(0,1)
	inp=[x,y]
	a.feedforward(inp,not(x and y))
t=0
correct=0
for i in range(10000):
	x=random.randint(0,1)
	y=random.randint(0,1)
	inp=[x,y]
	r=a.guess(inp)
	t=t+(abs((x*y)-r))
	'''if r<=.50 and x^y==0:
		correct=correct+1
	if r>=.50 and x^y==1:
		correct=correct+1'''
print "Avg:"+str(t/10000)
s=a.guess([0,0])
print s
print not(0 and 0)

s=a.guess([0,1])
print s
print not (0 and 1)

s=a.guess([1,0])
print s
print not(1 and 0)

s=a.guess([1,1])
print s
print not(1 and 1)

print "accuracy:"+str((correct/10000)*100)
