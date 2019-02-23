import numpy as np
import random
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
		self.lr=0.4
		self.wih=np.zeros(shape=(self.hidden,self.input))
		for i in range(self.hidden):
			for j in range(self.input):
				self.wih[i][j]=random.uniform(0,1)

		self.who=np.zeros(shape=((self.output,self.hidden)))
		for i in range(self.output):
			for j in range(self.hidden):
				self.who[i][j]=random.uniform(0,1)

	def guess(self,input_array):
		self.i=np.array(input_array,ndmin=2).T
		
		self.hidden_input=np.dot(self.wih,self.i)# Multiply Input and Weights of input and hidden 
		self.hidden_output = np.ones_like(self.hidden_input)
		for i in range(self.hidden_input.shape[0]):
			self.hidden_output[i]=sigmoid(self.hidden_input[i])
		#print self.hidden_input

		#WERE DONE WITH THE INPUT HIDDEN LAYER PART

		#Now to start wtih the hidden output layer!
		self.output=np.dot(self.who,self.hidden_output)
		for i in range(self.output.shape[0]):
			self.output[i]=sigmoid(self.output[i])
		#print type(self.output[0])
		return self.output
		
	def feedforward(self,input_array,output_array):
		self.i=np.array(input_array,ndmin=2).T
		self.targets=np.array(output_array,ndmin=2).T
		self.hidden_input=np.dot(self.wih,self.i)# Multiply Input and Weights of input and hidden 
		self.hidden_output = np.ones_like(self.hidden_input)
		for i in range(self.hidden_input.shape[0]):
			self.hidden_output[i]=sigmoid(self.hidden_input[i])
		#print self.hidden_input

		#WERE DONE WITH THE INPUT HIDDEN LAYER PART

		#Now to start wtih the hidden output layer!
		self.output=np.dot(self.who,self.hidden_output)
		for i in range(self.output.shape[0]):
			self.output[i]=sigmoid(self.output[i])

		#till here
		#Beginf Feeding Bck
		self.output_error=self.targets-self.output
		self.hidden_errors=np.dot(self.who.T,self.output_error)
		self.who += self.lr * np.dot((self.output_error * self.output* (1.0-self.output)), np.transpose(self.hidden_output))

		#Now we've updated weights of the hidden-output layer! Now to propogate this backwards

		#begin fucking with the weights of the input hidden layer

		self.wih += self.lr*np.dot((self.hidden_errors*self.hidden_input*(1-self.hidden_output)),np.transpose(self.i))


			

a=NeuralNetwork(2,10,1)
for i in range(100000):
	print "Operation"+str(i)
	x=random.randint(0,1)
	y=random.randint(0,1)
	inp=[x,y]
	a.feedforward(inp,x^y)
t=0
correct=0
for i in range(10000):
	x=random.randint(0,1)
	y=random.randint(0,1)
	inp=[x,y]
	r=a.guess(inp)
	t=t+(abs((x^y)-r))
	if r<=.50 and x^y==0:
		correct=correct+1
	if r>=.50 and x^y==1:
		correct=correct+1
print "Avg:"+str(t/10000)
s=a.guess([0,0])
print s
print 0^0

s=a.guess([0,1])
print s
print 0^1

s=a.guess([1,0])
print s
print 1^0

s=a.guess([1,1])
print s
print 1^1

print "accuracy:"+str((correct/10000)*100)
