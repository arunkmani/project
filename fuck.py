import numpy as np
import random
from matplotlib import pyplot as plt
import scipy.special
def sigmoid (x): 
	return 1/(1 + np.exp(-x)) 
class NeuralNetwork():
	def __init__(self,x,y,z):
		self.input=x
		self.hidden=y
		self.output=z
		self.lr=0.14
		self.wih=np.random.normal(0.0,pow(self.hidden,-0.5),(self.hidden,self.input))
		self.who=np.random.normal(0.0,pow(self.output,-0.5),(self.output,self.hidden))
		self.activation_function = lambda x: scipy.special.expit(x)
	def guess(self,input_array):
		self.i=np.array(input_array,ndmin=2).T
		
		self.hidden_input=np.dot(self.wih,self.i)# Multiply Input and Weights of input and hidden 
		self.hidden_output = np.ones_like(self.hidden_input)
		self.hidden_output=self.activation_function(self.hidden_input)
		#print self.hidden_input

		#WERE DONE WITH THE INPUT HIDDEN LAYER PART

		#Now to start wtih the hidden output layer!
		self.output=np.dot(self.who,self.hidden_output)
		self.output=self.activation_function(self.output)
		#print self.output
		return self.output
		
	def feedforward(self,input_array,output_array):
		self.i=np.array(input_array,ndmin=2).T
		self.targets=np.array(output_array,ndmin=2).T
		self.hidden_input=np.dot(self.wih,self.i)# Multiply Input and Weights of input and hidden 
		#self.hidden_output = np.ones_like(self.hidden_input)
		self.hidden_output=self.activation_function(self.hidden_input)
		#print self.hidden_input

		#WERE DONE WITH THE INPUT HIDDEN LAYER PART

		#Now to start wtih the hidden output layer!
		self.output=np.dot(self.who,self.hidden_output)
		self.output=self.activation_function(self.hidden_input)

		#till here
		return self.output
		#Beginf Feeding Bck
		self.output_error=self.targets-self.output
		self.hidden_errors=np.dot(self.who.T,self.output_error)
		self.who += self.lr * np.dot((self.output_error * self.output* (1.0-self.output)), np.transpose(self.hidden_output))

		#Now we've updated weights of the hidden-output layer! Now to propogate this backwards

		#begin fucking with the weights of the input hidden layer

		self.wih += self.lr*np.dot((self.hidden_errors*self.hidden_input*(1-self.hidden_output)),np.transpose(self.i))

a=NeuralNetwork(784,500,10)
data_file=open("data.csv",'r')
data_list=data_file.readlines()
data_file.close()
correct=0
for i in range(10000):
	print i
	all_values=data_list[random.randint(0,99)].split(',')
	scaled_input=(np.asfarray(all_values[1:])/255.0*0.99)+0.01
	output_nodes=10
	targets=np.zeros(output_nodes)+0.01
	targets[int(all_values[0])]=0.99
	s=a.feedforward(scaled_input,targets)
	if all_values[0]==list(s).index(np.max(s)):
		correct=correct+1
print "correct"+ str(correct)

test_data_file=open("test.csv",'r')
test_data_list=test_data_file.readlines()
test_data_file.close()

values=test_data_list[0].split(',')
scaled_test_input=(np.asfarray(values[1:])/255.0*0.99)+0.01
s=a.guess(scaled_test_input)
print values[0]
print s 
