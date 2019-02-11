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
		self.lr=0.4
		self.wih=np.random.normal(0.0,pow(self.hidden,-0.5),(self.hidden,self.input))
		self.who=np.random.normal(0.0,pow(self.output,-0.5),(self.output,self.hidden))
		self.activation_function = lambda x: scipy.special.expit(x)
	def guess(self,input_array):
		self.i=np.array(input_array,ndmin=2).T
		
		self.hidden_input=np.dot(self.wih,self.i)# Multiply Input and Weights of input and hidden 
		self.hidden_output = np.ones_like(self.hidden_input)
		'''for i in range(self.hidden_input.shape[0]):
			self.hidden_output[i]=sigmoid(self.hidden_input[i])'''
		self.hidden_output = self.activation_function(self.hidden_input)

		#print self.hidden_input

		#WERE DONE WITH THE INPUT HIDDEN LAYER PART

		#Now to start wtih the hidden output layer!
		self.output=np.dot(self.who,self.hidden_output)
		'''for i in range(self.output.shape[0]):
			self.output[i]=sigmoid(self.output[i])'''
		self.output = self.activation_function(self.output)
		#print self.output
		return self.output
		
	def feedforward(self,input_array,output_array):
		self.i=np.array(input_array,ndmin=2).T
		self.targets=np.array(output_array,ndmin=2).T
		self.hidden_input=np.dot(self.wih,self.i)# Multiply Input and Weights of input and hidden 
		self.hidden_output = np.ones_like(self.hidden_input)
		self.hidden_output = self.activation_function(self.hidden_input)
		#print self.hidden_input

		#WERE DONE WITH THE INPUT HIDDEN LAYER PART

		#Now to start wtih the hidden output layer!
		self.output=np.dot(self.who,self.hidden_output)
		self.output = self.activation_function(self.output)

		#till here
		#Beginf Feeding Bck
		self.output_error=self.targets-self.output
		self.hidden_errors=np.dot(self.who.T,self.output_error)
		self.who += self.lr * np.dot((self.output_error * self.output* (1.0-self.output)), np.transpose(self.hidden_output))

		#Now we've updated weights of the hidden-output layer! Now to propogate this backwards

		#begin fucking with the weights of the input hidden layer

		self.wih += self.lr*np.dot((self.hidden_errors*self.hidden_input*(1-self.hidden_output)),np.transpose(self.i))
