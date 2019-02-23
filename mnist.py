from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from io import BytesIO
import random
import nn
import numpy as np
import json
letters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
a=nn.NeuralNetwork(784,200,62)
def listmaker(temp):
	target=[]
	for i in range(62):
		if i==letters.index(temp):
			target.append(0.99)
		else:
			target.append(0.01)
	return target

for i in range(10000):
	print "Iteration:"+str(i)
	chosen=random.choice(letters)
	img = Image.open("white.jpg")
	draw = ImageDraw.Draw(img)
	# font = ImageFont.truetype(<font-file>, <font-size>)
	font = ImageFont.truetype('Arial.ttf',300)
	# draw.text((x, y),"Sample Text",(r,g,b))
	draw.text((0,10),chosen,(0,0,0),font=font)
	img = img.resize((28,28), Image.ANTIALIAS)
	img=img.convert('1')
	#img.save('sample-out.jpg')
	data=list(img.getdata())
	data=(np.asfarray(data)/255.0*0.99)+0.01
	targets=listmaker(chosen)
	a.feedforward(data,targets)
corr=0
print "Testing"
for i in range(10000):
	print i
	chosen=random.choice(letters)
	img = Image.open("white.jpg")
	draw = ImageDraw.Draw(img)
	# font = ImageFont.truetype(<font-file>, <font-size>)
	font = ImageFont.truetype('Arial.ttf',300)
	# draw.text((x, y),"Sample Text",(r,g,b))
	draw.text((0,10),chosen,(0,0,0),font=font)
	img = img.resize((28,28), Image.ANTIALIAS)
	img=img.convert('1')
	#img.save('sample-out.jpg')
	data=list(img.getdata())
	data=(np.asfarray(data)/255.0*0.99)+0.01
	output=a.guess(data)
	d=-10000
	index=-1
	for x in range(len(output)):
		if output[x]>d:
			d=output[x]
			index=x
	#print "The Letter was "+chosen+" I guessed "+letters[index]
	#print output
	if chosen==letters[index]:
		corr=corr+1
print "The Accuracy is ="+str((corr/10000)*100)
with open('my_json.txt', 'w') as fp:
    json.dump(a, fp)

