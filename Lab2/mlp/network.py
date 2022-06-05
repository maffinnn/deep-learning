""" 定义网络 """

class Network():
	def __init__(self):
		self.layerList = []
		self.numLayer = 0

	def add(self, layer):
		self.numLayer += 1
		self.layerList.append(layer)

	def forward(self, x):
		for i in range(self.numLayer):
			x = self.layerList[i].forward(x)
		return x

	def backward(self, delta):
		for i in reversed(range(self.numLayer)): 
			delta = self.layerList[i].backward(delta)
