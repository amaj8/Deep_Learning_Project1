import torch
import sys
from torch import nn
import torch.nn.functional as F
#from dl_assignment.py import Network

print("Aalo Majumdar\nM.Tech R, CSA, IISc\nSR#: 16116")

testfile = sys.argv[2]
input_file = open(testfile,"r")
output_file1 = open("Software1.txt","w")
output_file2 = open("Software2.txt","w")

PATH = "model/network.pth"

NUM_DIGITS = 16

def decToBin(n,num_digits):
    return [n >> i & 1 for i in range(num_digits)]


def fizzBuzzOutput(n):
	if n%15 == 0:     return 3

	elif n%5 == 0:    return 2
	elif n%3 == 0:    return 1
   
	else:             return 0

NUM_HIDDEN_UNITS = 150

class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden = nn.Linear(NUM_DIGITS,NUM_HIDDEN_UNITS)
    self.output = nn.Linear(NUM_HIDDEN_UNITS,4)
    
  def forward(self,x):
    x = F.relu(self.hidden(x))
    x = self.output(x)
    return x

#LEARNING_RATE = 0.01
#SGD optimizer + negative log likelihood -> cross entropy loss
#optimizer = torch.optim.SGD(net.parameters(), lr = LEARNING_RATE, momentum=0.09)
#loss_fn = nn.CrossEntropyLoss()

net = torch.load(PATH)
net.eval()

for line in input_file:
  line = int(line.split()[0])
  output_file1.write([str(line),"fizz","buzz","fizzbuzz"][fizzBuzzOutput(line)] + "\n")
  bin_line = decToBin(line,NUM_DIGITS)
  # line = int(line)
  testX = torch.Tensor([bin_line])
  net_output = net(testX)
  _,predicted = torch.max(net_output.data,1)
  prediction = predicted[0]
  output_file2.write([str(line),"fizz","buzz","fizzbuzz"][prediction] + "\n")
  #print([str(line),"fizz","buzz","fizzbuzz"][prediction])
