import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import fight
import os

MODEL_NAME = 'Squeezenet'
root = '/home/malar/prject_vision/'
path = os.path.join(root, 'dataset/test/')
print(path)
test_dataset = fight(path)
test_loader = DataLoader(test_dataset, batch_size=1)
classifier = torch.load('models/{}.pt'.format(MODEL_NAME)).eval()
correct = 0

print(len(test_dataset))
for step, data in enumerate(test_loader, 0):
    test_x,test_y = data
    test_x = test_x.cuda()
    pred = classifier.forward(test_x) 
    #print(len(test_x),len(test_y))
    pred = pred.cpu()
    #y_hat = np.argmax(pred.data)
    y_hat =pred.argmax()
    y_hat = y_hat.item()
    y =int(test_y[0])
    print(y_hat)
    print(y)
    if y_hat == y:
        correct+= 1
     # print("correct")
print(correct)
print("Accuracy={}".format(correct*100 / len(test_dataset)))
