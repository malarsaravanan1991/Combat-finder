## Training the model and updating the weights


from model import SqueezeNet, Fire
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import fight
import matplotlib.pyplot as plt

root = '/home/malar/prject_vision/'

# Instantiating the fight dataset class we created
train_dataset = fight(os.path.join(root, 'dataset/train'))
print("Loaded data")

# Creating a dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=256,pin_memory=True)
print(len(train_dataset))

net = SqueezeNet(num_classes=8).cuda()
#print(net.is_cuda)
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(net.parameters())

loss_history = []


def train(epoch):
    	epoch_loss = 0
	n_batches = len(train_dataset) // 256
        print("n_batch {}".format(n_batches)) 
	for step, data in enumerate(train_loader, 0):
		train_x, train_y = data
		train_x = train_x.cuda()
		y_hat = net.forward(train_x)
		train_y = torch.Tensor(np.array(train_y,dtype='int32'))
		train_y = train_y.cuda()
		# CrossEntropyLoss requires arg2 to be torch.LongTensor
		loss = criterion(y_hat, train_y.long())
		epoch_loss += loss.item()
		optimizer.zero_grad()
                #print("Epoch {}, loss {} step{}".format(epoch, epoch_loss,step))
                  
		# Backpropagation
		loss.backward()
		optimizer.step()
		# There are len(dataset)/BATCH_SIZE batches.
		# We print the epoch loss when we reach the last batch.
		if step % n_batches == 0 and step != 0:
			epoch_loss = epoch_loss / n_batches
			loss_history.append(epoch_loss)
			print("Epoch {}, loss {}".format(epoch, epoch_loss))
			epoch_loss = 0


for epoch in range(1, 50 + 1):
	train(epoch)

# Saving the model
torch.save(net, 'models/{}.pt'.format(SqueezeNet))
print("Saved model...")

# Plotting loss vs number of epochs
#plt.plot(np.array(range(1, 50 + 1)), loss_history)
#plt.xlabel('Iterations')
#plt.ylabel('Loss')
#plt.show()

