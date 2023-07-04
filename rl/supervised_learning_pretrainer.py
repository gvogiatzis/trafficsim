import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
import tqdm

from typing import Callable


class SupervisedLearningPretrainer:
    def __init__(self, dataset, model, loss_fn = torch.nn.CrossEntropyLoss(), batch_size=64, learning_rate=0.0001,training_test_split = 0.8, use_gpu=False):
        dev = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.full_dataset = dataset
        self.model = model
        self.training_test_split = 0.8
        train_dataset, test_dataset = random_split(dataset, [training_test_split, 1-training_test_split])

        self.train_loader = DataLoader(train_dataset, batch_size,)
        self.test_loader = DataLoader(test_dataset, batch_size)

        # self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    
    def train_epochs(self, num_epochs):
        from rich import get_console
        print = get_console().print

        stats=dict()
        stats["training_loss_series"]=[]
        stats["test_loss_series"]=[]
        stats["training_acc_series"]=[]
        stats["test_acc_series"]=[]
        loss=torch.nn.CrossEntropyLoss()

        self.model.to(self.device)
        for e in range(num_epochs):
            print(f"Epoch [yellow]{e+1}/{num_epochs}[/] Training: ",end="", highlight=False)
            training_loss = 0.0
            training_acc = 0.0
            num_training_samples=0.0
            self.model.train()
            for inputs, labels in tqdm.tqdm(self.train_loader):
            # for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                r = loss(outputs, labels)
                r.backward()
                self.optimizer.step()

                # collect statistics
                training_loss += r.item() * len(inputs)
                _,output_preds = outputs.max(dim=1)
                training_acc += sum(output_preds==labels).item()
                num_training_samples += len(inputs)

            training_loss /= num_training_samples
            training_acc /= num_training_samples
            print(f"Avg Training loss: [blue]{training_loss:0.4f}[/] Avg Training Acc: [green]{100*training_acc:0.1f}[/] Testing:", end="", highlight=False)

            self.model.eval()
            with torch.no_grad():
                num_test_samples=0
                test_loss = 0.0
                test_acc = 0.0
                # for inputs, labels in tqdm.tqdm(self.test_loader):
                for inputs, labels in self.test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)                    
                    outputs = self.model(inputs)
                    _,output_preds = outputs.max(dim=1)
                    test_acc += sum(output_preds==labels).item()
                    test_loss += loss(outputs, labels).item() * len(inputs)
                    num_test_samples += len(inputs)
 
            test_loss /= num_test_samples
            test_acc /= num_test_samples

            print(f"Avg Test loss: [blue]{test_loss:0.4f}[/] Avg Test Acc: [green]{100*test_acc:0.1f}[/]", highlight=False)

            stats["training_loss_series"].append(training_loss)
            stats["test_loss_series"].append(test_loss)
            stats["training_acc_series"].append(training_acc)
            stats["test_acc_series"].append(test_acc)
        self.model.cpu()
        return stats
            

if __name__ == "__main__":
    from models import MLPnet, loadModel, saveModel, loadModel_from_dict
    import matplotlib.pyplot as plt

    dataset = []
    input_dim = 127
    output_dim = 8
    W = np.random.randn(input_dim,output_dim)
    print("Generating dataset:")
    for i in tqdm.tqdm(range(1000000)):
        x = 50*torch.randn(size=(input_dim,), dtype=torch.float32)

        a = np.argsort(x @ W)
        # t = a[(len(a)-1)//2]
        t = a[0]
        # t = np.random.randint(output_dim)
        dataset.append((x, t))
    # model = MLPnet(input_dim,256,512,256,output_dim)
    model = MLPnet(input_dim,1028,1028,output_dim)
    trainer = SupervisedLearningPretrainer(dataset, model)

    stats = trainer.train_epochs(100)
    plt.figure()
    plt.title("Loss")
    plt.plot(stats["training_loss_series"], 'r-')
    plt.plot(stats["test_loss_series"], 'b-')
    plt.figure()
    plt.title("Accuracy")
    plt.plot(stats["training_acc_series"], 'r-')
    plt.plot(stats["test_acc_series"], 'b-')
    plt.show()