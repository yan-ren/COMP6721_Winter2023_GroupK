import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import image
from matplotlib import pyplot
import time
import random

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(1)
np.random.seed(1)

def load_data(path, test_split, val_split, batch_size, input_size):
    
    ######## Write your code here ########
    
    transform_dict = {
    'src': transforms.Compose(
    [transforms.Resize(input_size),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.554, 0.450, 0.343],
                          std=[0.231, 0.241, 0.241]),
     ])}

    data = datasets.ImageFolder(root=path, transform=transform_dict['src'])

    dataset_size = len(data)
    print(dataset_size)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - (test_size + val_size)

    train_dataset, test_dataset, val_dataset = td.random_split(data,
                                               [train_size, test_size, val_size])

    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    data_loader_test  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    data_loader_val   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    
    #########################################
    
    return data_loader_train, data_loader_test, data_loader_val



path = "C:\\Users\\Administrator\\Desktop\\Dataset 10"
train_loader, test_loader, val_loader = load_data(path, 0.1, 0.1, 32, (256, 256))


from torchvision import models
modelAlexNet3 = models.alexnet(weights=None)
modelAlexNet3.classifier[6] = nn.Linear(4096,10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(modelAlexNet3.parameters(), lr=0.001, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5, gamma=0.1)

print(modelAlexNet3)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))
modelAlexNet3.to(device);         

AccuracyList=[]
AccuracyList2=[]
AccuracyListV=[]
AccuracyList2V=[]
###### Define and run your training loop here #########
num_epochs = 15
total_steps = len(train_loader)
t1 = time.time()
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        images, labels = data[0].to(device), data[1].to(device)
        # Forward pass
        outputs = modelAlexNet3(images)
        loss = criterion(outputs, labels)
        # Backprop and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Train accuracy
        total = labels.size(0)
        _,predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        AccuracyList.append((correct / total) * 100)
        if (i + 1) % 10 == 0:
            AccuracyList2.append((correct / total) * 100)
            correct_v = 0
            total_v = 0
            for dataVal in val_loader:
                images_v, labels_v = dataVal[0].to(device), dataVal[1].to(device)
                outputs = modelAlexNet3(images_v)
                _, predicted = torch.max(outputs.data, 1)
                correct_v += (predicted == labels_v).sum().item()
                total_v += labels_v.size(0)
                AccuracyListV.append((correct_v / total_v) * 100)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, Validation Accuracy: {:.2f}%'
                .format(epoch + 1, num_epochs, i + 1, total_steps, loss.item(),
                    (correct / total) * 100, (correct_v / total_v) * 100))
            AccuracyList2V.append((correct_v / total_v) * 100)
print("######## Training Finished in {} seconds ###########".format(time.time()-t1))   



modelAlexNet3.eval() 
predicted_labels = []
true_labels = []

y_scores = []
y_true = []

with torch.no_grad(): 
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = modelAlexNet3(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted_labels += predicted.cpu().numpy().tolist()
        true_labels += labels.cpu().numpy().tolist()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        probs = F.softmax(outputs, dim=1)
        y_scores += probs.cpu().numpy().tolist()
        y_true += labels.cpu().numpy().tolist()
    print('Test Accuracy of the model on the {} test images: {} %'
        .format(total, (correct / total) * 100))
    
    
from sklearn.metrics import confusion_matrix, classification_report,roc_curve, auc
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion matrix:")
print(cm)

# Calculate precision, recall, f1-score, and support for each class
report = classification_report(true_labels, predicted_labels,zero_division=1)
print("Classification report:")
print(report)    

#torch.save(modelAlexNet3, "C:\\Users\\Administrator\\Desktop\\ModelsAI\\Dataset-3\\ALEX10SDGModelTA68TT53.pt")



import matplotlib.pyplot as plt            
plt.plot( AccuracyList2)
plt.plot( AccuracyList2V)
plt.rcParams["figure.figsize"]=(20,10)
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.legend(["Training","Validation"])
plt.title('Training Accuracy per Batch')
plt.show()  


plt.plot( AccuracyList)
plt.plot( AccuracyListV)
plt.rcParams["figure.figsize"]=(20,10)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend(["Training","Validation"])
plt.title('Training Accuracy per Iteration')
plt.show()  




from sklearn.manifold import TSNE

# Get embeddings for the training data
embeddings = []
labels = []
with torch.no_grad():
    for data in train_loader:
        images, targets = data[0].to(device), data[1].to(device)
        outputs = modelAlexNet3(images)
        embeddings.append(outputs.cpu().numpy())
        labels.append(targets.cpu().numpy())
embeddings = np.concatenate(embeddings)
labels = np.concatenate(labels)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
embeddings_tsne = tsne.fit_transform(embeddings)

# Plot t-SNE embeddings
plt.figure(figsize=(8,8))
for i in range(10):
    plt.scatter(embeddings_tsne[labels==i,0], embeddings_tsne[labels==i,1], label=f'Class {i}')
plt.legend()
plt.title('t-SNE Embeddings for Training Data')
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()
