import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import shutil
import os

# Удаление существующей папки "testset" и создание новой папки "testset"
shutil.rmtree("testset")
os.mkdir("testset")

# Копирование изображений из папки "kaggle_simpson_testset" в папку "testset"
test_data_name = os.listdir("./Data/kaggle_simpson_testset/kaggle_simpson_testset")
for i in range(len(test_data_name)):
  name = test_data_name[i]
  dst = "testset/"+ name[:name.rfind('_')]
  path = "./Data/kaggle_simpson_testset/kaggle_simpson_testset/"+name

  if not os.path.exists(dst):
    os.mkdir(dst)
  shutil.copyfile(path, dst+"/"+name)

# Загрузка данных из папки "simpsons_dataset"
content = os.listdir('./Data/simpsons_dataset')
content.sort()

# Копирование одного изображения из каждой папки в папку "testset"
for i in content:
    dst = "testset/"+i
    src = "./Data/simpsons_dataset/"+i
    file_name = os.listdir(src)[0]
    src =src +  "/" + os.listdir(src)[0]

    print(src)
    if not os.path.exists(dst):
      os.mkdir(dst)
      dst = dst + "/" + file_name
      shutil.copyfile(src,dst)

# Определение класса ConvNN для сверточной нейронной сети
class ConvNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3))
    self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.fc1 = nn.Linear(64*124*124, 128)
    self.fc2 = nn.Linear(128, 42)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(-1, 64*124*124)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# Создание экземпляра модели ConvNN и определение функции потерь CrossEntropyLoss
model = ConvNN().cuda()
criterion = nn.CrossEntropyLoss().cuda()

# Загрузка данных и преобразование
data_path = "/Data/simpsons_dataset"
trans = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
dataset = datasets.ImageFolder(data_path, transform=trans)

# Разделение набора данных на обучающий и проверочный наборы
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Создание загрузчиков данных для обучающего и проверочного наборов
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Определение количества эпох обучения и запуск цикла обучения
print("Введите количество эпох: ")
num_epochs = int(input())
for epoch in range(num_epochs):
  model.train()
  optimizer = optim.SGD(model.parameters(), lr=0.001)
  running_loss = 0.0
  TP = 0
  TN = 0
  FN = 0
  FP = 0
  for i, data in enumerate(train_loader,0):
    inputs, labels = data
    inputs=inputs.cuda()
    labels=labels.cuda()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    real = sum(outputs.argmax(dim=1)==labels).to('cpu').item()
    TP += real
    TN += 40*32+real
    FN += 32-real
    FP += 32-real
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
  with torch.no_grad():
    for i, data in enumerate(val_loader, 0):
      inputs, labels = data
      inputs = inputs.cuda()
      labels = labels.cuda()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      val_loss += loss.item()
      _, predicted = torch.max(outputs,1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  accuracy_per_epoch = ((TP+TN)/(TP+TN+FN+FP))
  print(f'Epoch {epoch+1}/{num_epochs}, Precision: {TP/(TP+FP)}, Train Loss: {running_loss/len(train_loader)}, Train Accuracy: {accuracy_per_epoch}, Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {100*correct/total}%')

# Сохранение модели в файл "model.pth"
torch.save(model.state_dict(), 'model.pth')
print("Модель сохранена в файл model.pth")
