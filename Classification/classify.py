import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class ConvNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3))
    self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    # self.f1 = nn.Flatten()
    self.fc1 = nn.Linear(64*124*124, 128)
    self.fc2 = nn.Linear(128, 42)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    # x = self.f1(x)
    x = x.view(-1, 64*124*124)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

data_path = input("Введите путь к папке с датасетом: ")#"./Data/simpsons_dataset"
trans = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
dataset = datasets.ImageFolder(data_path, transform=trans)

data_path = "./testset"
testset = datasets.ImageFolder(data_path, transform=trans)

test_loader = DataLoader(testset, batch_size=32, shuffle=True)

labels = ['Дед Абрахам',
          'Агнес Скиннер',
          'Апу индус',
          'Барни Гамбли',
          'Барт',
          'Карл Карлсон',
          'Чарльз Бёрнс',
          'Шеф Виггум',
          'Клетус',
          'Чел с комиксами',
          'Диско Стью',
          'Една Крабаппел',
          'Толстяк Тони',
          'Гил',
          'Вилли',
          'Гомер',
          'Кент',
          'Клоун Красти',
          'Ленни Леонард',
          'Лионель Хатз',
          'Лизякула',
          'Мэгги',
          'Мардж',
          'Мартин Принц',
          'Мэр Кимби',
          'Милхаус ван Хутен',
          'Мисс Хувер',
          'Мо Сизлок',
          'Нед Фландерс',
          'Нельсон Мунтз',
          'Отто Манн',
          'Пэтти Бувер',
          'Директор Скиннер',
          'Фрик',
          'Райнер Волкзамок',
          'Ральф Виггум',
          'Сельма Бувер',
          'Сайдшоу Боб',
          'Змея Птицаклетка',
          'Трой Маккларен',
          'Вейлон Смиттерс']

model = ConvNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

image = Image.open('./Data/kaggle_simpson_testset/kaggle_simpson_testset/comic_book_guy_1.jpg')
plt.imshow(image)
plt.show
image = trans(image).unsqueeze(0).cuda()
with torch.no_grad():
  output = model(image)
pred = (output.argmax(dim=1)).to('cpu').numpy()
print(labels[pred[0]])
