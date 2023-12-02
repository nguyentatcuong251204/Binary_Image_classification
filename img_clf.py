import os
import glob

from PIL import Image
import torch
from torch import save,load
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
from torchvision import transforms
import matplotlib.pyplot as plt

car_path_folder="D:\image_clf\car"
truck_path_folder="D:\image_clf\\truck"
train_transform=transforms.Compose([
                transforms.RandomResizedCrop(224,scale=(0.5,1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
class ImageTransform():
    def __init__(self):
        self.data_transform = {
            'train' : transforms.Compose([
                transforms.RandomResizedCrop(224,scale=(0.5,1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ),
            'val' : transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        }
    
    def __call__(self,img,phase='train'):
        return self.data_transform[phase](img)


class CarAndTruck(Dataset):
    def __init__(self,folder_path1,folder_path2,phase="train",transform=None):
        self.folder_path1=folder_path1
        self.folder_path2=folder_path2
        self.transform=transform
        self.phase=phase
    def __len__(self):
        return len(os.listdir(self.folder_path1))+len(os.listdir(self.folder_path2))
    
    def __getitem__(self,idx ) :
        image_path_list=glob.glob("D:\image_clf\**\*.jpg")
        image=Image.open(image_path_list[idx])
        image_transed=self.transform(image,self.phase)
        label=1 if "car" in image_path_list[idx] else 0
        return image_transed,label
    



class Imageclassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3 ,out_channels=16,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
           
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(128*144,2)
        
    def forward(self,X):
        X=self.model(X)
        X=self.flatten(X)
        X=self.linear1(X)
        return X
    
model=Imageclassification().to("cuda")
dataset=CarAndTruck(car_path_folder,truck_path_folder,phase='train',transform=ImageTransform())
print(dataset)
train=DataLoader(dataset,batch_size=20,shuffle=True)
optimizer=Adam(model.parameters(),lr=0.001)
criterion=nn.CrossEntropyLoss()

#TRAIN MODEL 
# for epoch in range(200):
#     for batch in train:
#         X,y=batch
#         X,y=X.to('cuda'),y.to('cuda')
#         yhat=model(X)
#         loss=criterion(yhat,y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
     
#     print(f"epoch {epoch} loss is {loss.item()}")

# img=Image.open(r"D:\image_clf\truck\25295937.jpg")
# preprocess=ImageTransform()
# print(model(preprocess(img,phase='train')))

# with open('weight.pt','wb') as f:
#     save(model.state_dict(),f)


#PREDICTION
if __name__=='__main__':
    with open('weight.pt','rb') as f:
        model.load_state_dict(load(f))
    path=r'D:\image_clf\truck\13033585.jpg'
    image=Image.open(path)
    trans=ImageTransform()
    image=trans(image,phase='train')
    image=image.unsqueeze(0).to('cuda')
    if (torch.argmax(model(image))==1):
        image=Image.open(path)
        plt.imshow(image)
        plt.title("CAR",size=40)
        plt.show()
    else:
        image=Image.open(path)
        plt.imshow(image)
        plt.title("TRUCK",size=40)
        plt.show()

