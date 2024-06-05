from torchvision import datasets
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision



class Denormalise(transforms.Normalize):

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-12)
        mean_inv = -mean * std_inv
        super(Denormalise, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(Denormalise, self).__call__(tensor.clone())
image_transform = transforms.ToPILImage()
image_denormalise = Denormalise([0.5] * 3, [0.5] * 3)

def visual(data):

    train_loader=torch.utils.data.DataLoader(data,batch_size=100,shuffle=False)
    i = 0
    images = []
    for batch in train_loader:
        if (i == 0):
            for j in range(len(batch)):
                image = image_denormalise(batch[j])
                image = image_transform(image.clamp(min=0.0, max=1.0))
                images.append(image)
              # image.shape==torch.Size([1, 3, 32, 32])

            # print(label)
            i += 1
        else:
            break
    images = np.stack(images)
    images = torch.tensor(images)

    #images=torch.squeeze(images)#torch.Size([3, 32, 32])#显示单张的时候用
    print(images.shape)  # torch.Size([10,  32, 32,3])
    images = np.transpose(images, (0, 3, 1, 2))
    #显示
    grid=torchvision.utils.make_grid(images,nrow=10)
    plt.imshow(np.transpose(grid, (1,2,0)))#交换维度，从GBR换成RGB
    plt.show()

