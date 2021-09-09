
import torch
import numpy as np
def crop(size, lam):

    cx = np.random.randint(size[2])
    cy = np.random.randint(size[3])

    cut_rat = np.sqrt(1.0-lam)
    cut_w = np.int(size[2] * cut_rat)
    cut_H = np.int(size[3] * cut_rat)

    x1 = np.clip(cx-cut_w//2, 0, size[2])
    y1 = np.clip(cy-cut_H//2, 0, size[3])
    x2 = np.clip(cx + cut_w//2, 0, size[2])
    y2 = np.clip(cy + cut_H//2, 0, size[3])
    return x1,y1,x2,y2

import numpy as np
def cutmix(data, target, alpha):
    indices = np.random.permutation(data.shape[0])
    target = np.array(target)
    # indices = list(indices)
    tmp_target = target[indices]

    lam = np.clip(np.random.beta(alpha,alpha), 0.3, 0.4)
    x1, y1, x2, y2 = crop(data.shape, lam)
    print(x1,y1,x2,y2)
    # 随机生成切片的中心cx，cy，然后切W*lam, H*lam大小的切片，切片的x1,y1,x2,y2已计算
    # 将bbox放到其他的indices上，
    new_data = data.copy()

    new_data[:, :, y1:y2, x1:x2] = data[indices, :, y1:y2, x1:x2]
    lam = 1.0 - (x2-x1)*(y2-y1)/data.shape[3]*data.shape[2]
    target = (target, tmp_target, lam)
    return new_data, target


from PIL import Image
import matplotlib.pyplot as plt
def get_result():
    image_fullname = r'E:\learning\learn\images\train.jpg'
    image_fullname1 = r'E:\second-competition\data\train_data\train_image\00c4fc80-6288-421f-9f16-05dac324f096.jpg'
    image = Image.open(image_fullname).convert('RGB').resize((256,256))
    image1 = Image.open(image_fullname1).convert('RGB').resize((256,256))

    image = np.array(image)
    image = np.reshape(image, (image.shape[2], image.shape[0], image.shape[1]))

    image = np.expand_dims(image, axis=0)
    image1 = np.array(image1)
    image1 = np.reshape(image1, (image1.shape[2], image1.shape[0], image1.shape[1]))
    image1 = np.expand_dims(image1, axis=0)

    result = np.concatenate([image,image1], axis=0)
    print(result.shape)
    new_data, target = cutmix(result, [0,1], 1)
    show1 = new_data[0]

    show1 = np.reshape(show1, (show1.shape[1], show1.shape[2], show1.shape[0]))

    show2 = new_data[1]

    plt.figure()
    plt.subplot(1,2,1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(show1)
    plt.subplot(1,2,2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.reshape(show2, (show2.shape[1],show2.shape[2],show2.shape[0])))
    plt.show()

def test_cutmix():
    image_fullname = r'E:\learning\learn\images\train.jpg'
    image = Image.open(image_fullname).convert('RGB').resize((256, 256))
    image = np.array(image)
    image = np.reshape(image, (image.shape[2], image.shape[0], image.shape[1]))

    new_image = image.copy() #chw
    image = np.reshape(image, (image.shape[1],image.shape[2],image.shape[0]))#hwc


    new_image = new_image[:, 50:256, 113:256] #chw 切块
    #切块后的chw转换为hwc
    new_image = np.reshape(new_image, (new_image.shape[1], new_image.shape[2], new_image.shape[0]))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(new_image)
    plt.show()


# get_result()
test_cutmix()