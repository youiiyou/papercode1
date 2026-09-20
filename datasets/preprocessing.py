import random
import math


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability #执行擦除的概率
        self.mean = mean#填充像素的均值
        self.sl = sl#面积的最小比例
        self.sh = sh#面积的最大比例
        self.r1 = r1#最小高宽比

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img #决定要不要擦除

        for attempt in range(100): #尝试100次擦除，防止随机数异常
            area = img.size()[1] * img.size()[2]#面积

            target_area = random.uniform(self.sl, self.sh) * area#随机目标面积
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)#随机宽高比

            h = int(round(math.sqrt(target_area * aspect_ratio))) #高度
            w = int(round(math.sqrt(target_area / aspect_ratio))) #宽度

            if w < img.size()[2] and h < img.size()[1]: #宽高不超出
                x1 = random.randint(0, img.size()[1] - h) #随机x坐标
                y1 = random.randint(0, img.size()[2] - w) #随机y坐标
                if img.size()[0] == 3: #三通道
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else: #单通道 灰度图
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

