import torch
import torch.nn as nn
from GradientHighway import GHU as ghu
from CausalLSTMCell import CausalLSTMCell as cslstm
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from dateprocess_convlstm import *
from tensorboardX import SummaryWriter
from dateprocess_convlstm import *
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
batch_size = 1
valid_size = 0
#batch_size = 1000
shuffle_dataset = True
random_seed = 1222

if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def MNISTdataLoader(path):
    # load moving mnist data, data shape = [time steps, batch size, width, height] = [20, batch_size, 64, 64]
    # B S H W -> S B H W
    data = np.load(path)
    data_trans = data.transpose(1, 0, 2, 3)
    return data_trans


class MovingMNISTdataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = MNISTdataLoader(path)

    def __len__(self):
        return len(self.data[:, 0, 0, 0])

    def __getitem__(self, indx):
        self.trainsample_ = self.data[indx, ...]
        # self.sample_ = self.trainsample_/255.0   # normalize
        self.sample_ = self.trainsample_
        self.sample = torch.from_numpy(np.expand_dims(self.sample_, axis=1)).float()
        return self.sample

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, shape):
        super(RNN, self).__init__()

        self.img_width = shape[2] # 倒数第二个
        self.img_height = shape[3]
        self.batch = shape[0]
        self.total_length = 12 # what is total_length
        self.input_length = 6 # what is input_length
        self.shape = shape
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.MSE_criterion = nn.MSELoss()
        cell_list = []
        ghu_list = []

        for i in range(self.num_layers):
            if i == 0:
                #此处与tf代码不一致
                num_hidden_in = 3
            else:
                num_hidden_in = self.num_hidden[i - 1]
            cell_list.append(cslstm(num_hidden_in,
                                   num_hidden[i],
                                   self.shape, 1.0))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[-1], 3, 1, 1, 0)
        ghu_list.append(ghu(self.shape, self.num_hidden[0]))
        self.ghu_list = nn.ModuleList(ghu_list)


    def forward(self, images_tensor):
        # [batch, length, channel, width, height]
        images = images_tensor.permute(0, 1, 4, 2, 3).contiguous()
        #mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        #print()
        batch = images.shape[0]
        height = images.shape[3]
        width = images.shape[4]

        next_images = []
        h_t = []
        c_t = []
        z_t = None
        m_t = None

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.total_length - 1):
            #if self.scheduled_sampling == 1:
            if t < self.input_length:
                net = images[:, t]
            else:
            #     net = mask_true[:, t - self.input_length] * images[:, t] + \
            #             (1 - mask_true[:, t - self.input_length]) * x_gen
                net = x_gen
            h_t[0], c_t[0], m_t = self.cell_list[0](net, h_t[0], c_t[0], m_t)
            z_t = self.ghu_list[0](h_t[0],z_t)
            h_t[1], c_t[1], m_t = self.cell_list[1](z_t, h_t[1], c_t[1], m_t)

            for i in range(2, self.num_layers):
                h_t[i], c_t[i], m_t = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], m_t)

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_images.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_images = torch.stack(next_images, dim=1)
        out = next_images
        # print(out.shape)
        # print(images[:, 1:].shape)
        #loss = self.MSE_criterion(next_images, images[:, 1:])
        return out

train_set = GBC_PV_pred_dataset(csv_file=r'E:\STPred\group_meeting.csv',
                                root_dir= r'E:\STPred\train_new',
                                transform=transforms.Compose({
                                    #Rescale(224),
                                    ToTensor()
                                }))

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True)
#a=torch.randn(1, 20, 32, 32, 1)
#b=torch.randn(1, 9, 32, 32, 3)

# batch_size = 8
# mnistdata = MovingMNISTdataset(r"E:\py\.py\mnist_test_seq.npy")
# train_size = int(0.8 * len(mnistdata))
# test_size = len(mnistdata) - train_size
# torch.manual_seed(torch.initial_seed())
# train_dataset, test_dataset = random_split(mnistdata, [train_size, test_size])
#
# num_train = len(train_dataset)
# indices = list(range(num_train))
# np.random.shuffle(indices)
# split = int(np.floor(valid_size * num_train))
#
# if shuffle_dataset:
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_idx, valid_idx = indices[split:], indices[:split]
#
# # define samplers for obtaining training and validation batches
# train_sampler = SubsetRandomSampler(train_idx)
# valid_sampler = SubsetRandomSampler(valid_idx)
#
# # load training data in batches
# train_loader = DataLoader(train_dataset,
#                           batch_size=batch_size,
#                           sampler=train_sampler,
#                           num_workers=2)

use_gpu = False
device = torch.device("cuda" if use_gpu else "cpu")
shape = [1, 3, 224, 224]
num_layers = 4
num_hidden = [64, 64, 64, 64]
#writer = SummaryWriter(r'E:\STPred\pythonProject1\venv\Scripts\runs/scalar_example')
predrnnpp = RNN(num_layers, num_hidden, shape).to(device)
predrnnpp.load_state_dict(torch.load("predrnnpp1128_epoch0.pkl", map_location='cpu'))
optimizer = torch.optim.Adam(predrnnpp.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
batch_size=1
total_length = 20
input_length = 10
img_width = 32
img_channel = 1
def schedule_sampling(eta):
    random_flip = np.random.random_sample((batch_size, total_length - input_length - 1))
    #随机给出在[0, 1)半开半闭区间的随机数
    true_token = (random_flip < eta)
    #print(true_token.shape)
    ones = np.ones((img_width, img_width, img_channel))
    zeros = np.zeros((img_width, img_width, img_channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(total_length - input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  total_length - input_length - 1,
                                  img_width,
                                  img_width,img_channel))
    mask = torch.from_numpy(real_input_flag).to(torch.float32)
    return mask
def main():
    for epoch in range(1000):
        predrnnpp.eval()
        # #loss_total = 0
        # #count = 0
        # mask = schedule_sampling(0.5)
        # optimizer.zero_grad()
        # out, loss = predrnnpp(a, mask)
        # loss.backward()
        # optimizer.step()
        # print(loss)
        for m, sample in enumerate(train_loader):
            image1 = Variable(sample['image1']).float()
            image2 = Variable(sample['image2']).float()
            image3 = Variable(sample['image3']).float()
            image4 = Variable(sample['image4']).float()
            image5 = Variable(sample['image5']).float()
            image6 = Variable(sample['image6']).float()
            # image7 = Variable(sample['image7']).float()
            # image8 = Variable(sample['image8']).float()
            # image9 = Variable(sample['image9']).float()
            # image10 = Variable(sample['image10']).float()
            # image11 = Variable(sample['image11']).float()
            # image12 = Variable(sample['image12']).float()
            # image7 = torch.randn(1, 3, 224, 224)
            # image8 = torch.randn(1, 3, 224, 224)
            # image9 = torch.randn(1, 3, 224, 224)
            # image10 = torch.randn(1, 3, 224, 224)
            # image11 = torch.randn(1, 3, 224, 224)
            # image12 = torch.randn(1, 3, 224, 224)
            # label1 = Variable(sample['label1']).float()
            # label2 = Variable(sample['label2']).float()
            # label3 = Variable(sample['label3']).float()
            # label4 = Variable(sample['label4']).float()
            # label5 = Variable(sample['label5']).float()
            # label6 = Variable(sample['label6']).float()
            label7 = Variable(sample['label7']).float()
            label8 = Variable(sample['label8']).float()
            label9 = Variable(sample['label9']).float()
            label10 = Variable(sample['label10']).float()
            label11 = Variable(sample['label11']).float()
            label12 = Variable(sample['label12']).float()
            images = torch.stack(
                (image1, image2, image3, image4, image5, image6, label7, label8, label9, label10, label11, label12),
                dim=1)

            images = images.permute(0, 1, 3, 4, 2)
            img = images / 255
            label = img[:, 1:].permute(0, 1, 4, 2, 3)
            #print(img.dtype)
            print(label.shape)
            #mask = schedule_sampling(0.5)
            #optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0
            out = predrnnpp(img)
            #print(out.shape)
            #out_img = np.squeeze(out, 0)
            for n in range(out.shape[1]):
                out_img = out[:, n:n+1,:,:,:]
                out_img = np.squeeze(out_img, 0)
                out_img = np.squeeze(out_img, 0)
                print(out_img.shape)
                #out_img = np.squeeze(out_img, 0)
                #print(out_img.shape)
                out_img = out_img.transpose(0, 1)
                out_img = out_img.transpose(1, 2)
                print(out_img.shape)
                out_img = out_img.data.cpu().numpy()
                plt.axis('off')
                fig = plt.gcf()
                fig.set_size_inches(9/3, 9/3)  # dpi = 300, output = 700*700 pixels
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.imshow(out_img)
                plt.show()
            loss = loss_fn(out, label)
            print(loss)
            #loss.backward()
            #loss_total+=loss.item()
            #count+=1
            #loss_epoch = loss_total/count
            #optimizer.step()
            #print(loss_epoch/count)
            #writer.add_scalar('Loss', loss_total/count, global_step=m)
        #print(loss)


if __name__=="__main__":
    main()

# from tensorboardX import SummaryWriter
# import cv2 as cv
#
# writer = SummaryWriter(r'E:\STPred\pythonProject1\venv\Scripts\runs\image_example')
# for i in range(1, 2):
#     writer.add_image('countdown',
#                      cv.cvtColor(cv.imread('{}.jpg'.format(i)), cv.COLOR_BGR2RGB),
#                      global_step=i,
#                      dataformats='HWC')
