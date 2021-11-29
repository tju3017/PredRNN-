import torch
import torch.nn as nn
from GradientHighway import GHU as ghu
from CausalLSTMCell import CausalLSTMCell as cslstm


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, shape, lnorm):
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
                                   self.shape, 1.0, lnorm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[-1], 3, 1, 1, 0)
        ghu_list.append(ghu(self.shape, self.num_hidden[0], lnorm))
        self.ghu_list = nn.ModuleList(ghu_list)


    def forward(self, images_tensor， mask_true):
        # [batch, length, channel, width, height]
        images = images_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
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
            if self.scheduled_sampling == 1:
                if t < self.input_length:
                    net = images[:, t]
                else:
                    net = mask_true[:, t - self.input_length] * images[:, t] + \
                         (1 - mask_true[:, t - self.input_length]) * x_gen

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
        loss = self.MSE_criterion(next_images, images[:, 1:])
        return out， loss


a=torch.randn(1, 20, 224, 224, 3)

use_gpu = False
device = torch.device("cuda" if use_gpu else "cpu")
shape = [1, 3, 224, 224]
num_layers = 4
num_hidden = [64, 64, 64, 64]
predrnnpp = RNN(num_layers, num_hidden, shape).to(device)
optimizer = torch.optim.Adam(predrnnpp.parameters(), lr=1e-3)

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
        
        mask = schedule_sampling(0.5)
        optimizer.zero_grad()
        out, loss = predrnnpp(a, mask)
        loss.backward()
        optimizer.step()
        print(loss)


if __name__=="__main__":
    main()
