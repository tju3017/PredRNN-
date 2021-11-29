import torch
import torch.nn as nn

class CausalLSTMCell(nn.Module):
    def __init__(self, num_hidden_in,num_hidden_out,
                 seq_shape, forget_bias):
        super(CausalLSTMCell, self).__init__()
        """Initialize the Causal LSTM cell.
        Args:
            layer_name: layer names for different lstm layers.
            filter_size: int tuple that the height and width of the filter.
            num_hidden_in: number of units for input tensor.
            num_hidden_out: number of units for output tensor.
            seq_shape: shape of a sequence.
            forget_bias: float, The bias added to forget gates.
            tln: whether to apply tensor layer normalization
        """
        self.num_hidden_in = num_hidden_in
        self.num_hidden_out = num_hidden_out
        self.batch = seq_shape[0]
        self.height = seq_shape[3]
        self.width = seq_shape[2]
        self.layer_norm = 0
        self._forget_bias = forget_bias
        #self.width = width
        #self.bn_h_cc = tensor_layer_norm(self.num_hidden_out * 4)
        #self.bn_c_cc = tensor_layer_norm(self.num_hidden_out * 3)
        #self.bn_m_cc = tensor_layer_norm(self.num_hidden_out * 3)
        #self.bn_x_cc = tensor_layer_norm(self.num_hidden_out * 7)
        #self.bn_c2m = tensor_layer_norm(self.num_hidden_out * 4)
        #self.bn_o_m = tensor_layer_norm(self.num_hidden_out)

        self.h_cc_conv = nn.Conv2d(self.num_hidden_out,self.num_hidden_out*4,5,1,2)
        self.c_cc_conv = nn.Conv2d(self.num_hidden_out,self.num_hidden_out*3,5,1,2)
        self.m_cc_conv = nn.Conv2d(self.num_hidden_out,self.num_hidden_out*3,5,1,2)
        self.x_cc_conv = nn.Conv2d(self.num_hidden_in,self.num_hidden_out*7,5,1,2)
        self.c2m_conv  = nn.Conv2d(self.num_hidden_out,self.num_hidden_out*4,5,1,2)
        self.o_m_conv = nn.Conv2d(self.num_hidden_out,self.num_hidden_out,5,1,2)
        self.o_conv = nn.Conv2d(self.num_hidden_out, self.num_hidden_out, 5, 1, 2)
        self.cell_conv = nn.Conv2d(self.num_hidden_out*2,self.num_hidden_out,1,1,0)


    def init_state(self):
        return torch.zeros((self.batch, self.num_hidden_out,self.width,self.height),dtype=torch.float32)

    def forward(self,x,h,c,m):
        if h is None:
            h = self.init_state()
        if c is None:
            c = self.init_state()
        if m is None:
            m = self.init_state()
        h_cc = self.h_cc_conv(h)
        c_cc = self.c_cc_conv(c)
        m_cc = self.m_cc_conv(m)
        if self.layer_norm:
            h_cc = nn.LayerNorm(self.num_hidden_out*4, self.width, self.height)
            c_cc = nn.LayerNorm(self.num_hidden_out*3, self.width, self.height)
            m_cc = nn.LayerNorm(self.num_hidden_out*3, self.width, self.height)

        #最后一个1表示dim=1
        i_h, g_h, f_h, o_h = torch.split(h_cc, self.num_hidden_out, 1)
        i_c, g_c, f_c = torch.split(c_cc, self.num_hidden_out, 1)
        i_m, f_m, m_m = torch.split(m_cc, self.num_hidden_out, 1)
        if x is None:
            i = torch.sigmoid(i_h+i_c)
            f = torch.sigmoid(f_h + f_c + self._forget_bias)
            g = torch.tanh(g_h + g_c)
        else:
            x_cc = self.x_cc_conv(x)
            if self.layer_norm:
                x_cc = nn.LayerNorm(self.num_hidden_out*7, self.width, self.height)

            i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = torch.split(x_cc,self.num_hidden_out, 1)
            i = torch.sigmoid(i_x + i_h+ i_c)
            f = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
            g = torch.tanh(g_x + g_h + g_c)
        c_new = f * c + i * g
        c2m = self.c2m_conv(c_new)
        if self.layer_norm:
            c2m = nn.LayerNorm(self.num_hidden_out*4, self.width, self.height)

        i_c, g_c, f_c, o_c = torch.split(c2m, self.num_hidden_out, 1)

        if x is None:
            ii = torch.sigmoid(i_c + i_m)
            ff = torch.sigmoid(f_c + f_m + self._forget_bias)
            gg = torch.tanh(g_c)
        else:
            ii = torch.sigmoid(i_c + i_x_ + i_m)
            ff = torch.sigmoid(f_c + f_x_ + f_m + self._forget_bias)
            gg = torch.tanh(g_c + g_x_)
        m_new = ff * torch.tanh(m_m) + ii * gg
        o_m = self.o_m_conv(m_new)
        if self.layer_norm:
             o_m = nn.LayerNorm(self.num_hidden_out, self.width, self.height)
        if x is None:
            o = torch.tanh(o_c + o_m + o_h)

        else:
            o = torch.tanh(o_x + o_c + o_m + o_h)
        #o = self.o_conv(o)
        #此时c_new以及m_new的格式均为[b,c,w,h]
        cell = torch.cat([c_new, m_new],1)
        cell = self.cell_conv(cell)
        # print(cell.shape)
        h_new = o * torch.tanh(cell)
        # print(m_new.shape)
        return h_new, c_new, m_new
