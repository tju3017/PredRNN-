import torch
import torch.nn as nn

class CausalLSTMCell(nn.Module):
    def __init__(self, num_hidden_in,num_hidden_out,
                 seq_shape, forget_bias, lnorm):
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
        self.layer_norm = lnorm
        self._forget_bias = forget_bias
        if self.layer_norm:
            self.conv_h_cc = nn.Sequential(
                nn.Conv2d(self.num_hidden_out,self.num_hidden_out*4,kernel_size=5,stride=1,padding=2, bias=False),
                nn.LayerNorm([self.num_hidden_out*4, self.width, self.height]))
            self.conv_c_cc = nn.Sequential(
                nn.Conv2d(self.num_hidden_out,self.num_hidden_out*3,kernel_size=5,stride=1,padding=2, bias=False),
                nn.LayerNorm([self.num_hidden_out*3, self.width, self.height]))
            self.conv_m_cc = nn.Sequential(
                nn.Conv2d(self.num_hidden_out,self.num_hidden_out*3,kernel_size=5,stride=1,padding=2, bias=False),
                nn.LayerNorm([self.num_hidden_out*3, self.width, self.height]))
            self.conv_x_cc = nn.Sequential(
                nn.Conv2d(self.num_hidden_in,self.num_hidden_out*7,kernel_size=5,stride=1,padding=2, bias=False),
                nn.LayerNorm([self.num_hidden_out*7, self.width, self.height]))
            self.conv_c2m = nn.Sequential(
                nn.Conv2d(self.num_hidden_out,self.num_hidden_out*4,kernel_size=5,stride=1,padding=2, bias=False),
                nn.LayerNorm([self.num_hidden_out*4, self.width, self.height]))
            self.conv_o_m = nn.Sequential(
                nn.Conv2d(self.num_hidden_out,self.num_hidden_out,kernel_size=5,stride=1,padding=2, bias=False),
                nn.LayerNorm([self.num_hidden_out, self.width, self.height]))
            self.conv_o = nn.Sequential(
                nn.Conv2d(self.num_hidden_out,self.num_hidden_out,kernel_size=5,stride=1,padding=2, bias=False),
                nn.LayerNorm([self.num_hidden_out, self.width, self.height]))
            self.conv_cell = nn.Sequential(
                nn.Conv2d(self.num_hidden_out*2,self.num_hidden_out,kernel_size=5,stride=1,padding=2, bias=False),
                nn.LayerNorm([self.num_hidden_out, self.width, self.height]))
        else:
            self.conv_h_cc = nn.Sequential(
                nn.Conv2d(self.num_hidden_out, self.num_hidden_out * 4, kernel_size=5,stride=1,padding=2, bias=False))
            self.conv_c_cc = nn.Sequential(
                nn.Conv2d(self.num_hidden_out, self.num_hidden_out * 3, kernel_size=5,stride=1,padding=2, bias=False))
            self.conv_m_cc = nn.Sequential(
                nn.Conv2d(self.num_hidden_out, self.num_hidden_out * 3, kernel_size=5,stride=1,padding=2, bias=False))
            self.conv_x_cc = nn.Sequential(
                nn.Conv2d(self.num_hidden_in, self.num_hidden_out * 7, kernel_size=5,stride=1,padding=2, bias=False))
            self.conv_c2m = nn.Sequential(
                nn.Conv2d(self.num_hidden_out, self.num_hidden_out * 4, kernel_size=5,stride=1,padding=2, bias=False))
            self.conv_o_m = nn.Sequential(
                nn.Conv2d(self.num_hidden_out, self.num_hidden_out, kernel_size=5,stride=1,padding=2, bias=False))
            self.conv_o = nn.Sequential(
                nn.Conv2d(self.num_hidden_out, self.num_hidden_out, kernel_size=5,stride=1,padding=2, bias=False))
            self.conv_cell = nn.Sequential(
                nn.Conv2d(self.num_hidden_out * 2, self.num_hidden_out, kernel_size=5,stride=1,padding=2, bias=False))
    def init_state(self):
        return torch.zeros((self.batch, self.num_hidden_out,self.width,self.height),dtype=torch.float32).cuda()

    def forward(self,x,h,c,m):
        if h is None:
            h = self.init_state()
        if c is None:
            c = self.init_state()
        if m is None:
            m = self.init_state()
        h_cc = self.conv_h_cc(h)
        c_cc = self.conv_c_cc(c)
        m_cc = self.conv_m_cc(m)

        i_h, g_h, f_h, o_h = torch.split(h_cc, self.num_hidden_out, 1)
        i_c, g_c, f_c = torch.split(c_cc, self.num_hidden_out, 1)
        i_m, f_m, m_m = torch.split(m_cc, self.num_hidden_out, 1)
        if x is None:
            i = torch.sigmoid(i_h+i_c)
            f = torch.sigmoid(f_h + f_c + self._forget_bias)
            g = torch.tanh(g_h + g_c)
        else:
            x_cc = self.conv_x_cc(x)
            
            i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = torch.split(x_cc,self.num_hidden_out, 1)
            i = torch.sigmoid(i_x + i_h+ i_c)
            f = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
            g = torch.tanh(g_x + g_h + g_c)
        c_new = f * c + i * g
        c2m = self.conv_c2m(c_new)

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
        o_m = self.conv_o_m(m_new)

        if x is None:
            o = torch.tanh(o_c + o_m + o_h)

        else:
            o = torch.tanh(o_x + o_c + o_m + o_h)

        cell = torch.cat([c_new, m_new],1)
        cell = self.conv_cell(cell)

        h_new = o * torch.tanh(cell)

        return h_new, c_new, m_new
