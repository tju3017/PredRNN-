import torch
import torch.nn as nn

class GHU(nn.Module):
    def __init__(self, inputs_shape, num_features, lnorm):
        super(GHU,self).__init__()
        """Initialize the Gradient Highway Unit.
        """
        self.num_features = num_features
        self.layer_norm = lnorm
        self.batch = inputs_shape[0]
        self.height = inputs_shape[3]
        self.width = inputs_shape[2]
        if self.layer_norm:
            self.z_concat_conv = nn.Sequential(
                nn.Conv2d(self.num_features, self.num_features * 2, kernel_size=5, stride=1, padding=2, bias=False),
                nn.LayerNorm([self.num_features*2, self.width, self.width])
            )
            self.x_concat_conv = nn.Sequential(
                nn.Conv2d(self.num_features, self.num_features * 2, kernel_size=5, stride=1, padding=2, bias=False),
                nn.LayerNorm([self.num_features*2, self.width, self.width])
            )
        else:
            self.z_concat_conv = nn.Sequential(
                nn.Conv2d(self.num_features, self.num_features * 2, kernel_size=5, stride=1, padding=2, bias=False)
            )
            self.x_concat_conv = nn.Sequential(
                nn.Conv2d(self.num_features, self.num_features * 2, kernel_size=5, stride=1, padding=2, bias=False)
            )
    def init_state(self):
        return torch.zeros((self.batch,self.num_features,self.width,self.height), dtype=torch.float32).cuda()


    def forward(self,x,z):
        if z is None:
            z = self.init_state()
        z_concat = self.z_concat_conv(z)
        x_concat = self.x_concat_conv(x)
        
        gates = torch.add(x_concat, z_concat)

        p, u = torch.split(gates, self.num_features, 1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1 - u) * z
        return z_new
