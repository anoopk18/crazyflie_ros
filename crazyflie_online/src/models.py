from NODE.NODE import *


class RigidHybrid(ODEF):
    def __init__(self):  # quadparams is only a placeholder, not actually used
        super(RigidHybrid, self).__init__()
        self.layers = [nn.Linear(9, 32),
                       nn.Tanh(),
                       nn.Linear(32, 6)]
        
        for i in range(len(self.layers)):
            if str(self.layers[i])[:6] == 'Linear':
                torch.nn.init.xavier_uniform_(self.layers[i].weight)
                torch.nn.init.zeros_(self.layers[i].bias)
    
        self.nn_model = nn.Sequential(*self.layers)
        self.n_old_layers = 0
    
    def cascade(self):
        self.n_old_layers = len(self.nn_model)
        new_layers = [nn.Linear(6, 16),
                      nn.Tanh(),
                      nn.Linear(16, 6)]
        for i in range(len(new_layers)):
            if str(self.layers[i])[:6] == 'Linear':
                torch.nn.init.xavier_uniform_(new_layers[i].weight)
                torch.nn.init.zeros_(new_layers[i].bias)

        self.layers += new_layers
        self.nn_model = nn.Sequential(*self.layers)
 

    def forward(self, z):
        bs,_,_ = z.size()
        z = z.squeeze(1)
        x = z[:, :6]  # 6D state
        u = z[:, 6:]  # 3D input
        
        x_dot = torch.zeros([bs, 6])
        x_dot[:, :3] = x[:, 3:]
        x_dot[:, 3:] = u
        
        x_dot_nn = self.nn_model(z)

        x_dot_hybrid = x_dot + x_dot_nn
        out = torch.cat([x_dot_hybrid, torch.zeros([bs, 3])], 1)
        #print("output shape is: ", out.shape)
        return out.unsqueeze(1)
        
