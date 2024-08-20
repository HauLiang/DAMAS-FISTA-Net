import torch 
import torch.nn as nn


class DAMAS_FISTA_Net(nn.Module):
    """ The DAMAS-FISTA-Net Architecture """

    def __init__(self, LayerNo, show_pre_imaging_result=False):
        super(DAMAS_FISTA_Net, self).__init__()

        # Learnable parameters
        self.wk_weight = nn.Parameter(torch.Tensor([1]))
        self.iota = nn.Parameter(torch.Tensor([1 for i in range(LayerNo)]))
        self.rho = nn.Parameter(torch.Tensor([1 for i in range(LayerNo)]))
        self.tau = nn.Parameter(torch.Tensor([1]))
        self.mu = nn.Parameter(torch.Tensor([1]))
        self.eta_1 = nn.Parameter(torch.Tensor([1]))
        self.eta_2 = nn.Parameter(torch.Tensor([1]))

        self.LayerNo = LayerNo
        self.show_pre_imaging_result = show_pre_imaging_result
        self.Sp = nn.Softplus()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(1681, 1681)

    def forward(self, CSM_K, wk_reshape_K, A_K, ATA_K, L_K):
        
        # Number of microphone array
        N_mic = CSM_K.shape[1]

        # Pre-imaging layer
        b = torch.sum(torch.mul(self.wk_weight*wk_reshape_K.conj(), torch.matmul(CSM_K, self.wk_weight*wk_reshape_K)), dim=1, keepdim=True) / (N_mic ** 2)
        b = torch.real(b).permute(0, 2, 1)

        if self.show_pre_imaging_result:
            return b

        # Initialization
        xold = torch.zeros(b.shape).to(torch.float64)
        if torch.cuda.is_available():
            xold = xold.cuda()
        y = xold 

        # Pre-calculation
        ATb_K = torch.matmul(A_K.permute(0, 2, 1), b)

        # Iteration layers
        for k in range(self.LayerNo):

            # Reconstruction layer
            rk = self.iota[k] * y - (1 / L_K) * self.rho[k] * torch.matmul(ATA_K, y) + (1 / L_K) * self.rho[k] * ATb_K

            # Nonlinear transform layer
            xk = self.relu(rk)

            # Momentum layer
            y = self.tau * xk + self.mu * (xk - xold)

            xold = xk
        
        # Mapping layer
        temp = xk
        xk = torch.squeeze(xk, 2).to(torch.float32)
        xk = self.linear(xk)
        xk = torch.unsqueeze(xk, 2).to(torch.float64)
        xk = self.relu(self.eta_1 * xk + self.eta_2 * temp)
    
        return xk
