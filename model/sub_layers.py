import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.silu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        d_hidden = int(d_model * ff_mult)

        self.proj_in = nn.Linear(d_model, d_hidden * 2, bias=False)
        self.proj_out = nn.Linear(d_hidden, d_model, bias=False)

        self.dropout_in = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, x):
        gate, up = self.proj_in(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.dropout_in(x)

        x = self.proj_out(x)
        x = self.dropout_out(x)

        return x
