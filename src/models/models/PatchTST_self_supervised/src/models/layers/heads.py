import torch
from torch import nn


class LinearRegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)        
        return y


class LinearClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y


class LinearPredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.linear(x)
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class ResNetClassificationHead(nn.Module):
    """
    ResNet-style classification head for PatchTST.

    Unlike ClassificationHead (which only uses the last patch), this head
    uses ALL patches by treating the patch dimension as a 1D sequence and
    running it through residual conv blocks before classification.

    Input:  [bs x nvars x d_model x num_patch]
    Output: [bs x n_classes]

    Architecture:
        1. Reshape → [bs x (nvars * d_model) x num_patch]
           (treat each (variable, embedding) pair as a channel)
        2. Two 1D residual blocks (conv → BN → ReLU → conv → BN + skip)
        3. Global average pool → [bs x hidden_dim]
        4. Dropout → Linear → [bs x n_classes]

    hidden_dim controls the number of conv filters in the residual blocks.
    Defaults to n_vars * d_model (same width as input channels).
    """
    def __init__(self, n_vars, d_model, n_classes, head_dropout, hidden_dim=None):
        super().__init__()
        in_channels = n_vars * d_model
        hidden_dim = hidden_dim or in_channels

        # Residual block 1: in_channels → hidden_dim
        self.res1_conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.res1_bn1   = nn.BatchNorm1d(hidden_dim)
        self.res1_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.res1_bn2   = nn.BatchNorm1d(hidden_dim)
        # 1x1 conv to match dimensions for skip connection if channels differ
        self.res1_skip  = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False) \
                          if in_channels != hidden_dim else nn.Identity()

        # Residual block 2: hidden_dim → hidden_dim
        self.res2_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.res2_bn1   = nn.BatchNorm1d(hidden_dim)
        self.res2_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.res2_bn2   = nn.BatchNorm1d(hidden_dim)

        self.relu    = nn.ReLU()
        self.gap     = nn.AdaptiveAvgPool1d(1)   # global average pool over patches
        self.dropout = nn.Dropout(head_dropout)
        self.linear  = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        """
        bs, nvars, d_model, num_patch = x.shape
        # [bs x (nvars * d_model) x num_patch]
        x = x.reshape(bs, nvars * d_model, num_patch)

        # Residual block 1
        identity = self.res1_skip(x)
        out = self.relu(self.res1_bn1(self.res1_conv1(x)))
        out = self.res1_bn2(self.res1_conv2(out))
        x = self.relu(out + identity)

        # Residual block 2
        identity = x
        out = self.relu(self.res2_bn1(self.res2_conv1(x)))
        out = self.res2_bn2(self.res2_conv2(out))
        x = self.relu(out + identity)

        # Global average pool → [bs x hidden_dim]
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.linear(x)   # [bs x n_classes]


class LinearPretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x

