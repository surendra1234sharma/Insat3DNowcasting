import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_dim, hid_dim, k):
        super().__init__()
        pad = k // 2
        self.hid_dim = hid_dim

        self.conv = nn.Conv2d(in_dim + hid_dim, 4 * hid_dim, k, padding=pad)

    def forward(self, x, h, c):
        out = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(out, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()

        self.cell1 = ConvLSTMCell(1, hidden, 3)
        self.cell2 = ConvLSTMCell(hidden, hidden, 3)

        self.out = nn.Conv2d(hidden, 1, 1)

    def forward(self, x, future=2):
        B, T, C, H, W = x.shape
        device = x.device

        h1 = torch.zeros(B, 32, H, W).to(device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros_like(h1)
        c2 = torch.zeros_like(h1)

        for t in range(T):
            h1, c1 = self.cell1(x[:, t], h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)

        outputs = []
        frame = x[:, -1]

        for _ in range(future):
            h1, c1 = self.cell1(frame, h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)
            frame = self.out(h2)
            outputs.append(frame)

        return torch.stack(outputs, dim=1)