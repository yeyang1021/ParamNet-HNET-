from torch import nn


class HNet(nn.Module):
    """HNet encoder."""

    def __init__(self):
        """Init HNet encoder."""
        super(HNet, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv block
            # input [1 x 28 x 28]
            # output [64 x 12 x 12]
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),           
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),             
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),             
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),             
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),            
            nn.MaxPool2d(2)

        )
        self.fc0 =  nn.Sequential(
            nn.Linear( 128*11*20, 1024),
            #nn.BatchNorm2d(1024),
            nn.ReLU()
            )
        self.fc1 = nn.Linear(1024, 5)

    def forward(self, input):
        """Forward the HNet."""
        conv_out = self.encoder(input)
        fc0 = self.fc0(conv_out.view(-1, 128*11*20))
        fc = self.fc1(fc0)
        return fc
