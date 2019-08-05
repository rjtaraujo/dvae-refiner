import torch
import torch.nn as nn

class Unet_2levels(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                    align_corners=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    stride=2,
                                    padding=0)

        self.l11 = nn.Conv2d(1,64,3,padding=1)
        self.l12 = nn.Conv2d(64,64,3,padding=1)
        self.l21 = nn.Conv2d(64,128,3,padding=1)
        self.l22 = nn.Conv2d(128,128,3,padding=1)
        self.l31 = nn.Conv2d(128,256,3,padding=1)
        self.l32 = nn.Conv2d(256,256,3,padding=1)
        self.l41 = nn.Conv2d(256,128,3,padding=1)
        self.l42 = nn.Conv2d(128,128,3,padding=1)
        self.l51 = nn.Conv2d(128,64,3,padding=1)
        self.l52 = nn.Conv2d(64,64,3,padding=1)
        self.l53 = nn.Conv2d(64,1,1,padding=0)

        self.up1 = nn.ConvTranspose2d(256,128,2,2,padding=0, output_padding=0)
        self.up2 = nn.ConvTranspose2d(128,64,2,2,padding=0, output_padding=0)

    def forward(self, x):
        h11 = self.relu(self.l11(x))
        h12 = self.relu(self.l12(h11))
        h21 = self.relu(self.l21(self.maxpool(h12)))
        h22 = self.relu(self.l22(h21))
        h31 = self.relu(self.l31(self.maxpool(h22)))
        h32 = self.relu(self.l32(h31))

        h41 = self.relu(self.l41(torch.cat([h22, self.up1(h32)],dim=1)))
        h42 = self.relu(self.l42(h41))
        h51 = self.relu(self.l51(torch.cat([h12, self.up2(h42)],dim=1)))
        h52 = self.relu(self.l52(h51))

        return self.sigmoid(self.l53(h52))
   

class Dunet_2levels(nn.Module):
    def __init__(self):
        super().__init__()

        self.segmentator = Unet_2levels()
        self.refiner = Unet_2levels()

    def segment(self, x):
        return self.segmentator(x)

    def refine(self, x):
        return self.refiner(x)

    def forward(self, x):
        seg = self.segment(x)
        return seg, self.refine(seg)


class DVAE(nn.Module):
    def __init__(self, zdim):
        super().__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    stride=2,
                                    padding=0)

        self.enc_l1 = nn.Conv2d(1,64,3,stride=1,padding=1, dilation=1)
        self.enc_l2 = nn.Conv2d(64,64,3,stride=1,padding=1, dilation=1)
        self.enc_l3 = nn.Conv2d(64,256,3,stride=1,padding=1, dilation=1)
        self.enc_l4 = nn.Conv2d(256,256,3,stride=1,padding=1, dilation=1)
        self.enc_l51 = nn.Conv2d(256,zdim,1,stride=1,padding=0, dilation=1)
        self.enc_l52 = nn.Conv2d(256,zdim,1,stride=1,padding=0, dilation=1)

        self.dec_l1 = nn.ConvTranspose2d(zdim,256,4,stride=2, padding=1, output_padding=0)
        self.dec_l2 = nn.ConvTranspose2d(256,256,4,stride=2, padding=1, output_padding=0)
        self.dec_l3 = nn.ConvTranspose2d(256,64,4,stride=2, padding=1, output_padding=0)
        self.dec_l4 = nn.Conv2d(64,64,3,stride=1, padding=1, dilation=1)
        self.dec_l5 = nn.Conv2d(64,1,3,stride=1,padding=1, dilation=1)

    def encode(self, x):
        enc_h1 = self.relu(self.enc_l1(x))
        enc_h2 = self.relu(self.enc_l2(self.maxpool(enc_h1)))
        enc_h3 = self.relu(self.enc_l3(self.maxpool(enc_h2)))
        enc_h4 = self.relu(self.enc_l4(self.maxpool(enc_h3)))
        return self.enc_l51(enc_h4), self.enc_l52(enc_h4)
        
    def sample(self, mu, logvar, phase):
        if phase=='training':
            std = logvar.exp().sqrt()
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decode(self, z):
        dec_h1 = self.relu(self.dec_l1(z))
        dec_h2 = self.relu(self.dec_l2(dec_h1))
        dec_h3 = self.relu(self.dec_l3(dec_h2))
        dec_h4 = self.relu(self.dec_l4(dec_h3))
        return self.sigmoid(self.dec_l5(dec_h4))

    def forward(self, x, phase):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar, phase)
        ref = self.decode(z)
        return mu, logvar, z, ref


class DVAE_refiner(nn.Module):
    def __init__(self, zdim):
        super().__init__()

        self.segmentator = Unet_2levels()
        self.refiner = DVAE(zdim)

    def segment(self, x):
        return self.segmentator(x)

    def refine(self, x, phase):
        return self.refiner(x, phase)

    def forward(self, x, phase):
        seg = self.segment(x)
        mu, logvar, z, ref = self.refine(seg, phase)
        return seg, mu, logvar, z, ref
