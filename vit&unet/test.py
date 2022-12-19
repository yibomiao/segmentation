from networks import *
import torch
import torch.nn as nn

x=torch.zeros([4,768,14,14],dtype=torch.float)
bilinear = False
up1 = Up(768, 384 , bilinear)
y1  = up1(x)
print("y1 size(): ", y1.size())

up2 = Up(384, 192 , bilinear)
y2  = up2(y1)
print("y2 size(): ", y2.size())

up3 = Up(192, 96 , bilinear)
y3  = up3(y2)
print("y3 size(): ", y3.size())

up4 = Up(96, 48 , bilinear)
y4  = up4(y3)
print("y4 size(): ", y4.size())

outconv1 = OutConv(48,16)
y5 = outconv1(y4)
print("y5 size(): ", y5.size())


outconv2 = OutConv(16,48)
y6 = outconv2(y5)
print("y6 size(): ", y6.size())

down1 = Down(48, 96)
y7 = down1(y6)
print("y7 size(): ", y7.size())

down2 = Down(96, 192)
y8 = down2(y7)
print("y8 size(): ", y8.size())

down3 = Down(192,384)
y9 = down3(y8)
print("y9 size(): ", y9.size())

down4 = Down(384,768)
y10 = down4(y9)
print("y10 size(): ", y10.size())

outconv3 = OutConv(768,768)
y11 = outconv3(y10)
print("y11 size(): ", y11.size())