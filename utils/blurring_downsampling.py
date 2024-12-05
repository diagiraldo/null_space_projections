# D. Giraldo 
# July, 2024

import torch
import torch.nn as nn
from scipy.signal import windows

import math
import numpy as np

def box(thickness, pad = None, spacing = None):
    if not pad: pad = np.ceil(thickness/2.)
    if not spacing: spacing = thickness
    
    # init kernel with appropriate size
    bk = np.zeros(int(spacing) + 2*int(pad))
    
    # blurring kernel according to slice thickness 
    l = np.ceil(thickness) + 2*pad 
    x = np.abs(np.arange(0, l) - (l-1)/2) 
    y = np.piecewise(
        x, 
        [x <= thickness/2.,
         x > thickness/2.], 
        [1, 
         0]
    )
    
    # 
    bk[:y.shape[0]] = y
        
    return bk

def smoothed_box(thickness, pad = None, spacing = None):
    if not pad: pad = np.ceil(thickness/2)
    if not spacing: spacing = thickness
        
    # init kernel with appropriate size
    bk = np.zeros(int(spacing) + 2*int(pad))
    
    # blurring kernel according to slice thickness 
    l = np.ceil(thickness) + 2*pad 
    x = np.abs(np.arange(0, l) - (l-1)/2) 
    y = np.piecewise(
        x, 
        [x <= thickness/3, 
         (x > thickness/3) * (x <= thickness*2/3),
         x > thickness*2/3], 
        [1, 
         lambda x: 0.5 - 0.5 * np.sin(3*np.pi*((x/thickness)-0.5)),
         0]
    )
    
    # 
    bk[:y.shape[0]] = y
        
    return bk

# Blurring & Donwsampling
class BlurringDownsampling(nn.Module):
    def __init__(
        self,
        slice_thickness, #in pixel/voxel units: INTEGER
        slice_spacing, #in pixel/voxel units: INTEGER
        dimension='2d',
        slice_model = "box",
        normalize_kernel = True,
        dtype=torch.float32,
    ):
        super(BlurringDownsampling, self).__init__()
        
        self.dimension = dimension
        self.normalize_kernel = normalize_kernel
        
        if slice_model == "box":
            padding = int(np.ceil(slice_thickness/2))
            blur_kernel = box(thickness = slice_thickness, pad = padding, spacing = slice_spacing)

        elif slice_model == "smoothed-box":
            padding = int(np.ceil(slice_thickness/2))
            blur_kernel = smoothed_box(thickness = slice_thickness, pad = padding, spacing = slice_spacing)
            
        elif slice_model == "gaussian":
            fwhm = slice_thickness
            sd = fwhm / (2 * math.sqrt(2 * math.log(2)))
            padding = int(np.ceil(slice_thickness/2))
            kernel_size = np.ceil(fwhm) + 2*padding + np.round(slice_spacing - slice_thickness)          
            blur_kernel = windows.gaussian(kernel_size, std = sd, sym = True)
        else:
            raise NotImplementedError

        self.integral_blur_kernel = blur_kernel.sum()#.item()?

        blur_kernel = torch.tensor(blur_kernel, dtype=dtype, requires_grad = False).squeeze()

        if self.dimension == '2d':
            blur_kernel = blur_kernel[None, :]
            self.stride = (1, int(slice_spacing))
            self.padding = (0, padding)
            
        elif self.dimension == '3d':
            blur_kernel = blur_kernel[None, None, :]
            self.stride = (1,1, int(slice_spacing))            
            self.padding = (0, 0, padding)
        else:
            raise ValueError('dimension not valid')
        
        self.blur_kernel = nn.Parameter(
            blur_kernel.unsqueeze(0).unsqueeze(1),
            requires_grad = False,
        )

    def forward(self, x):
        
        if self.normalize_kernel: 
            k = self.blur_kernel/self.integral_blur_kernel
        else:
            k = self.blur_kernel

        if self.dimension == '2d':
            n_channels = x.shape[1]
            if n_channels == 1:
                y = nn.functional.conv2d(x, k.detach(), bias=None, stride=self.stride, padding=self.padding)
            else:
                y = nn.functional.conv2d(
                    x.view(-1,1,x.shape[2],x.shape[3]),
                    k, 
                    bias=None, 
                    stride=self.stride, 
                    padding=self.padding
                ).view(x.shape[0], n_channels, x.shape[2], -1)

        if self.dimension == '3d':
            y = nn.functional.conv3d(x, k, bias=None, stride=self.stride, padding=self.padding)
            
        # if self.normalize_kernel: 
        #     y = y/self.integral_blur_kernel

        return y

    def transpose(self, x):
        
        if self.normalize_kernel: 
            #x = x*self.integral_blur_kernel
            kt = self.blur_kernel/self.integral_blur_kernel
        else:
            kt = self.blur_kernel

        if self.dimension == '2d':
            n_channels = x.shape[1]
            if n_channels == 1:
                y = nn.functional.conv_transpose2d(x, kt, bias=None, stride=self.stride, padding=self.padding)
            else:
                y = nn.functional.conv_transpose2d(
                    x.view(-1,1,x.shape[2],x.shape[3]),
                    kt, 
                    bias=None, 
                    stride=self.stride, 
                    padding=self.padding
                ).view(x.shape[0], n_channels, x.shape[2], -1)

        if self.dimension == '3d':
            y = nn.functional.conv_transpose3d(x, kt, bias=None, stride=self.stride, padding=self.padding)
            
        # if self.normalize_kernel: 
        #     y = y/self.integral_blur_kernel

        return y