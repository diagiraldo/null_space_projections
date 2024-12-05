# Diana Giraldo 
# Last modification: Dec 2024

import torch


# Wilson-Barret method to aproximate projection onto imaging operator null space
# See:
# - D.W. Wilson and H.H. Barrett, "Decomposition of images and objects into measurement and null components," Opt. Express 2, 254-260 (1998)
# - Joseph Kuo, Jason Granstedt, Umberto Villa, and Mark A. Anastasio, "Computing a projection operator onto the null space of a linear imaging operator: tutorial," J. Opt. Soc. Am. A 39, 470-481 (2022)

def wilson_barret(
    M,
    x,
    step = 1, 
    tolerance = 0.01, 
    max_iter = 20,
    print_info = False
):
    iteration = 0
    f = x
    r = M(f)
    
    if print_info: print(torch.linalg.norm(r))
    
    while torch.linalg.norm(r) >= tolerance and iteration <= max_iter:
        iteration += 1
        f = f - step * M.transpose(r)
        r = M(f)
        if print_info: print(torch.linalg.norm(r))
        
    return f