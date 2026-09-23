import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import posenc,NeRF

torch.set_default_dtype(torch.float32)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class HyperParams:

    epoch=10000
    i_plot=100
    N_samples=64
    lr=5e-5
    plot_img=True


def get_rays(H:int,W:int,focal:Optional[float],c2w:Optional[torch.Tensor]):

    tensor1=torch.arange(0,H,dtype=torch.float32).to(device)
    tensor2=torch.arange(0,W,dtype=torch.float32).to(device)

    i,j=torch.meshgrid(tensor1,tensor2,indexing="ij")
    i=i.transpose(-2,-1).to(device)
    j=j.transpose(-2,-1).to(device)

    # (H,W,3)
    dir_c=torch.stack([(i-W*0.5)/focal,-(j-H*0.5)/focal,-torch.ones_like(i)],dim=-1).to(device)

    # (H,W,1,3) * (3,3)-->(H,W,3,3) * (3,3)-->(H,W,3,1)-->(H,W,3)
    dir_w=torch.sum(dir_c[...,None,:]*c2w[:3,:3],dim=-1,keepdim=False).to(device)
    ray_o=c2w[:3,-1].expand(*dir_w.shape)

    return dir_w,ray_o

def batchify(ffn,chunk=1024*32):
    return lambda inputs:  torch.cat([ffn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], dim=0)

"""
在 NeRF 中，我们要算的是“光线到达当前点 i 之前，没有被前面 i-1 个点遮挡的概率”。所以计算第i个点的透过率时,不能把第i个点自身的不透明度乘进去。

alpha表示物体的透明度,亦可用于透过率
"""
def exclusive_cumprod(tensor:torch.Tensor):
    tensor=torch.cumprod(tensor,dim=-1)
    tensor1=torch.roll(tensor,shift=1,dims=-1)
    tensor1[...,0]=1.0
    return tensor1

def render_ray(ffn:NeRF,near,far,N_samples,dir_w,ray_o):

    assert far>near,"Wrong order of arg far and arg near! "
    assert (far-near)%N_samples==0,"U need to select correct args"

    # (N_samples,)
    z_vals=torch.linspace(near,far,N_samples).to(device)
    # (H,W,1,3)*(N_samples,)-->(H,W,N_samples,3)
    points=ray_o+dir_w[...,None,:]*z_vals

    # (H,W,N_samples,3)-->(-1,3)
    points_flt=points[-1,3]
    # (-1,3)-->(-1,4)
    raw_data=batchify(ffn)(points_flt)
    # (-1,4)-->(H,W,N_samples,4)
    raw_data=raw_data.reshape([*dir_w[:3],4])

    # (H,W,N_samples,3)
    rgb=F.sigmoid(raw_data[...,:3])
    # (H,W,N_samples,1)-->(H,W,N_samples)
    sigma=F.relu(raw_data[...,3])

    # (N_samples,)
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.tensor([1e10], device=device).expand(z_vals[..., :1].shape)], dim=-1)
    # (H,W,N_samples)
    alpha = 1.0 - torch.exp(-sigma * dists)
    # (H,W,N_samples)
    weights = alpha * exclusive_cumprod(1.0-alpha + 1e-10)

    # (H,W,N_samples,1) * (H,W,N_samples,3)-->(H,W,N_samples,3)-->(H,W,3)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map,depth_map,acc_map

"""
我们来区分两句代码：

一个是位于函数exclusive_cumprod中的:

    tensor1[...,0]=1.0

另一个是位于函数render_ray中的:

    sigma=F.relu(raw_data[...,3])

    
提取（等号右侧）= 降维：例如 x = raw[..., 3]。操作会抽离特定索引的数据，生成全新的张量，被提取的那一维会直接消失。

赋值（等号左侧）= 形状不变：例如 tensor[..., 0] = 1.0。操作是精准定位到原张量内存位置进行原地修改(In-place)，原容器的总维度和形状完好无损。

提取属于浅拷贝,而赋值是原地操作。原地操作的切片不会降维

"""

def train(params:HyperParams,data_filename:str):

    data=np.load(data_filename)
    imgs=data["images"]
    poses=data["poses"]
    focal=data["focal"]

    # (Batch,H,W,C)
    imgs=torch.from_numpy(imgs).to(device)
    # (B,row,column)
    poses=torch.from_numpy(poses).to(device)
    # (1,)
    focal=torch.from_numpy(focal).to(device)

    # (H,W,C)
    val_img=imgs[105,...]
    # (row,column)
    val_pose=poses[105,...]

    train_imgs=imgs[:105,...]
    train_poses=poses[:105,...]

    psnrs = []
    iternums = []

    model=NeRF()
    model=model.to(device)
    optimizor=optim.Adam(model.parameters(),lr=params.lr)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizor,params.epoch,1e-6)

    

    for i in tqdm(range(params.epoch)):

        idx=np.random.randint(train_imgs.shape[0])
        # (H,W,3)
        input_img=train_imgs[idx,...]
        input_pose=train_poses[idx,...]

        H,W,_=input_img.shape
        dir_w,ray_o=get_rays(H,W,focal.item(),input_pose)
        rgb_pred,depth_pred,acc_pred=render_ray(model,2,6,params.N_samples,dir_w,ray_o)

        loss=F.mse_loss(rgb_pred,input_img)
        loss.backward()
        optimizor.zero_grad()
        optimizor.step()
        scheduler.step()

    if params.plot_img:
        if i % 100 == 0:
            with torch.no_grad():
                dir_w_pred,ray_o_pred=get_rays(H,W,focal.item(),val_pose)
                rgb,depth,acc=render_ray(model,2,6,params.N_samples,dir_w_pred,ray_o)
                psnr = -10.0 * torch.log10(loss)
                
                psnrs.append(psnr.item())
                iternums.append(i)

                plt.figure(figsize=(12, 4))
                plt.subplot(131)
                plt.imshow(rgb.cpu().detach().numpy())
                plt.title(f"Iteration {i}")
                plt.subplot(132)
                plt.plot(iternums, psnrs)
                plt.title("PSNR")
                plt.subplot(133)
                plt.imshow(depth.cpu().detach().numpy(), cmap="gray")
                plt.title("Depth Map")

                # Auto close
                plt.show(block=False)
                plt.pause(1)
                plt.close()

    print("All Fine")

if __name__=='__main__':

    params=HyperParams()
    train(params,'tiny_nerf_data.npz')


