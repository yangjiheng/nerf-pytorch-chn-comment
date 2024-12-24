# NeRF训练主文件

import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


# 初始化device，检查是否有CUDA支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """
    对更小的batch使用fn进行推理
    """
    if chunk is None:
        return fn
    def ret(inputs):
        # 对当前batch进行推理
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """
    准备数据，使用网络推理
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    # 对输入进行位置编码
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        # 对方向进行位置编码
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # 使用网络进行推理并输出结果
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """
    为了避免OOM，将射线渲染过程划分成多个minibatches
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        # 渲染一个minibatch
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """
    射线的渲染方法
    输入参数：
      H: int. 每张图的高度像素数
      W: int. 每张图的宽度像素数
      focal: float. 针孔相机的焦距
      chunk: int. 每次最多可并行处理的射线数。用于控制最大的内存消耗，调整本值只影响速度和内存消耗量，不影响渲染结果
      rays: [2, batch_size, 3]. batch中射线的起点的方向
      c2w: [3, 4]. 相机空间到世界空间的变换矩阵
      ndc: bool. 如果是True，则使用NDC坐标表达起点和方向
      near: [batch_size]，射线近平面位置
      far: [batch_size]，射线远平面位置
      use_viewdirs: bool. 如果是Ture，则在模型中使用观察方向
      c2w_staticcam: [3, 4]. 如果非None，则使用这个矩阵为相机矩阵，对观察方向使用c2w参数
    返回:
      rgb_map: [batch_size, 3]. 每条射线预测的RGB值
      disp_map: [batch_size]. disparity_map，深度度的逆
      acc_map: [batch_size]. accumulation map
      extras: 一个字典对象，由render_rays返回
    """

    # 初始化所有射线
    if c2w is not None:
        # 渲染整图
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # 使用参数中的射线进行渲染
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]

    # 对forward-facing场景(如LLFF)，使用NDC坐标系
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # 创建ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    # 对每第射线计算近点、远点
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # 渲染所有的射线
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    """
    渲染过程，对应书中3.2.11小节
    """
    H, W, focal = hwf

    if render_factor!=0:
        # 下采样渲染，如果被指定，则渲染更小分辨率的图像，以达到更快的渲染速度。
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    # 渲染结果RGB与disparity map
    rgbs = []
    disps = []

    t = time.time()
    # 渲染所有的pose
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        # 调用渲染方法，得到RGB, disparity和acumulation map
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        # 保存渲染的RGB图像
        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    # 堆叠渲染结果, 用于返回
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """
    初始化nerf model，以及训练中即将用到的训练数据结构
    """

    # 初始化位置编码
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    # 初始化NeRF model
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    # 初始化精细NeRF model
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # 初始化优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # 加载checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # 加载模型
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC坐标对LLFF的forward-facing场景效果好，因此对错误设置进行修正
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """
    体渲计算部分，对应书中3.2.8小节，并将模型的预测结果渲染为语义上有意义的图像数据
    输入参数：
        raw: [num_rays, num_samples along ray, 4]. 模型的预测结果
        z_vals: [num_rays, num_samples along ray]. 积分时间
        rays_d: [num_rays, 3]. 每条射线的方向
    返回:
        rgb_map: [num_rays, 3]. 射线预测的RGB渲染结果
        disp_map: [num_rays]. Disparity map，为深度度的逆
        acc_map: [num_rays]. 每条射线的权重累积
        weights: [num_rays, num_samples]. 每个sample的权重
        depth_map: [num_rays]. 到物体的预测距离
    """

    # 祼数据到alpha的lambda计算，使用ReLU进行激活
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    # RGB数据的激活转化
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    # 如果设置了raw_noise_std，则按标准差施加随机噪声
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # 代码中所有的pytest都是为了让噪声有确定性，方便调试，复现数据
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # alpha数据的激活转化
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

    # 体渲染公式渲染weights
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    # 获得rgb_map, depth_map, disp_map, acc_map
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    # 如果设置了白色背景，则RGB与白色进行alpha blending
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """.
    渲染过程，对应书中3.2.9小节
    输入参数：
      ray_batch: [batch_size, ...]. 对于每条射线采样所有有用的信息，包括起始点、射线方向、最小距离、最大距离和单位观察方向 
      network_fn: function，用于预测每个点的RGB和密度的模型
      network_query_fn: 用二向network_fn传递查询的方法
      N_samples，int, 每条射线采样点个数
      retraw: bool, 如果是True，则输出模型的原始、未处理数据
      lindisp：bool，如果是True，则深度逆上进行线性采样，而非在深度上线性采样
      perturb: float，0或1，如果非0，则对采样点位置进行随机扰动
      N_importance: int，在分层采样时，对每条射线上添加的采样点个数
      network_fine: 精细网络 [TODO]
      white_bkgd: bool，如果是True，则假定白色渲染背景
      raw_noise_std: ...
      verbose: bool，如果是True，则打印更多的运行细节
    返回:
      rgb_map: [num_rays, 3]. 使用精细模型预测得到一条射线预测的RGB值
      disp_map: [num_rays]. Disparity map. 为深度图的逆，即1 / depth
      acc_map: [num_rays]. 使用精细模型得到的每条射线的累计不透明度
      raw: [num_rays, num_samples, 4]. 模型预测的原始结果
      rgb0: [num_rays, 3]. 粗糙模型的预测结果
      disp0: [num_rays]. 粗糙模型的disparity map
      acc0: [num_rays]，粗糙模型的accumulation map
      z_std: [num_rays]. 每个样本沿射线方向距离的标准差
    """

    # N_rays指当前batch中射线的数量
    N_rays = ray_batch.shape[0]

    # 所有射线的源、方向
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None

    # 近平面、远平面
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    # 从近平面到远平面采样N_samples个
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    # 如果设置了perturb，则对位置进行随机扰动
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    # 所有的采样点坐标
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    # 粗糙阶段，使用采样点，查询网络获得输出，并转换为rgb_map, disp_map, acc_map, weights, depth_map
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    # 如果做精细阶段，则继续使用精细网络进行渲染
    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # 使用sample_pdf基于PDF进行分层采样
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        # 精细层的采样点计算
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        # 精细阶段，使用采样点，查询网络获得输出，并转换为rgb_map, disp_map, acc_map, weights, depth_map
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}

    # 如果要求返回祼数据，则在字典中带入
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    # 容错处理
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    """
    程序运行配置
    """
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, 
                        help='config file path') # 运行配置路径，比如./config/lego.conf
    parser.add_argument("--expname", type=str, 
                        help='experiment name') # 当前训练任务的名字
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs') # 默认存储checkpoint和log的路径
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory') # 默认输入数据的路径

    # 训练设置
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')  # MLP网络的层数
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')  # MLP网络的宽度
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network') # MLP精细层网络的层数
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network') # MLP精细层网络的宽度
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)') # 每次训练过程中的batch大小
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate') # 学习率设置
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)') # 学习率下调率
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory') # 每次训练并行的射线数量，如果显存不足，则可以设置更小的chunk
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory') # 每次并行送给network的pts量，如果显存不足，则可以设置更小的netchunk
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time') # 仅从一张图像中采样，而不从所有图像生成的射线采样
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt') # 不从checkpoint加载权重
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network') # 粗糙网络的numpy权重文件

    # 渲染设置
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray') # 在粗糙采样阶段每条光线上的采样个数
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray') # 在精细采样阶段，每条光线上多采样的点数
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter') # 采样时为了提升训练数据多样性所施加的随机位置扰动
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D') # 训练中是否使用5D输入（3D位置与2D方向），还是仅3D位置输入
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none') # 是否使用位置编码
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)') # 指定位置编码时使用的频段数
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)') # 对方向信息进行位置编码时使用的频段数
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended') # 随机噪声的标准差

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path') # 是否仅渲染，不训练
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path') # 渲染测试pose
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview') # 渲染下采样因子，渲染分辨率低，则速度快，用于快速预览

    # 当仅从单张图像采样时的训练设置
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')  # 前多少次epoch时，需要使用图像中心区域进行训练
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') # 取图像中心训练时，保留图像多大比例

    # 数据集设置
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels') # 数据集类型
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels') # 选择数据集中多大比例作为test/val集

    ## 数据集设置 - DeepVoxels数据集的flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase') # 要加载的DeepVoxels场景类型

    ## 数据集设置 - Blender数据集的flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)') # 渲染时是否使用白色背景
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800') # 是否使用半分辨率进行训练，参考load_blender.py

    ## 数据集设置 - LLFF数据集的flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images') # LLFF数据集下采样因子
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)') # 不使用NDC，LLFF数据集使用NDC效果更好
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth') # 采样时，是用disparity (1/depth)还是depth
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes') # 渲染360度场景
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8') # 从LLFF数据集中提取test set的步长，原文中使用了8，也即1/8的图像为test set

    # 日志与记录设置
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin') # 打印训练日志的频率设置，默认为100个epoch一次输出
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging') # Tensorboard日志记录频率，默认每500个epoch保存一次
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving') # 存储checkpoint权重的频率，默认为10000个epoch一次输出
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving') # 测试pose渲染结果保存频率，默认为每50000个epoch保存一次
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving') # 存储渲染视频的频率，默认每50000个epoch保存一次

    return parser

def train():
    """
    核心训练过程，对应书中3.2.10小节
    """
    parser = config_parser()
    args = parser.parse_args()

    # 加载训练数据集
    K = None
    if args.dataset_type == 'llff':
        # 调用load_llff.py中的加载方法，加载LLFF格式数据集
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        # 如果指定llffhold，则从数据集中按步长llffhold选为test/val set
        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            # 非NDC时，近平面为边界的0.9倍，远平面为边界的1.0倍，这样实际上近景比边界数据会更大一些，变相扩大了取景的范围
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            # NDC时，近平面为0，远平面为1
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        # 调用load_blender.py中的加载方法，加载blender格式数据集
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        # 设置近平面、远平面位置
        near = 2.
        far = 6.

        # 如果要求渲染白色背景，如输入图像为RGBA，则使用alpha blending，否则直接使用RGB通道
        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        # 调用load_linemod.py中的加载方法，加载linemod格式数据集
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        # 如果要求渲染白色背景，如输入图像为RGBA，则使用alpha blending，否则直接使用RGB通道
        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':
        # 调用load_deepvoxels.py中的加载方法，加载deepvoxels格式数据集
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # 将相机内参转换为正确的数据格式
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # 定义内参矩阵
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # 如果目标为仅渲染，则构造render_poses
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # 创建工作目录等
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # 创建NeRF模型
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # 将待渲染位姿上传device（一般为CUDA，CPU非常慢）
    render_poses = torch.Tensor(render_poses).to(device)

    # 如果设置了仅渲染，则只使用当前模型渲染结果，并退出
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # 准备训练的batch
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # 生成所有的射线batching
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # 仅使用数据集中train set图像
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        # 射线随机打乱，提升训练的多样化
        np.random.shuffle(rays_rgb)

        i_batch = 0

    # 将训练数据(images, poses) 上传到device（一般为CUDA，CPU非常慢）
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    # 总训练次数设定为200000次
    N_iters = 200000 + 1

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    # 开始训练
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # 训练采样过程
        if use_batching:
            # 从所有图像中随机采样N_rand个射线
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            # 重shuffle射线
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # 每次仅从同一张随机图像中进行采样
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                # 从单图中生成射线
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                # 对于前precrop_iters次训练，从图像中选择中间的一个部分进行训练，提升训练效率和质量，会使模型更关注图像中心区域的物体
                if i < args.precrop_iters:
                    # 只取中心precrop_frac比例大小的图像进行训练
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    # 使用完整图像进行训练
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                # 重新计算射线的origin和direction
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  最核心的渲染过程，NeRF的核心 #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        # 计算损失
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        # 误差反向传播并更新优化器
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   更新学习率过程  ###
        # 在训练后期逐渐降低学习率的变化速度有利于提升模型的收敛速度、并更好地适应数据集的特性。
        # 相关理论知识可以参考机器学习中学习率的设计思路
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # 余下的部分是日志、保存模型、保存渲染结果等等记录过程。

        # 每i_weights个epoch，保存一次checkpoint
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # 每i_video个epoch，保存一次渲染视频结果
        if i%args.i_video==0 and i > 0:
            # 按render_poses进行渲染
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            # 渲染图像和disparity map都渲染到视频中
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        # 每i_testset，在测试pose上渲染一次效果
        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            # 渲染test pose
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


        # 每i_print个epoch，打印一次当前训练的Loss, PSNR
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


# 主程序启动函数
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # 训练函数入口
    train()
