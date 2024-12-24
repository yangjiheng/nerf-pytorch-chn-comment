# NeRF重要的工具方法

import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# MSE, PSNR的计算方法，对应书中2.4节
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

# 从0-1的float类型，转换到8位0-255的unsigned int类型方法
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# 位置编码部分实现，原文的5.1节，书中的3.2.5小节
class Embedder:
    # 位置编码类的初始化方法
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    # 位置编码主要函数的初始化
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0

        # 编码结果是否包括输入
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        # 最大编码频率，与频率采样间隔
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        # 生成所有的频段
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        # 对每个频段生成所有的位置编码方法，方法类型使用构造时的periodic_fns即可。
        # 位置编码的本质是捕捉信号里的高频信息，并在学习过程中对高频信息得以保持，因此所有有助于保持高频信号的位置编码方法都可以被尝试。
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        # 位置编码的两个核心成员：embed_fns: 位置编码方法，out_dim: 输出的维度
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    # 对inputs实施位置编码，输出编码后结果
    def embed(self, inputs):
        # 对每一个输入，按设定的embed_fns进行编码，并将所有的编码结果连接在一起作为结果。
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    """
    位置编码帮助方法
    """
    if i == -1:
        return nn.Identity(), 3
    
    # 构造位置编码类初始化参数
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos], # 以原文为例，使用sin, cos进行位置编码。据作者讲，这个是尝试了大量方法后，效果最好的一个
    }
    
    # 构造位置编码工具对象
    embedder_obj = Embedder(**embed_kwargs)
    # 执行位置编码操作
    embed = lambda x, eo=embedder_obj : eo.embed(x)

    # 返回位置编码结果与输出的维度
    return embed, embedder_obj.out_dim


class NeRF(nn.Module):
    """
    NeRF对象类，书中3.2.6节对应
    """
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()

        # self.D, self.W分别为MLP的深度和宽度
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        # Skip connection所在的层数
        self.skips = skips

        # 模型中是否使用方向进行预测
        self.use_viewdirs = use_viewdirs
        
        # 线性预测层，NeRF MLP中前面的部分
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        # 参考实现中的网络设计与论文中略有差异，本实现以参考实现为准。
        # 说明在此: https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105
        # 这种情况很常见，不用在意，选择结果较好的那种
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### 如果遵循原文设计的话，应该是以下的实现方式
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        # NeRF MLP中后面的部分
        # 使用或不使用方向信息进行推理的网络结构是有差异的，可以参考书中3.2.6节中的介绍，或是论文中附录里的说明。
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """
        模型的推理方法
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        # 对每一个pts_linears进行推理，在原始论文中，使用了8个线性全连接层，每层宽度为256，且都使用ReLU进行激活
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # Skip Connection机器，对于标记了skip connection的层，加入输入特征继续训练。原文中在第5层中加入了skip connection
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # 如果使用方向，则在8层网络后，1. 推理得到alpha，2. 合并方向特征并再输入到一层宽度为256的线性层，推理得到输出
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            # 经过view_linears层，并使用ReLU进行激活，原文是一个128宽度的线性层
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            # 推理得到RGB值，并将RGB和Alpha一并输出
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            # 如果不使用方向，则直接推理得出RGB和alpha
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        """
        权重加载方法
        """
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # 从模型中加载pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # 从模型中加载feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # 从模型中加载views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # 从模型中加载rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # 从模型中加载alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# 射线相关工具方法，对应书中3.2.4小节
# get_rays方法与get_rays_np方法功能一致，get_rays方法使用了pytorch实现，get_rays_np使用了numpy实现

# 1. get_rays的pytorch实现
def get_rays(H, W, K, c2w):
    # 创建宽为W，高为H的meshgrid，用来以此为成像面的射线集合
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()

    # 计算相机空间各射线方向
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # 将射线方向从相机空间转换到世界空间
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # 点积计算，等同于：[c2w.dot(dir) for dir in dirs]
    # 将相机的起始点转换到世界空间，作为所有生成射线的起始点
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


# 1. get_rays的numpy实现
def get_rays_np(H, W, K, c2w):
    # 创建宽为W，高为H的meshgrid，用来以此为成像面的射线集合
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')

    # 计算相机空间各射线方向
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)

    # 将射线方向从相机空间转换到世界空间
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # 点积计算，等同于：[c2w.dot(dir) for dir in dirs]
    # 将相机的起始点转换到世界空间，作为所有生成射线的起始点
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    NDC空间射线工具方法，对应小节2.2.1中的讲述，适用于forward-facing场景的重建问题。
    """
    # 将射线起点，转换到近平面端
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # 对射线起点位置进行投影
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    # 计算光线的投影
    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    # 堆叠计算结果，构造射线的起始点和射线方向
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    分层采样方法，原文5.2节，书中对应3.2.7小节
    根据权重分布，添加N_samples个采样点，完成分层采样
    """
    # 计算PDF
    weights = weights + 1e-5 # 防止weights为0时，被除后得到nan，加上一个极小值来防御。
    pdf = weights / torch.sum(weights, -1, keepdim=True) # 计算Probability Density Function (PDF_), 例如: [0.3, 0.2, 0.5]
    cdf = torch.cumsum(pdf, -1) # 使用cumsum方法，得到Cumulative Density Function (CDF), 例如将上面的PDF生成为：[0.3, 0.5, 1.0]
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # 在CDF开头增加一个值为0的tensor

    # 均匀采样或随机采样
    if det:
        # N_samples个线性采样
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        # N_samples个随机采样
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # 测试时，为了保证数值的确定性，使用np的随机数进行覆盖，以保证结果的可复现性。
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    u = u.contiguous()

    # 为每个u的值生成它所在CDF中的上限和下限值，并保存在inds_g数组中
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2), 每个batch中每个sample，一个下限，一个上限两个数据

    # 按目标采样点数生成新的cdf_g和bins_g数组，根据粗糙网络生成的权重生成新的采样点
    # 原始tensorflow实现，使用了gather方法，但在pytorch中，需要计算匹配的shape，因此稍调整
    # 因此以下两行代码是原TensorFlow参考实现，而后面的三行是pytorch中对应的等效实现
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # 插值生成最终采样点的位置
    # denom表示CDF上下界的差
    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom # u在CDF区间内的相对位置
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0]) # 坐标计算

    return samples
