# Blender格式数据加载实现，对应于书中的3.2.3 数据加载小节

import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


# 平移计算的lambda方法
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# 绕y轴旋转的lambda方法
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

# 绕x轴旋转的lambda方法
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    """
    以输入的平移(radius)、旋转(theta, phi)参数构建相机姿态(c2w)
    """
    # 1. z转平移radius
    c2w = trans_t(radius)
    # 2. 绕y轴旋转phi角度
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    # 3. 绕x轴旋转theta角度
    c2w = rot_theta(theta/180.*np.pi) @ c2w

    # 4. 输出c2w位姿矩阵
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    """
    Blender格式数据加载主方法
    """
    splits = ['train', 'val', 'test'] # 将数据分为train, val, test三类，对于显存比较紧张的情况，可以考虑修改这里，只加载train和val [TODO]
    metas = {} # metas中包括了数据的所有内容，数据来源为Blender数据集中transforms_train.json, transforms_val.json和transforms_test.json
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []  # 用于存储所有的图片
    all_poses = []  # 用于存储所有图片的pose与all_imgs一一对应
    counts = [0]

    for s in splits:
        # 对于train/val/test三种不同类型的数据进行分别的解析
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        # 加载每一帧图像数据以及pose
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        # 图像数据从0-255，归一化到0-1的float32，便于后续的计算
        imgs = (np.array(imgs) / 255.).astype(np.float32) # 暂时保持所有通道的数据（RGBA），后续在训练时，只使用RGB进行训练，处理过程在[TODO]
        poses = np.array(poses).astype(np.float32)

        # 记录数据类型存储在位置
        counts.append(counts[-1] + imgs.shape[0])

        # 合并数据
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    # 获取每个部分图像在合并数据中的区间
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    # 将所有加载到的图像、pose都合并在同一个数组中。
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    # 获得图片的宽高，以及camera_angle_x
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])

    # 计算焦距，参考书中63页，公式3.3
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    # 为渲染时生成所有目标的pose，用于进行效果呈现。
    # 生成的结果为一个360度的环形渲染轨迹（每40度一个位姿），从而渲染出一个围绕着中心的360环形视频
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    # 半分辨率训练方法，会将所有数据的宽、高都缩减为一半，这样会提升训练速度、降低显存消耗，可以通过在run_nerf.py执行时加入--half_res 参数启用该逻辑
    # 但这样也会因为训练源分辨率下降，导致训练的质量下降。
    if half_res:
        # 宽、高、焦距都降低为一半
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))  # 临时图片数据，在缩放结束后被销毁

        # 缩放方法，插值算法使用INTER_AREA，也即图片被缩放面积的平均像素值作为缩放后图像的像素值，也可以替换成其他的opencv或自己实现的算法代替
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        imgs = imgs_half_res # 替代原始的图像数据，后续训练使用半分辨率计算

        # 原始nerf参考代码的实现方式，使用tensorflow进行图片缩放，但在pytorch框架下使用了opencv实现缩放，效果一致，仅在缩放filter上有差异。
        # 对应于https://github.com/bmild/nerf/blob/master/load_blender.py，line 82
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    # 返回加载好的图像、pose、渲染的pose、宽高、焦距以及各类型数据的split位置点。
    return imgs, poses, render_poses, [H, W, focal], i_split


