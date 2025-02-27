# ================== model ========================
embed_dims = 128  # 设置嵌入维度的大小为128
num_groups = 4  # 设置组的数量为4
num_decoder = 6  # 设置解码器的数量为6
num_single_frame_decoder = 1  # 设置单帧解码器的数量为1
use_deformable_func = True  # 使用可变形功能，需执行setup.py
num_levels = 4  # 设置特征金字塔的层数为4
drop_out = 0.1  # 设置dropout的比例为0.1
pc_range = [5.0, 1.20, 0, 60.0, 1.95, 2 * 3.1415926535]  # 设置点云的范围
scale_range = [0.1, 0.6]  # 设置缩放范围

include_opa = True  # 包含OPA模块
semantics = True  # 启用语义功能
semantic_dim = 17  # 设置语义维度为17
phi_activation = 'loop'  # 设置phi激活函数为'loop'
xyz_coordinate = 'polar'  # 设置坐标系为极坐标

model = dict(
    type="BEVSegmentor",  # 模型类型为BEVSegmentor
    img_backbone=dict(
        type="ResNet",  # 使用ResNet作为图像骨干网络
        depth=50,  # ResNet的深度为50
        num_stages=4,  # ResNet的阶段数为4
        frozen_stages=-1,  # 未冻结任何阶段
        norm_eval=False,  # 不在评估时冻结BN层
        style="pytorch",  # 使用pytorch风格
        with_cp=True,  # 启用检查点以节省内存
        out_indices=(0, 1, 2, 3),  # 输出的阶段索引
        norm_cfg=dict(type="BN", requires_grad=True),  # BN层的配置，启用梯度
        pretrained="ckpt/resnet50-19c8e357.pth",  # 预训练模型的路径
    ),
    img_neck=dict(
        type="FPN",  # 使用FPN作为图像颈部网络
        num_outs=num_levels,  # FPN的输出层数
        start_level=0,  # FPN的起始层
        out_channels=embed_dims,  # FPN的输出通道数
        add_extra_convs="on_output",  # 在输出上添加额外的卷积层
        relu_before_extra_convs=True,  # 在额外卷积前使用ReLU激活
        in_channels=[256, 512, 1024, 2048],  # 输入通道数
    ),
    lifter=dict(
        type='GaussianLifter',  # 使用GaussianLifter模块
        num_anchor=3600,  # 锚点数量为3600
        embed_dims=embed_dims,  # 嵌入维度
        anchor_grad=True,  # 启用锚点梯度
        feat_grad=False,  # 禁用特征梯度
        phi_activation=phi_activation,  # phi激活函数
        semantics=semantics,  # 启用语义
        semantic_dim=semantic_dim,  # 语义维度
        include_opa=include_opa,  # 包含OPA模块
    ),
    encoder=dict(
        type='GaussianOccEncoder',  # 使用GaussianOccEncoder模块
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',  # 使用SparseGaussian3DEncoder模块
            embed_dims=embed_dims,  # 嵌入维度
            include_opa=include_opa,  # 包含OPA模块
            semantics=semantics,  # 启用语义
            semantic_dim=semantic_dim  # 语义维度
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),  # 使用LN层归一化
        ffn=dict(
            type="AsymmetricFFN",  # 使用AsymmetricFFN模块
            in_channels=embed_dims * 2,  # 输入通道数
            pre_norm=dict(type="LN"),  # 前归一化使用LN
            embed_dims=embed_dims,  # 嵌入维度
            feedforward_channels=embed_dims * 4,  # 前馈通道数
            num_fcs=2,  # 全连接层数量
            ffn_drop=drop_out,  # FFN的dropout比例
            act_cfg=dict(type="ReLU", inplace=True),  # 激活函数配置
        ),
        deformable_model=dict(
            type='DeformableFeatureAggregation',  # 使用DeformableFeatureAggregation模块
            embed_dims=embed_dims,  # 嵌入维度
            num_groups=num_groups,  # 组的数量
            num_levels=num_levels,  # 层数
            num_cams=6,  # 相机数量
            attn_drop=0.15,  # 注意力dropout比例
            use_deformable_func=use_deformable_func,  # 使用可变形功能
            use_camera_embed=True,  # 使用相机嵌入
            residual_mode="cat",  # 残差模式为'cat'
            kps_generator=dict(
                type="SparseGaussian3DKeyPointsGenerator",  # 使用SparseGaussian3DKeyPointsGenerator模块
                embed_dims=embed_dims,  # 嵌入维度
                phi_activation=phi_activation,  # phi激活函数
                xyz_coordinate=xyz_coordinate,  # 坐标系
                num_learnable_pts=6,  # 可学习点的数量
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],  # 固定缩放比例
                pc_range=pc_range,  # 点云范围
                scale_range=scale_range  # 缩放范围
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',  # 使用SparseGaussian3DRefinementModule模块
            embed_dims=embed_dims,  # 嵌入维度
            pc_range=pc_range,  # 点云范围
            scale_range=scale_range,  # 缩放范围
            restrict_xyz=False,  # 不限制xyz
            unit_xyz=None,  # 单位xyz
            refine_manual=None,  # 手动细化参数
            phi_activation=phi_activation,  # phi激活函数
            semantics=semantics,  # 启用语义
            semantic_dim=semantic_dim,  # 语义维度
            include_opa=include_opa,  # 包含OPA模块
            xyz_coordinate=xyz_coordinate,  # 坐标系
            semantics_activation='softmax',  # 语义激活函数为softmax
        ),
        spconv_layer=None,  # 不使用spconv层
        num_decoder=num_decoder,  # 解码器数量
        num_single_frame_decoder=num_single_frame_decoder,  # 单帧解码器数量
        operation_order=None,  # 操作顺序
    ),
    head=dict(
        type='GaussianHead',  # 使用GaussianHead模块
        apply_loss_type=None,  # 不应用特定损失类型
        num_classes=17,  # 类别数量
        empty_args=None,  # 空参数
        with_empty=False,  # 不包含空类别
        cuda_kwargs=None,  # CUDA参数
    )
)