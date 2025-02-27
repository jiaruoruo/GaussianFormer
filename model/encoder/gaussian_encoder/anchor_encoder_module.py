from mmengine.model import BaseModule
from .utils import linear_relu_ln
import torch.nn as nn, torch
from mmengine.model import BaseModule  # 从mmengine.model导入BaseModule基类
from .utils import linear_relu_ln  # 从当前模块的utils中导入linear_relu_ln函数
import torch.nn as nn, torch  # 导入torch.nn模块和torch库


@MODELS.register_module()
class SparseGaussian3DEncoder(BaseModule):
    def __init__(
@MODELS.register_module()  # 使用MODELS注册模块装饰器注册类
class SparseGaussian3DEncoder(BaseModule):  # 定义SparseGaussian3DEncoder类，继承自BaseModule
    def __init__(  # 初始化函数
        self,
        embed_dims: int = 256,
        include_opa=True,
        semantics=False,
        semantic_dim=None
        embed_dims: int = 256,  # 嵌入维度，默认值为256
        include_opa=True,  # 是否包含不透明度，默认值为True
        semantics=False,  # 是否包含语义信息，默认值为False
        semantic_dim=None  # 语义维度，默认值为None
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.include_opa = include_opa
        self.semantics = semantics
        super().__init__()  # 调用父类的初始化函数
        self.embed_dims = embed_dims  # 设置嵌入维度
        self.include_opa = include_opa  # 设置是否包含不透明度
        self.semantics = semantics  # 设置是否包含语义信息

        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))
        def embedding_layer(input_dims):  # 定义嵌入层函数
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))  # 返回一个线性+ReLU+LayerNorm的序列模型

        self.xyz_fc = embedding_layer(3)
        self.scale_fc = embedding_layer(3)
        self.rot_fc = embedding_layer(4)
        if include_opa:
            self.opacity_fc = embedding_layer(1)
        if semantics:
            assert semantic_dim is not None
            self.semantics_fc = embedding_layer(semantic_dim)
            self.semantic_start = 10 + int(include_opa)
        self.xyz_fc = embedding_layer(3)  # 定义xyz的全连接层
        self.scale_fc = embedding_layer(3)  # 定义scale的全连接层
        self.rot_fc = embedding_layer(4)  # 定义rotation的全连接层
        if include_opa:  # 如果包含不透明度
            self.opacity_fc = embedding_layer(1)  # 定义不透明度的全连接层
        if semantics:  # 如果包含语义信息
            assert semantic_dim is not None  # 确保语义维度不是None
            self.semantics_fc = embedding_layer(semantic_dim)  # 定义语义信息的全连接层
            self.semantic_start = 10 + int(include_opa)  # 计算语义信息的起始索引
        else:
            semantic_dim = 0
        self.semantic_dim = semantic_dim
        self.output_fc = embedding_layer(self.embed_dims)
            semantic_dim = 0  # 如果不包含语义信息，将语义维度设为0
        self.semantic_dim = semantic_dim  # 设置语义维度
        self.output_fc = embedding_layer(self.embed_dims)  # 定义输出的全连接层

    def forward(self, box_3d: torch.Tensor):
        xyz_feat = self.xyz_fc(box_3d[..., :3])
        scale_feat = self.scale_fc(box_3d[..., 3:6])
        rot_feat = self.rot_fc(box_3d[..., 6:10])
        if self.include_opa:
            opacity_feat = self.opacity_fc(box_3d[..., 10:11])
    def forward(self, box_3d: torch.Tensor):  # 定义前向传播函数
        xyz_feat = self.xyz_fc(box_3d[..., :3])  # 计算xyz特征
        scale_feat = self.scale_fc(box_3d[..., 3:6])  # 计算scale特征
        rot_feat = self.rot_fc(box_3d[..., 6:10])  # 计算rotation特征
        if self.include_opa:  # 如果包含不透明度
            opacity_feat = self.opacity_fc(box_3d[..., 10:11])  # 计算不透明度特征
        else:
            opacity_feat = 0.
        if self.semantics:
            semantic_feat = self.semantics_fc(box_3d[..., self.semantic_start: (self.semantic_start + self.semantic_dim)])
            opacity_feat = 0.  # 如果不包含不透明度，将不透明度特征设为0
        if self.semantics:  # 如果包含语义信息
            semantic_feat = self.semantics_fc(box_3d[..., self.semantic_start: (self.semantic_start + self.semantic_dim)])  # 计算语义特征
        else:
            semantic_feat = 0.
            semantic_feat = 0.  # 如果不包含语义信息，将语义特征设为0

        output = xyz_feat + scale_feat + rot_feat + opacity_feat + semantic_feat
        output = self.output_fc(output)
        return output
        output = xyz_feat + scale_feat + rot_feat + opacity_feat + semantic_feat  # 将所有特征相加
        output = self.output_fc(output)  # 通过输出全连接层
        return output  # 返回输出结果
            assert semantic_dim is not None  # 确保语义维度不是None
            self.semantics_fc = embedding_layer(semantic_dim)  # 定义语义信息的全连接层
            self.semantic_start = 10 + int(include_opa)  # 计算语义信息的起始索引
        else:
            semantic_dim = 0
        self.semantic_dim = semantic_dim
        self.output_fc = embedding_layer(self.embed_dims)
            semantic_dim = 0  # 如果不包含语义信息，将语义维度设为0
        self.semantic_dim = semantic_dim  # 设置语义维度
        self.output_fc = embedding_layer(self.embed_dims)  # 定义输出的全连接层

    def forward(self, box_3d: torch.Tensor):
        xyz_feat = self.xyz_fc(box_3d[..., :3])
        scale_feat = self.scale_fc(box_3d[..., 3:6])
        rot_feat = self.rot_fc(box_3d[..., 6:10])
        if self.include_opa:
            opacity_feat = self.opacity_fc(box_3d[..., 10:11])
    def forward(self, box_3d: torch.Tensor):  # 定义前向传播函数
        xyz_feat = self.xyz_fc(box_3d[..., :3])  # 计算xyz特征
        scale_feat = self.scale_fc(box_3d[..., 3:6])  # 计算scale特征
        rot_feat = self.rot_fc(box_3d[..., 6:10])  # 计算rotation特征
        if self.include_opa:  # 如果包含不透明度
            opacity_feat = self.opacity_fc(box_3d[..., 10:11])  # 计算不透明度特征
        else:
            opacity_feat = 0.
        if self.semantics:
            semantic_feat = self.semantics_fc(box_3d[..., self.semantic_start: (self.semantic_start + self.semantic_dim)])
            opacity_feat = 0.  # 如果不包含不透明度，将不透明度特征设为0
        if self.semantics:  # 如果包含语义信息
            semantic_feat = self.semantics_fc(box_3d[..., self.semantic_start: (self.semantic_start + self.semantic_dim)])  # 计算语义特征
        else:
            semantic_feat = 0.
            semantic_feat = 0.  # 如果不包含语义信息，将语义特征设为0

        output = xyz_feat + scale_feat + rot_feat + opacity_feat + semantic_feat
        output = self.output_fc(output)
        return output

        output = xyz_feat + scale_feat + rot_feat + opacity_feat + semantic_feat  # 将所有特征相加
        output = self.output_fc(output)  # 通过输出全连接层
        return output  # 返回输出结果
