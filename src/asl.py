import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        # 初始化，设置各个超参数
        self.gamma_neg = gamma_neg  # 负类的聚焦因子
        self.gamma_pos = gamma_pos  # 正类的聚焦因子
        self.clip = clip  # 负类概率裁剪值，用于稳定数值
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss  # 控制是否禁用焦点损失中的梯度计算
        self.eps = eps  # 数值稳定性的小常数，防止log计算时出现负无穷

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: input logits, 模型的输出logits（未经过sigmoid的预测值）
        y: targets (multi-label binarized vector), 真实标签，二进制向量形式
        """
        # 计算概率值，sigmoid函数用于将logits转化为概率
        x_sigmoid = torch.sigmoid(x)  # 对logits进行sigmoid转换为概率
        xs_pos = x_sigmoid  # 正类的概率
        xs_neg = 1 - x_sigmoid  # 负类的概率
        # 对负类概率进行裁剪，避免数值过小或过大
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)  # 裁剪负类概率，防止过小的概率导致数值不稳定
        # 计算基本的交叉熵损失
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))  # 正类损失：y=1时，log(p)的损失
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))  # 负类损失：y=0时，log(1-p)的损失
        loss = los_pos + los_neg  # 将正负类损失加和
        # 加入焦点损失的加权部分，以对难分类的样本进行加权
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)  # 禁用梯度计算，以加速计算（节省内存）
            # 计算正类和负类的概率加权项
            pt0 = xs_pos * y  # 正类的预测概率
            pt1 = xs_neg * (1 - y)  # 负类的预测概率
            pt = pt0 + pt1  # 总的预测概率，pt为正类或负类概率
            # 计算每个类别的聚焦因子
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)  # 对正类和负类应用不同的gamma值
            # 焦点损失的加权因子，增强困难样本的损失
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)  # (1 - pt)的gamma次方，表示难分类样本的加权
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)  # 恢复梯度计算
            loss *= one_sided_w  # 将损失加权
        return -loss.sum()  # 返回总损失的负值，sum()为所有样本的损失之和



class AsymmetricLossOptimized(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()
        # 初始化各个超参数
        self.gamma_neg = gamma_neg  # 负类的聚焦因子
        self.gamma_pos = gamma_pos  # 正类的聚焦因子
        self.clip = clip  # 裁剪负类概率的阈值
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss  # 是否禁用焦点损失中的梯度计算
        self.eps = eps  # 数值稳定性的小常数，防止log计算时出现负无穷
        # 初始化各类张量为None，后续在计算中再赋值
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: input logits, 模型的输出logits（未经sigmoid的预测值）
        y: targets (multi-label binarized vector), 真实标签，二进制向量形式
        """
        # 设置真实标签和反向标签
        self.targets = y  # 正类标签
        self.anti_targets = 1 - y  # 负类标签（反向标签）
        # 计算概率值：将logits通过sigmoid转化为概率
        self.xs_pos = torch.sigmoid(x)  # 正类的预测概率
        self.xs_neg = 1.0 - self.xs_pos  # 负类的预测概率
        # 对负类概率进行裁剪，避免数值过小或过大
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)  # 裁剪负类概率，防止过小的概率影响稳定性
        # 计算基础的交叉熵损失
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))  # 正类的损失
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))  # 负类的损失
        # 如果开启了聚焦损失
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)  # 禁用梯度计算，提高计算效率
            # 计算聚焦损失的加权因子
            self.xs_pos = self.xs_pos * self.targets  # 仅对正类的概率进行加权
            self.xs_neg = self.xs_neg * self.anti_targets  # 仅对负类的概率进行加权
            # 计算加权因子，其中 (1 - xs_pos - xs_neg) 表示错误率，gamma_pos 和 gamma_neg 控制正负类的加权
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)  # 恢复梯度计算
            # 对基础损失进行加权
            self.loss *= self.asymmetric_w
        # 返回损失的负值（损失通常是负对数，因此这里返回的是总损失之和的负值）
        return -self.loss.sum()



class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        # 初始化损失函数的超参数
        self.eps = eps  # 标签平滑（label smoothing）的系数
        self.logsoftmax = nn.LogSoftmax(dim=-1)  # 用于计算每个类别的对数概率
        self.targets_classes = []  # 存储目标标签的张量
        self.gamma_pos = gamma_pos  # 正类的聚焦因子
        self.gamma_neg = gamma_neg  # 负类的聚焦因子
        self.reduction = reduction  # 损失的归约方式，支持 'mean' 和 'sum'
    
    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size, number_classes)
        "target" dimensions: - (batch_size)
        '''
        # 将目标标签转化为one-hot编码形式，self.targets_classes是一个形状为(batch_size, num_classes)的张量
        # 目标标签的位置置为1，其余位置为0
        num_classes = inputs.size()[-1]  # 获取类别数目
        log_preds = self.logsoftmax(inputs)  # 计算输入logits的对数softmax
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
        # ASL weights
        targets = self.targets_classes  # 正类标签的one-hot表示
        anti_targets = 1 - targets  # 负类标签的one-hot表示（反转）
        xs_pos = torch.exp(log_preds)  # 计算正类的预测概率
        xs_neg = 1 - xs_pos  # 计算负类的预测概率
        xs_pos = xs_pos * targets  # 正类预测概率，乘以目标标签
        xs_neg = xs_neg * anti_targets  # 负类预测概率，乘以反目标标签
        # 计算聚焦损失的权重
        # (1 - xs_pos - xs_neg) 是对困难样本的权重计算公式，gamma根据标签来控制
        # 如果目标是正类，则使用gamma_pos；如果目标是负类，则使用gamma_neg
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg, self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w  # 用计算的权重对log_preds加权
        if self.eps > 0:  # label smoothing
            # 标签平滑处理（避免标签过于极端，改善训练效果）
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)
        # 计算损失
        loss = - self.targets_classes.mul(log_preds)  # 计算每个样本的损失
        loss = loss.sum(dim=-1)  # 按类别维度求和
        if self.reduction == 'mean':
            loss = loss.mean()  # 如果选择“均值”归约方式，计算所有样本的平均损失
        return loss
