import torch
from cfg import CFG

class AWP:
    def __init__(
            self,
            model,
            optimizer,
            adv_param='weight',
            adv_lr=1,
            adv_eps=0.2,
            start_step=0,
            adv_step=1,
            scaler=None
    ):
        self.model = model  # 模型
        self.optimizer = optimizer  # 优化器
        self.adv_param = adv_param  # 对哪些参数进行对抗训练
        self.adv_lr = adv_lr  # AWP学习率
        self.adv_eps = adv_eps  # AWP扰动大小
        self.start_step = start_step  # AWP开始步数
        self.adv_step = adv_step  # AWP步数
        self.backup = {}  # 参数存储备份字典
        self.backup_eps = {}  # 参数扰动范围存储备份字典
        self.scaler = scaler  # 梯度缩放器

    def attack_backward(self, batch, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_step):
            return None

        self._save()  # 备份参数
        for i in range(self.adv_step):  # AWP步数
            self._attack_step()  # 对抗攻击
            for key, value in batch.items():
                batch[key] = value.to(CFG.device)# labels
            adv_loss, _ = self.model(**batch)  # 对抗训练
            #adv_loss = adv_loss.mean()  # 平均loss
            adv_loss.backward()  # 反向传播
            #self.optimizer.zero_grad()
        self._restore()  # 恢复参数

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)  # 正规化梯度
                norm2 = torch.norm(param.data.detach())  # 正规化参数
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)  # 梯度乘以对抗学习率，再除以梯度的正规化，再乘以参数的正规化
                    param.data.add_(r_at)
                    # 保证参数在扰动范围之内
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1]
                    )

    def _save(self):
        '''
        1. 保存备份原参数，以便恢复
        2. 添加参数的扰动
        '''
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()  # 保存备份原参数
                    grad_eps = self.adv_eps * param.abs().detach()  # 扰动范围
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self, ):
        '''
        恢复原来的参数
        '''
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}