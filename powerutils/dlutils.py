import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from powerutils.common import prefix_dict

def set_random_seed(seed, deterministic=False):
    '''设置随机种子, 尽量固定torch的结果.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def to_tensor(data, **kwargs):
    '''list, ndarray或DataFrame转为Tensor. 会复制底层数据.'''
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    kwargs.setdefault('dtype', torch.float)
    tensor = torch.tensor(data, **kwargs)

    return tensor

def to_numpy(tensor):
    '''Tensor转为ndarray. 尽量返回view.'''
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()

    return tensor.numpy()

def to_device(data, device):
    '''将Tensor移动到指定设备上. Tensor可以被字典, 列表或元组容器容纳.'''
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {name: elem.to(device) for name, elem in data.items()}
    elif isinstance(data, (list, tuple)):
        return [elem.to(device) for elem in data]
    else:
        raise NotImplementedError

class Reshape(nn.Module):
    '''对张量进行reshape的模块.'''
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)

class Transpose(nn.Module):
    '''对张量进行转置的模块.'''
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class BatchNormTs(nn.Module):
    '''对形如(N, L, C)的时间序列在C上做批归一化.'''
    def __init__(self, *args, **kwargs):
        super(BatchNormTs, self).__init__()
        self.bn = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(*args, **kwargs),
            Transpose(1, 2)
        )

    def forward(self, x):
        return self.bn(x)

class MultiOutputModel(nn.Module):
    '''将多个模型的预测结果沿最后一维连在一起的模型.'''
    def __init__(self, models):
        super(MultiOutputModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        preds = [model(x) for model in self.models]
        return torch.cat(preds, dim=-1)

def is_model_trainable(model):
    '''模型是否含可训练的参数.'''
    for parameter in model.parameters():
        if parameter.requires_grad:
            return True
    return False

class DefaultFormatter:
    '''将每个epoch的训练结果格式化为字符串.'''
    def __init__(self, decimals=4):
        self.decimals = decimals

    def __call__(self, record):
        strings = ['']  # 为了在开头多一个额外的' - '.
        for k, v in record.items():
            strings.append(f'{k}: {v:.{self.decimals}f}')
        info = ' - '.join(strings)

        return info

class EarlyStopping:
    '''
    实现早停法的类.
    patience次出现loss > min_loss + min_delta时, 返回False.
    '''
    def __init__(self, monitor='val_loss', min_delta=0, patience=1):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.reset()

    def __call__(self, result):
        value = result[self.monitor]
        if value < self.min_value:
            self.min_value = value
            self.counter = 0
        if value > (self.min_value + self.min_delta):
            self.counter += 1

        return self.counter >= self.patience

    def reset(self):
        self.min_value = np.inf
        self.counter = 0

class Trainer:
    '''训练深度学习模型的类.'''
    def __init__(
        self, model, loss_fn,
        metric_dict=None, optimizer=None, device='cpu'
    ):
        '''metric_dict是计算指标的函数构成的字典.'''
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.metric_dict = {} if metric_dict is None else metric_dict
        self.optimizer = optimizer
        self.device = device

    @torch.no_grad()
    def _calc_metrics(self, y, yp):
        '''根据真值和预测值计算指标.'''
        return {
            name: metric_fn(y, yp).item()
            for name, metric_fn in self.metric_dict.items()
        }

    def train_step(self, x, y):
        '''在一批x和y上进行训练. 返回loss和metrics构成的字典.'''
        if self.optimizer is None:
            raise ValueError('optimizer不能为None')
        self.model.train()
        x = to_device(x, self.device)
        y = to_device(y, self.device)

        # 后向传播.
        yp = self.model(x)
        loss = self.loss_fn(y, yp)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        result = {'loss': loss.item()}
        metrics = self._calc_metrics(y, yp)
        result.update(metrics)

        return result

    @torch.no_grad()
    def evaluate_step(self, x, y):
        '''在一批x和y上进行评估. 返回loss和metrics构成的字典.'''
        self.model.eval()
        x = to_device(x, self.device)
        y = to_device(y, self.device)

        yp = self.model(x)
        loss = self.loss_fn(y, yp)
        result = {'loss': loss.item()}
        metrics = self._calc_metrics(y, yp)
        result.update(metrics)

        return result

    def _run_epoch(self, dataloader, fn):
        '''遍历dataloader并应用函数fn. 返回结果的平均值构成的字典.'''
        results = [fn(x, y) for x, y in dataloader]
        avg_result = pd.DataFrame.from_records(results).mean().to_dict()
        return avg_result

    def train_epoch(self, dataloader):
        '''在dataloader上训练一轮. 返回结果的平均值构成的字典.'''
        return self._run_epoch(dataloader, self.train_step)

    def evaluate(self, dataloader):
        '''在dataloader上评估一轮. 返回结果的平均值构成的字典.'''
        return self._run_epoch(dataloader, self.evaluate_step)

    def fit(
        self, dataloader, val_dataloader=None, epochs=10,
        early_stopping=None, formatter=None, prompt=True
    ):
        '''在dataloader上训练epochs轮. 打印每轮的结果并返回相应的表格.'''
        if early_stopping is None:
            early_stopping = lambda x: False
        else:
            early_stopping.reset()
        if formatter is None:
            formatter = DefaultFormatter()

        results = []
        for t in range(1, epochs + 1):
            result = self.train_epoch(dataloader)
            # 如果有验证集, 将验证集的结果并入record.
            if val_dataloader is not None:
                val_result = self.evaluate(val_dataloader)
                val_result = prefix_dict(val_result, 'val_')
                result.update(val_result)
            results.append(result)

            # 打印每轮的结果.
            if prompt:
                head = f'[Epoch {t}/{epochs}]'
                info = head + formatter(result)
                print(info)

            # 早停机制.
            if early_stopping(result):
                break

        # 以DataFrame表格的形式记录结果.
        index = pd.Index(np.arange(1, t + 1), name='epoch')
        log = pd.DataFrame.from_records(results, index=index)

        return log

class ForecastDataset(Dataset):
    '''构造时间序列预测窗口的数据集.'''
    def __init__(
        self, df,
        target_labels, history_labels, future_labels,
        history_length=1, future_length=1, gap=0, stride=1,
        transform=None
    ):
        '''
        Parameters
        ----------
        df : DataFrame
            时间序列数据.

        target_labels : list of str
            预测目标的名称.

        history_labels : list of str
            历史数据的名称.

        future_labels : list of str
            未来数据的名称.

        history_length : int, optional
            历史数据的时间长度. 默认为1.

        future_length : int, optional
            未来数据的时间长度. 默认为1.

        gap : int, optional
            历史数据与未来数据的间距. 默认为0.

        stride : int, optional
            各窗口起点间的间隔. 默认为1.

        transform : callable, optional
            对df底层数组采取的变换. 默认变换成float32的张量.
        '''
        # 对底层数据进行变换.
        data = df.to_numpy()
        if transform is None:
            data = to_tensor(data)
        else:
            data = transform(data)

        self.data = data
        self.transform = transform
        self.columns = list(df.columns)
        self.time_index = df.index

        # 计算窗口宽度.
        self.history_length = history_length
        self.future_length = future_length
        self.gap = gap
        self.stride = stride
        self.window_length = history_length + gap + future_length

        # 记录label对应的整数索引.
        self.target_labels = list(target_labels)
        self.history_labels = list(history_labels)
        self.future_labels = list(future_labels)
        self.target_label_indices = self.get_label_indices(target_labels)
        self.history_label_indices = self.get_label_indices(history_labels)
        self.future_label_indices = self.get_label_indices(future_labels)

        # 收集每个时间序列窗口的下标, 跳过不连续的部分.
        self.window_start_indices = []
        group_start_index = 0
        time_diff = self.time_index.to_series().diff()
        group_id = (time_diff > time_diff.min()).cumsum()
        for group in self.time_index.groupby(group_id).values():
            group_length = len(group)
            if group_length > self.window_length:
                for i in range(0, group_length - self.window_length + 1, stride):
                    self.window_start_indices.append(group_start_index + i)
            group_start_index += group_length

    def get_label_indices(self, labels):
        '''获取一组labels在df的列里对应的下标.'''
        return [self.columns.index(label) for label in labels]

    def get_history_time_indices(self, index):
        '''获取一个窗口的历史部分在df的行里对应的下标.'''
        start = self.window_start_indices[index]
        stop = start + self.history_length
        indices = list(range(start, stop))

        return indices

    def get_future_time_indices(self, index):
        '''获取一个窗口的未来部分在df的行里对应的下标.'''
        stop = self.window_start_indices[index] + self.window_length
        start = stop - self.future_length
        indices = list(range(start, stop))

        return indices

    def get_history_times(self, index):
        '''获取一个窗口的历史部分的时间索引.'''
        return self.time_index[self.get_history_time_indices(index)]

    def get_future_times(self, index):
        '''获取一个窗口的未来部分的时间索引.'''
        return self.time_index[self.get_future_time_indices(index)]

    def __len__(self):
        '''返回窗口数量.'''
        return len(self.window_start_indices)

    def __getitem__(self, index):
        '''
        返回第index个窗口的数据.

        Parameters
        ----------
        index : int
            索引下标.

        Returns
        -------
        x : dict of array-like
            - history: 已知的历史序列(可能含有过去的目标序列).
            - future: 已知的未来序列, 例如数值天气预报.

        y : array-like
            作为标签的未来的目标序列.
        '''
        history_time_indices = self.get_history_time_indices(index)
        future_time_indices = self.get_future_time_indices(index)
        x_history = self.data[history_time_indices, :][:, self.history_label_indices]
        x_future = self.data[future_time_indices, :][:, self.future_label_indices]
        x = {'history': x_history, 'future': x_future}
        y = self.data[future_time_indices, :][:, self.target_label_indices]

        return x, y

    def __repr__(self):
        '''打印数据集的基本信息.'''
        x, y = self[0]
        return '\n'.join([
            f'target_labels: {self.target_labels}',
            f'history_labels: {self.history_labels}',
            f'future_labels: {self.future_labels}',
            f'history_length: {self.history_length}',
            f'future_length: {self.future_length}',
            f'gap: {self.gap}',
            f'stride: {self.stride}',
            f'window_length: {self.window_length}',
            f'num_windows: {len(self)}',
            f'shape of x_history: {x["history"].shape}',
            f'shape of x_future: {x["future"].shape}',
            f'shape of y_future: {y.shape}'
        ])

    def new_dataset(self, df):
        '''利用数据集的参数在另一个df上创建数据集.'''
        return ForecastDataset(
            df=df,
            target_labels=self.target_labels,
            history_labels=self.history_labels,
            future_labels=self.future_labels,
            history_length=self.history_length,
            future_length=self.future_length,
            gap=self.gap,
            stride=self.stride,
            transform=self.transform
        )