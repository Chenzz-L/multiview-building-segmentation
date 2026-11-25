from typing import Dict,Optional,Union,Tuple
import sys
sys.path.append('/path/to/user/NAS/gwx/code/cnncode')
import torch
import torch.nn as nn

from layers.conv_norm_act import build_activation_layer,build_conv_layer,build_norm_layer,infer_norm_abbr

class ConvModule(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:Union[int,Tuple[int,int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 inplace: bool = False,
                 padding_mode: str = 'zeros',
                 ) -> None:
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        
        # if the conv layer is before a norm layer, bias is unnecessary.
        self.with_bias = not self.with_norm

        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
            bias=self.with_bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            norm_channels = out_channels
            norm = build_norm_layer(norm_cfg, norm_channels)
            self.norm_name=infer_norm_abbr(norm.__class__)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None
        # build activation layer
        if self.with_activation:
            _act_cfg=act_cfg.copy()
            if _act_cfg['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                _act_cfg.setdefault('inplace', inplace)
            self.activate = build_activation_layer(_act_cfg)
    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x=self.conv(x)
        if self.with_norm:
            x=self.norm(x)
        if self.with_activation:
            x=self.activate(x)
        return x

if __name__=='__main__':
    cnn=ConvModule(3,96,3,2,1,
                   conv_cfg=dict(type='Conv2d'),
                   norm_cfg=dict(type='BatchNorm2d'),
                   act_cfg=dict(type='ReLU'))
    x=torch.rand((1,3,512,512))
    y=cnn(x)
    print(y)
