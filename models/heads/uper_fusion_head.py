from models.heads.uper_head import *
from layers.spatial_fusion import SpatialFusion2D,SpatialFusion2DV2

class UPerFusionHead(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                pool_scales:Tuple[int]  = (1, 2, 3, 6),
                conv_cfg:Optional[Dict] = None,
                norm_cfg:Optional[Dict] = None,
                act_cfg:Optional[Dict]  = dict(type='ReLU'),
                align_corners=False,
                mode='nearest'):
        super(UPerFusionHead, self).__init__()
        self.uperhead=UPerHead(in_channels,out_channels,pool_scales,conv_cfg,norm_cfg,act_cfg,align_corners)
        self.spatial_fusion=SpatialFusion2DV2(in_channels,mode,align_corners,True)
        self.out_channels=self.uperhead.out_channels

    def forward(self,x,grid,batch_info):
        x=self.spatial_fusion(x,grid,batch_info)
        x=self.uperhead(x)
        return x
