# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Create by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Rainbowsecret (yuyua@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

STAGE_CFG = namedtuple('STAGE_CFG', ['NUM_MODULES', 'NUM_BRANCHES', 'NUM_BLOCKS', 'NUM_CHANNELS', 'BLOCK', 'FUSE_METHOD'])
HRNET_CFG = namedtuple('HRNET_CFG', ['FINAL_CONV_KERNEL', 'STAGE1', 'STAGE2', 'STAGE3', 'STAGE4'])

HRNET_18 = HRNET_CFG(FINAL_CONV_KERNEL=1,
                     STAGE1=STAGE_CFG(1, 1, [4], [64], 'BOTTLENECK', 'SUM'),
                     STAGE2=STAGE_CFG(1, 2, [4, 4], [18, 36], 'BASIC', 'SUM'),
                     STAGE3=STAGE_CFG(4, 3, [4, 4, 4], [18, 36, 72], 'BASIC', 'SUM'),
                     STAGE4=STAGE_CFG(3, 4, [4, 4, 4, 4], [18, 36, 72, 144], 'BASIC', 'SUM'))

HRNET_32 = HRNET_CFG(FINAL_CONV_KERNEL=1,
                     STAGE1=STAGE_CFG(1, 1, [4], [64], 'BOTTLENECK', 'SUM'),
                     STAGE2=STAGE_CFG(1, 2, [4, 4], [32, 64], 'BASIC', 'SUM'),
                     STAGE3=STAGE_CFG(4, 3, [4, 4, 4], [32, 64, 128], 'BASIC', 'SUM'),
                     STAGE4=STAGE_CFG(3, 4, [4, 4, 4, 4], [32, 64, 128, 256], 'BASIC', 'SUM'))

HRNET_48 = HRNET_CFG(FINAL_CONV_KERNEL=1,
                     STAGE1=STAGE_CFG(1, 1, [4], [64], 'BOTTLENECK', 'SUM'),
                     STAGE2=STAGE_CFG(1, 2, [4, 4], [48, 96], 'BASIC', 'SUM'),
                     STAGE3=STAGE_CFG(4, 3, [4, 4, 4], [48, 96, 192], 'BASIC', 'SUM'),
                     STAGE4=STAGE_CFG(3, 4, [4, 4, 4, 4], [48, 96, 192, 384], 'BASIC', 'SUM'))
MODEL_CONFIGS = {
    'hrnet18': HRNET_18,
    'hrnet32': HRNET_32,
    'hrnet48': HRNET_48,
}
