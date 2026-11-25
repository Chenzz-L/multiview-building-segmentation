from typing import Dict
import numpy as np


def convert_dicts_2_dict(input_dicts:Dict):
    """将嵌套的dict转化为不嵌套的dict
        dict[dict]--->dict    

    Args:
        input_dicts (dict): 嵌套dict

    Returns:
        dict: 非嵌套dict
    """
    out_dict={}
    for key,val in input_dicts.items():
        if isinstance(val,Dict):
            converted_dict=convert_dicts_2_dict(val)
            for key1,val1 in converted_dict.items():
                out_dict[key1]=val1
        else:
            out_dict[key]=val
    return out_dict

def conver_chw_and_hwc(input_data:np.ndarray,target_shape='chw'):
    """chw和hwc之间呼唤

    Args:
        input_data (darray): 输入数据
        target_shape (str, optional):['chw','hwc']. Defaults to 'chw'.

    Returns:
        darray
    """
    image_shape=input_data.shape
    if image_shape[0]>image_shape[2]:
        if target_shape=='chw':
            input_data=np.transpose(input_data,[2,0,1])
    else:
        if target_shape=='hwc':
            input_data=np.transpose(input_data,[1,2,0])

    return input_data
