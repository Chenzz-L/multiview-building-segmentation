__all__=['regist_dataset','supported_datasets']

import inspect
supported_datasets = {}

def RegistDATASET(dataset,dataset_name=None):
    # supported_datasets.append(dataset.__name__)
    if not inspect.isclass(dataset):
        raise TypeError(f'dataset must be a class,but got {type(dataset)}')

    if dataset_name is None:
        dataset_name=dataset.__name__
    if dataset_name in supported_datasets:
        raise KeyError(f'{dataset_name} is arealdy registed')
    supported_datasets[dataset_name]=dataset
    return supported_datasets


def regist_dataset(dataset_name=None):
    def _regist(dataset_class):
        RegistDATASET(dataset_class,dataset_name)
        return dataset_class
    return _regist
