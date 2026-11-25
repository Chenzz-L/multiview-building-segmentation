import torch

__all__=['HookClass','register_hook_for_model']


class HookClass:
    def __init__(self,model:torch.nn.Module,layer_hook):
        """
        hook在remove之前都会一致被调用,如果in_features或out_features是一个list,会保存整个训练过程的数据
        Args:
            model:
            layer_hook:
        """
        self.in_features=None
        self.out_features=None
        self.hook=self._register(model,layer_hook)


    def hook_fn(self,module, input, output):

        # self.in_features.append(input[0].clone().detach().cpu())
        # self.out_features.append(output.clone().detach().cpu())
        # self.in_features=input[0].clone().detach().cpu()
        if isinstance(output,list):
            self.out_features=[out.clone() for out in output]
        else:
            self.out_features=output.clone()

        self.in_features=input
    # def _register(self,model:torch.nn.Module,layer_hook):
    #     hook = []
    #     for name, module in model.named_children():
    #         if name in layers_hook:
    #             hook.append(module.register_forward_hook(self.hook_fn))
    #     return hook
    def _register(self,model:torch.nn.Module,layer_hook):
        # for name, module in model.named_children():
        for name, module in model.named_modules():
            if name != "" and layer_hook == name:
                hook=module.register_forward_hook(self.hook_fn)
                return hook

    def remove(self):
        self.hook.remove()

def register_hook_for_model(model,visual_layers=()):
    hooks = [HookClass(model, visual_layer) for visual_layer in visual_layers]
    return hooks

def print_model_stats(model,logger,mode='weight'):
    logger.info("Print model satas")
    if model == 'weight':
        for name, p in model.named_parameters():
            logger.info('\t' + name + ': {}\t{}\t{}\t{}'.format(p.mean(), p.std(), p.min(), p.max()))
    elif mode=='all':
        for name, p in model.named_parameters():
            logger.info('\t'+name + ': {}\t{}\t{}\t{}'.format(p.mean(), p.std(), p.min(), p.max()))
            if p.grad is not None:
                logger.info('\t' + name + '-gradient: {}\t{}\t{}\t{}'.format(p.grad.mean(), p.grad.std(), p.grad.min(), p.grad.max()))
            else:
                logger.info('\t'+name + '-gradient: None')
    else:
        for name, p in model.named_parameters():
            if p.grad is not None:
                logger.info('\t' + name + '-gradient: {}\t{}\t{}\t{}'.format(p.grad.mean(), p.grad.std(), p.grad.min(), p.grad.max()))
            else:
                logger.info('\t'+name + '-gradient: None')
    logger.info("Finish print model satas")

def print_model_stats_to_board(model,writer,epoch,mode='weight'):
    if mode=='weight':
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    elif mode=='all':
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            writer.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), epoch)
            # add_histogram(name + '/grad', param.requires_grad_().clone().cpu().data.numpy(), epoch)
    else:
        for name, param in model.named_parameters():
            writer.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), epoch)

class BaseHook:
    def initial_data_hook(self,trainner):
        """
        initial the data about what you want to record in training
        note:you must return EasyDict which include your info
        """
        pass

    def reset_initial_data_hook(self,trainner):
      pass

    def log_train_iteration_hook(self,trainner):
        pass

    def log_train_epoch_hook(self, trainner):
        pass

    def log_val_iteration_hook(self,trainner):
        pass

    def log_val_epoch_hook(self,trainner):
        pass

    def save_model_hook(self,trainner):
        pass
