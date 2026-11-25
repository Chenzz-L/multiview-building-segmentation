import torch
import numpy as np


__all__ = ['calc_confusion_matrix', 'calc_ious_from_confusion_matrix', 'calc_tp_fp_fn_from_conf_matrix',
           'save_acc_file_from_confusion_matrix', 'binarize', 'calc_tp_fp_fn_from_tensor', 'calculate_acc_from_tp_tn_fp',
           'save_acc_file_from_tp_tn_fp', 'compute_se_and_ae', 'compute_se_and_ae_np', 'calculate_errors_from_se_ae',
           'save_acc_file_from_se_ae', 'calc_tp_fp_fn_from_tensor_cuda','calculate_acc_from_tp_tn_fp_cuda','cal_eval_image']


def calc_confusion_matrix(prediction: np.ndarray, ground_truth: np.ndarray, class_number: int) -> np.ndarray:
    """Calculate confusion_matrix. fast:0.015s
    Args:
        prediction (1-dim int np.array): a flattened prediction array.
        ground_truth (1-dim int np.array): a flattened ground_truth array.
        class_number:how many classes to predict.
    Returns:
        confusion_matrix(np.array:WH): a 2-dim array.
    """
    if prediction.ndim != 1 or ground_truth.ndim != 1:
        raise Exception("ndim != 1")
    k = (prediction >= 0) & (prediction < class_number)
    confusion_matrix = np.bincount(class_number * prediction[k].astype(int) + ground_truth[k],
                                   minlength=class_number ** 2)
    return confusion_matrix.reshape(class_number, class_number)


def calc_ious_from_confusion_matrix(conf_matrix: np.ndarray):
    eps = 1e-6
    n_classes = conf_matrix.shape[0]
    ious = np.zeros(n_classes, dtype=np.float32)
    precision = np.zeros(n_classes, dtype=np.float32)
    recall = np.zeros(n_classes, dtype=np.float32)
    f1score = np.zeros(n_classes, dtype=np.float32)
    tp = np.diag(conf_matrix)
    total = conf_matrix.sum() + eps  # (tp+fp+tn+fn)
    total_class = 0.0
    for c in range(n_classes):
        fp = conf_matrix[c, :].sum() - conf_matrix[c, c]
        fn = conf_matrix[:, c].sum() - conf_matrix[c, c]
        ious[c], precision[c], recall[c], f1score[c] = calculate_acc_from_tp_tn_fp(tp[c], fp, fn)
        # ious[c] = (tp[c] + eps) / (tp[c] + fp + fn + eps)
        # precision[c] = (tp[c] + eps) / (tp[c] + fp + eps)
        # recall[c] = (tp[c] + eps) / (tp[c] + fn + eps)
        # f1score[c] = 2 * precision[c] * recall[c] / (recall[c] + precision[c])
        a = (conf_matrix[c, :] * conf_matrix[c, c]).sum() + eps  # (tp+fn)*tp
        b = (conf_matrix[c, :].sum() + conf_matrix[:, c].sum() - conf_matrix[c, c]) + eps  # (tp+fp+fn)
        total_class += a / b
    # [(tp+fn)/(tp+fp+tn+fn)]*[tp/(tp+fp+fn)]
    FWIoU = round(1 / total * total_class, 5)
    mIoU = np.mean(ious).item()
    return ious, FWIoU, mIoU, precision, recall, f1score


def calc_tp_fp_fn_from_conf_matrix(conf_matrix: np.ndarray, class_id=1):
    r"""
    calculate binary tp, fp, fn
    """
    if conf_matrix.ndim != 2:
        raise Exception('conf_matrix.ndim != 2')
    tp = np.diag(conf_matrix)[class_id]  # 0: background, 1: target
    fp = conf_matrix[class_id, :].sum() - conf_matrix[class_id, class_id]
    fn = conf_matrix[:, class_id].sum() - conf_matrix[class_id, class_id]
    return tp, fp, fn


def save_acc_file_from_confusion_matrix(confusion_matrix, class_num, statistics_file):
    ious, FWIoU, mIoU, precision, recall, f1score = calc_ious_from_confusion_matrix(confusion_matrix)
    stats = open(statistics_file, 'w')
    for i in range(class_num):
        stats.write('IoUs[%d] : %f ,' % (i, ious[i]))
        stats.write('precision[%d] : %f ,' % (i, precision[i]))
        stats.write('recall[%d] : %f ,' % (i, recall[i]))
        stats.write('f1score[%d] : %f ,\n' % (i, f1score[i]))
    stats.write('FWIoU : %f, mIoU : %f \n' % (FWIoU, mIoU))
    stats.close()


def binarize(y_data, threshold=0.5):
    """
    大坑，y_data需要从计算图中使用detach().clone()剥离，loss中的sigmoid不会影响模型的输出
    args:
        y_data : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0~1.0]
    return binarized y_data
    """
    data=y_data.detach().clone()
    data=torch.sigmoid(data)
    data[data < threshold] = 0.0
    data[data >= threshold] = 1.0
    return data

def calc_tp_fp_fn_from_tensor_cuda(prediction: torch.Tensor , ground_truth: torch.Tensor):
    """
    calc_tp_fp_fn_from_tensor cuda version
    Args:
        prediction:
        ground_truth:

    Returns:
        tp,fp,fn----in cuda
    """
    assert prediction.shape == ground_truth.shape
    assert type(prediction) == type(ground_truth)
    tp = (prediction * ground_truth).sum(dtype=torch.float32)  # prediction == 1 & ground_truth == 1
    fp = (prediction * (1 - ground_truth)).sum(dtype=torch.float32)  # prediction == 1 & ground_truth == 0
    fn = ((1 - prediction) * ground_truth).sum(dtype=torch.float32)  # prediction == 0 & ground_truth == 1
    return tp, fp, fn

def calc_tp_fp_fn_from_tensor(prediction: torch.Tensor or np.ndarray, ground_truth: torch.Tensor or np.ndarray):
    assert prediction.shape == ground_truth.shape
    assert type(prediction) == type(ground_truth)
    if type(prediction) == torch.Tensor:
        tp = (prediction * ground_truth).sum(dtype=torch.float32)  # prediction == 1 & ground_truth == 1
        fp = (prediction * (1 - ground_truth)).sum(dtype=torch.float32)  # prediction == 1 & ground_truth == 0
        fn = ((1 - prediction) * ground_truth).sum(dtype=torch.float32)  # prediction == 0 & ground_truth == 1
    elif type(prediction) == np.ndarray:
        tp = (prediction * ground_truth).sum(dtype=np.float32)  # prediction == 1 & ground_truth == 1
        fp = (prediction * (1 - ground_truth)).sum(dtype=np.float32)  # prediction == 1 & ground_truth == 0
        fn = ((1 - prediction) * ground_truth).sum(dtype=np.float32)  # prediction == 0 & ground_truth == 1
    else:
        raise Exception('unknown data type')
    return tp.item(), fp.item(), fn.item()


def calculate_acc_from_tp_tn_fp(tp: float, fp: float, fn: float, epsilon: float = 1e-6):
    if tp == fp == fn == 0:
        return 0, 0, 0, 0
    iou = tp / (tp + fp + fn + epsilon) if (tp + fp + fn != 0) else 0
    precision = float(tp + epsilon) / float(tp + fp + epsilon) if (tp + fp != 0) else 0
    recall = float(tp + epsilon) / float(tp + fn + epsilon) if (tp + fn != 0) else 0
    f1score = 2 * precision * recall / (recall + precision + epsilon) if (recall + precision != 0) else 0
    return iou, precision, recall, f1score

def calculate_acc_from_tp_tn_fp_cuda(tp:torch.Tensor, fp:torch.Tensor, fn:torch.Tensor, epsilon:float=1e-6):
    device = tp.device
    epsilon=torch.tensor(epsilon).to(device)
    if tp==fp==fn==0:
        return torch.tensor(0).to(device),torch.tensor(0).to(device),torch.tensor(0).to(device),torch.tensor(0).to(device)
    iou = tp / (tp + fp + fn + epsilon) if (tp + fp + fn != 0) else torch.tensor(0).to(device)
    precision = (tp + epsilon) / (tp + fp + epsilon) if (tp + fp != 0) else torch.tensor(0).to(device)
    recall = (tp + epsilon) / (tp + fn + epsilon) if (tp + fn != 0) else torch.tensor(0).to(device)
    f1score = 2 * precision * recall / (recall + precision + epsilon) if (recall + precision != 0) else torch.tensor(0).to(device)
    return iou, precision, recall, f1score

def save_acc_file_from_tp_tn_fp(tp: float, fp: float, fn: float, statistics_file: str, epsilon: float = 1e-6):
    iou, precision, recall, f1score = calculate_acc_from_tp_tn_fp(tp, fp, fn, epsilon)
    gs = open(statistics_file, 'w')
    gs.write('tp: %f\nfp: %f\nfn: %f\n' % (tp, fp, fn))
    gs.write('iou = %.6f\nprecision = %.6f\nrecall = %.6f\nf1score = %.6f' % (iou, precision, recall, f1score))
    gs.close()


def compute_se_and_ae(prediction: torch.Tensor, ground_truth: torch.Tensor):
    len_tensor = 1
    for l_ in prediction.shape:
        len_tensor *= l_
    SquaredError = torch.sum((prediction - ground_truth) ** 2) / len_tensor
    AbsoluteError = torch.sum(torch.abs(prediction - ground_truth)) / len_tensor
    var = torch.var(ground_truth)
    return SquaredError.item(), AbsoluteError.item(), var.item()  # , len_tensor


def compute_se_and_ae_np(prediction: np.ndarray, ground_truth: np.ndarray):
    len_tensor = 1
    for l_ in prediction.shape:
        len_tensor *= l_
    SquaredError = np.sum((prediction - ground_truth) ** 2) / len_tensor
    AbsoluteError = np.sum(np.absolute(prediction - ground_truth)) / len_tensor
    var = np.var(ground_truth)
    return SquaredError, AbsoluteError, var


def calculate_errors_from_se_ae(se: float, ae: float, var: float, steps: float = 1):
    r"""
    calculate global average errors from se, ae and var, need to divide by global steps
    :param se: SquaredError
    :param ae: AbsoluteError
    :param var:
    :param steps:
    :return:
    """
    # prediction = prediction.flatten()
    # ground_truth = ground_truth.flatten()
    # if prediction.shape[0] != ground_truth.shape[0]:
    #     raise ValueError('prediction.shape[0] != ground_truth.shape[0]')
    MeanSquaredError = se / steps
    RootMeanSquareError = MeanSquaredError ** 0.5
    MeanAbsoluteError = ae / steps
    RSquared = 1 - MeanSquaredError / var / steps
    return MeanSquaredError, RootMeanSquareError, MeanAbsoluteError, RSquared


def save_acc_file_from_se_ae(se: float, ae: float, var: float, statistics_file, steps: float = 1):
    MSE, RMSE, MAE, RS = calculate_errors_from_se_ae(se, ae, var, steps)
    gs = open(statistics_file, 'w')
    gs.write('MSE = %.6f\nRMSE = %.6f\nMAE = %.6f\nRS = %.6f' % (MSE, RMSE, MAE, RS))
    gs.close()

def cal_eval_image(binary_data:torch.Tensor, label_data:torch.Tensor):
    eval_data = torch.zeros([3] + list(binary_data.shape), dtype = torch.uint8, device = binary_data.device)
    # compute flag
    tn_flag = torch.logical_and(label_data == 0, binary_data == 0)
    tp_flag = torch.logical_and(label_data == 1, binary_data == 1)
    fp_flag = torch.logical_and(label_data == 0, binary_data == 1)
    fn_flag = torch.logical_and(label_data == 1, binary_data == 0)
    # R,G,B
    # TN: 255, 255, 255
    # eval_data[:, tn_flag] = 255
    eval_data[:, tn_flag] = 1
    # TP: 255, 255, 0
    eval_data[:, tp_flag] = torch.tensor([[0, 255, 0]], dtype = torch.uint8, device = binary_data.device).T
    # FP: 255, 0, 0
    eval_data[:, fp_flag] = torch.tensor([[255, 0, 0]], dtype = torch.uint8, device = binary_data.device).T
    # FN: 0, 0, 255
    eval_data[:, fn_flag] = torch.tensor([[0, 0, 255]], dtype = torch.uint8, device = binary_data.device).T
    return eval_data

if __name__ == "__main__":
    tp=torch.tensor(50).cuda()
    fp=torch.tensor(20).cuda()
    fn=torch.tensor(30).cuda()
    iou, precision, recall, f1score=calculate_acc_from_tp_tn_fp_cuda(tp,fp,fn)
    print(iou, precision, recall, f1score)
