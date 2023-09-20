import torch
import numpy as np
import matplotlib as plt
import torch_norm_factor

# def KL_Fisher(A, R, overreg=1.05):
#     # A is bx3x3
#     # R is bx3x3
#     global _global_svd_fail_counter
#     try:
#         U,S,V = torch.svd(A)
#         with torch.no_grad(): # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
#             rotation_candidate = torch.matmul(U,V.transpose(1,2))
#             s3sign = torch.det(rotation_candidate)
#         S_sign = S.clone()
#         S_sign[:, 2] *= s3sign
#         log_normalizer = torch_norm_factor.logC_F(S_sign)
#         log_exponent = -torch.matmul(A.view(-1,1,9), R.view(-1, 9,1)).view(-1)
#         _global_svd_fail_counter = max(0, _global_svd_fail_counter-1)
#         return log_exponent + overreg*log_normalizer
#     except RuntimeError as e:
#         _global_svd_fail_counter += 10 # we want to allow a few failures, but not consistent ones
#         if _global_svd_fail_counter > 100: # we seem to have gotten these problems more often than 10% of batches
#             for i in range(A.shape[0]):
#                 print(A[i])
#             raise e
#         else:
#             print('SVD returned NAN fail counter = {}'.format(_global_svd_fail_counter))
#             return None

def KL_Fisher(A, R, overreg=1.05):
    """
    @param A: (b, 3, 3)
    @param R: (b, 3, 3)
    We find torch.svd() on cpu much faster than that on gpu in our case, so we apply svd operation on cpu.
    """
    A, R = A.cpu(), R.cpu()
    U, S, V = torch.svd(A)
    with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(U, V.transpose(1, 2)))
    S_sign = torch.cat((S[:, :2], S[:, 2:] * s3sign[:, None]), -1)
    log_normalizer = torch_norm_factor.logC_F(S_sign)
    log_exponent = -torch.matmul(A.view(-1, 1, 9), R.view(-1, 9, 1)).view(-1)
    log_nll = log_exponent + overreg * log_normalizer
    log_nll = log_nll.cuda()
    return log_nll




def batch_torch_A_to_R(A):
    U,S,V = torch.svd(A)
    with torch.no_grad(): # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(U,V.transpose(1,2)))
    U[:, :, 2] *= s3sign.view(-1, 1)
    R = torch.matmul(U, V.transpose(1,2))
    return R



def vmf_loss(net_out, R, overreg=1.05):
    assert (net_out.shape[0] == R.shape[0]), \
        'batch size of R and prediction_out must be equal'
    A = net_out.view(-1,3,3)
    loss_v = KL_Fisher(A, R, overreg=overreg)
    if loss_v is None:
        R_est = torch.unsqueeze(torch.zeros(3,3, device=R.device, dtype=R.dtype), 0)
        R_est = torch.repeat_interleave(Rest, R.shape[0], 0)
        return None, R_est

    R_est = batch_torch_A_to_R(A)
    return loss_v, R_est



def gauss_log_likelihood(predicted_translation, variance, target):
    variance = variance + 1e-8
    dim = predicted_translation.size()[1]

    x = ((target - predicted_translation) ** 2)
    x = torch.sum(x / (variance), dim=1)
    det = 1.0
    for i in range(dim):
        det *= variance[:, i]
    log_p = -0.5 * (x + dim * torch.log(torch.tensor((2 * np.pi))) +
                    torch.log(det))
    return log_p





