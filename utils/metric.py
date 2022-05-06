from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calc_psnr(target, pred):
    target = target.detach().numpy()
    pred = pred.detach().numpy()
    return psnr(target, pred)


def calc_mse(target, pred):
    target = target.detach().numpy()
    pred = pred.detach().numpy()
    return mse(target, pred)


def calc_ssim(target, pred):
    target = target.reshape(128, 128, 2).detach().numpy()
    pred = pred.reshape(128, 128, 2).detach().numpy()
    return ssim(target, pred, multichannel=True)
