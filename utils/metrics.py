import numpy as np
import torch  
import torch.nn.functional as F 
import scipy
import lpips
from typing import Dict

class ImageQualityMetrics:
    def __init__(self):
        self.psnr_score = 0.0
        self.ssim_score = 0.0
        self.ms_ssim_score = 0.0
        self.vif_score = 0.0
        self.lpips_score = 0.0
        
        self.lpips_loss_fn = lpips.LPIPS(net='alex')

    def evaluate(self, test_img, comp_img) -> None:
        """
        Calculate PSNR, SSIM, MS-SSIM, VIF, and LPIPS between test_img and comp_img

        Args:
            test_img (np.ndarray): original image
            comp_img (np.ndarray): compressed or distorted image
        """
        # img should be numpy array with (H, W, C)
        assert type(test_img) == np.ndarray
        assert type(comp_img) == np.ndarray
        assert test_img.shape[-1] == 4 or test_img.shape[-1] == 3
        assert comp_img.shape[-1] == 4 or comp_img.shape[-1] == 3

        test_img = test_img[:,:,:3]
        comp_img = comp_img[:,:,:3]

        self.psnr_score = psnr(test_img, comp_img)
        self.ssim_score = ssim(test_img, comp_img)
        self.ms_ssim_score = ms_ssim(test_img, comp_img)
        self.vif_score = vif(test_img, comp_img)
        self.lpips_score = self.lpips_loss_fn(torch.from_numpy(test_img).unsqueeze(0).permute(0, 3, 1, 2), 
                          torch.from_numpy(comp_img).unsqueeze(0).permute(0, 3, 1, 2)).item()

    """
    Use the property of the metrics to get the metrics
    """
    @property
    def psnr(self) -> float:
        return self.psnr_score

    @property
    def ssim(self) -> float:
        return self.ssim_score

    @property
    def ms_ssim(self) -> float:
        return self.ms_ssim_score

    @property
    def vif(self) -> float:
        return self.vif_score

    @property
    def lpips(self) -> float:
        return self.lpips_score

    def get_metrics(self) -> Dict:
        """
        Return a dictionary of metrics
        """
        results = {}
        results["psnr"] = self.psnr
        results["ssim"] = self.ssim
        results["ms-ssim"] = self.ms_ssim
        results["vif"] = self.vif
        results["lpips"] = self.lpips
        return results

    def get_metrics_text(self) -> str:
        """
        Return printed results of the calculated metrics
        """
        ret = f'PSNR: {self.psnr_score}\n'
        ret += f'SSIM: {self.ssim_score}\n' + f'MS-SSIM: {self.ms_ssim_score}\n'
        ret += f'VIF: {self.vif_score}\nLPIPS:{self.lpips_score}'
        return ret



### PSNR
def psnr(img0, img1):
    mse = np.mean(np.square(img0 - img1))

    # Check if img0 and img1 are the same
    if mse == 0:
        return 0.0

    max_value = np.max(img0)
    return 20*np.log10(max_value/np.sqrt(mse))


### SSIM
def ssim(img0, img1, C1=0.01**2, C2=0.03**2, C3 = 0.03, alpha=1, beta=1, gamma=1):
    """
    Most basic Structural Similarity Index (SSIM)
    """
    mu0 = np.mean(img0)
    mu1 = np.mean(img1)
    std0 = np.std(img0)
    std1 = np.std(img1)
    std_0_1 = np.mean((img0 - mu0) * (img1 - mu1))
    luminance = (2*mu0*mu1 + C1) / (mu0**2 + mu1**2 + C1)
    contrast = (2*std0*std1 + C2) / (std0**2 + std1**2 + C2)
    structure = (std_0_1 + C3) / (std0*std1 + C3)
    return luminance**alpha + contrast**beta + structure**gamma


### MS-SSIM. Inspired by https://github.com/pranjaldatta/SSIM-PyTorch/blob/master/SSIM_notebook.ipynb

def ms_ssim(img0, img1, C1=0.01**2, C2=0.03**2, val_range=255, window_size=11):
    """
    Mean Structural Similarity Index. Assuming alpha = beta = gamma = 1 for better form
    """
    def gaussian(window_size, sigma):
        """
        Generate a normalized tensor with values drawn from Gaussian distribution with standord deviation sigma
        """    
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        return gauss

    def create_window(window_size, channel=1):
    # Generate an 1D tensor containing values sampled from a gaussian distribution
        _1d_window = gaussian(window_size, sigma=1.5).unsqueeze(1)
        
        # Converting to 2D  
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        
        window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

        return window

    pad = window_size // 2
    img0 = torch.from_numpy(img0).type(torch.FloatTensor)
    img1 = torch.from_numpy(img1).type(torch.FloatTensor)
    
    channels, height, width = img1.size()

    real_size = min(window_size, height, width) 
    window = create_window(real_size, channel=channels)
    
    # luminance calculated with convolutional sliding windows
    mu0 = F.conv2d(img0, window, padding=pad, groups=channels)
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)

    # Contrast calculated with convolutional sliding windows
    var0 = F.conv2d(img0 * img0, window, padding=pad, groups=channels) - mu0**2
    var1 = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1**2 
    var01 =  F.conv2d(img0 * img1, window, padding=pad, groups=channels) - mu0*mu1 

    ssim_score = ((2*mu0*mu1 + C1) * (2 * var01 + C2)) / ((mu0**2 + mu1**2  + C1) * (var0 + var1 + C2))
    
    return ssim_score.mean().item()

### VIF. Implementation from https://github.com/aizvorski/video-quality/blob/master/vifp.py
# The original implementation was very specific. Here using a working implementation for my experiments
def vif(ref, dist):
    sigma_nsq=2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):
       
        N = 2**(4-scale+1) + 1
        sd = N/5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]
                
        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2
        
        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0
        
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        
        g[sigma1_sq<eps] = 0
        sv_sq[sigma1_sq<eps] = sigma2_sq[sigma1_sq<eps]
        sigma1_sq[sigma1_sq<eps] = 0
        
        g[sigma2_sq<eps] = 0
        sv_sq[sigma2_sq<eps] = 0
        
        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=eps] = eps
        
        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
        
    vifp = num/den

    if np.isnan(vifp):
        return 1.0
    else:
        return vifp