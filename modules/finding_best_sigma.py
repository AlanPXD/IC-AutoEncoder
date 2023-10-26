from scipy.ndimage import gaussian_filter
from tensorflow.python.ops.image_ops_impl import ssim


def find_best_sigma_for_ssim (x, y) -> float:
    """
        Essa função encontra o sigma do filtro gaussiano que maximiza o ssim para uma base de dados.
    """
    def calc_ssim(sigma, x, y):
        gauss_imgs = gaussian_filter(x, sigma=(0,sigma,sigma,0))
        ssim_gauss = ssim(gauss_imgs.astype('uint8'), y.astype('uint8'), max_val=255).numpy()
        mean = ssim_gauss.mean()

        return mean

    end = False
    sigma = 0.2
    step = 0.2
    previous_mean = 0

    print(sigma)

    while (not end):

        sigma += step
        
        mean_ssim = calc_ssim(sigma, x, y)

        if (mean_ssim > previous_mean):
            previous_mean = mean_ssim
        else:
            sigma -= 2*step

            mean_ssim = calc_ssim(sigma, x, y)
            
            if (mean_ssim > previous_mean):
                previous_mean = mean_ssim
            else:
                sigma += step
                step = step/2
                

        if (step < 0.0001):
            end = True

    return sigma


        



