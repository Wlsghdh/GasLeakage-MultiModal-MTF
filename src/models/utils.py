import torch
import pywt
from scipy.fftpack import dct

def apply_dwt(x):
    x = x.view(x.size(0), -1)
    coeffs = []
    
    for i in range(x.size(0)):
        try:
            feature_data = x[i].detach().cpu().numpy()
            if len(feature_data) >= 4:
                c, _ = pywt.dwt(feature_data, 'haar')
                coeffs.append(torch.tensor(c, dtype=torch.float32))
            else:
                coeffs.append(x[i])
        except Exception as e:
            coeffs.append(x[i][:max(1, x[i].size(0) // 2)])
    return torch.stack(coeffs).to(x.device)


def apply_dct(x):
    try:
        x_np = x.detach().cpu().numpy()
        if x_np.shape[1] > 0:
            result = dct(x_np, norm='ortho', axis=1)
            return torch.tensor(result, dtype=torch.float32).to(x.device)
        else:
            return x
    except Exception as e:
        return x

