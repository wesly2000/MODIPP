import cupy as cp

gaussKernel = cp.ElementwiseKernel(
        'float32 x, float32 mu, float32 s',
        'float32 result',
        'result = exp(-(x-mu)**2 / (2**s))'
        'gaussKernel'
        )
        