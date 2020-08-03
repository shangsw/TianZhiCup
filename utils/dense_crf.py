# -*- coding: utf-8 -*-
import pydensecrf.densecrf as dcrf
import numpy as np

def dense_crf(probs, img=None, **config):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      
    Returns:
      Refined predictions after MAP inference.
    """
    #config
    kernel_gaussian=dcrf.DIAG_KERNEL
    normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC
    kernel_bilateral=dcrf.DIAG_KERNEL
    normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC
    n_iters=config['n_iters']
    sxy_gaussian=config['sxy_gaussian']
    compat_gaussian=config['compat_gaussian']
    sxy_bilateral=config['sxy_bilateral']
    compat_bilateral=config['compat_bilateral']
    srgb_bilateral=config['srgb_bilateral']
    
    
    h, w, n_classes = probs.shape
    
    probs = probs.transpose(2, 0, 1).copy(order='C') # Need a contiguous array.
    probs += 1e-10

    d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.
    U = -np.log(probs) # Unary potential.
    U = U.reshape((n_classes, -1)) # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)

    if img is not None:
        assert(img.shape[:2] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img)

    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)

    return preds