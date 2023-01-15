import numpy as np
import torch

from absl import logging

from .operators_lib import ParallelBeamGeometryOp, ParallelBeamGeometryOpBroken
from .operators_lib import ParallelBeamGeometry3DOp, ParallelBeamGeometry3DOpBroken

def get_operator_dict(config):

  if config.problem == 'radon':

    dense_operator = ParallelBeamGeometryOp(
      config.img_size,
      config.angles,
      angle_snr=np.inf)
    operator_dict = {'original_operator_clean': dense_operator}
    
    broken_operator = ParallelBeamGeometryOpBroken(dense_operator, config.angle_snr)
    operator_dict.update({'original_operator': broken_operator})

    dense_operator = ParallelBeamGeometryOp(
      config.img_size,
      config.dense_angles,
      angle_snr=np.inf)
    operator_dict['dense_operator'] = dense_operator
    
    logging.info(f'operator_dict: {operator_dict}')

  elif config.problem == 'radon_3d':
    operator_dict = {}

    dense_operator = ParallelBeamGeometry3DOp(config.img_size, config.angles, angle_snr=np.inf)
    operator_dict = {'original_operator_clean': dense_operator}
    
    broken_operator = ParallelBeamGeometry3DOpBroken(dense_operator, config.angle_snr)
    operator_dict.update({'original_operator': broken_operator})

    dense_operator = ParallelBeamGeometry3DOp(config.img_size, config.dense_angles, angle_snr=np.inf)
    operator_dict['dense_operator'] = dense_operator
    
    logging.info(f'operator_dict: {operator_dict}')

  else:
    raise ValueError('Inverse problem unrecognized.')

  return operator_dict