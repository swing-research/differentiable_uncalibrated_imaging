import numpy as np
import torch

import utils

def get_grid_fn(config):
  if config.problem == 'radon':
    grid_fn = utils.get_sinogram_mgrid

  elif config.problem == 'radon_3d':
    grid_fn = utils.get_sinogram_3d_mgrid

  else:
    raise ValueError('Inverse problem unrecognized.')

  return grid_fn

def get_loss_fn(config, operators, models, spline_info):
  
  original_operator = operators.get('original_operator')
  dense_operator = operators.get('dense_operator')

  unet = models.get('unet')
  measurement_rep = models.get('measurement_rep')

  grid_fn = get_grid_fn(config)

  if config.problem == 'radon':
    dense_grid_params = {
      'angles': torch.cos(torch.tensor(dense_operator.angles, dtype=torch.float32)),
      'num_detectors': dense_operator.num_detectors
    }

  elif config.problem == 'radon_3d':
    dense_grid_params = {
      'angles': torch.tensor(dense_operator.angles, dtype=torch.float32),
      'num_detectors_x': dense_operator.num_detectors_x,
      'num_detectors_y': dense_operator.num_detectors_y
    }

  else:
    raise ValueError('Inverse problem unrecognized.')

  dense_grid = grid_fn(**dense_grid_params).cuda()

  def loss_fn(y_measured, return_model_out=False):
    if config.problem == 'radon':
      og_grid_params = {
        'angles': torch.cos(models['input_parameters']),
        'num_detectors': original_operator.num_detectors
      }

    elif config.problem == 'radon_3d':
      og_grid_params = {
        'angles': models['input_parameters'],
        'num_detectors_x': original_operator.num_detectors_x,
        'num_detectors_y': original_operator.num_detectors_y
      }

    else:
      raise ValueError('Inverse problem unrecognized.')

    y_max = torch.max(y_measured)

    if config.representation.type != 'spline':
      og_grid = grid_fn(**og_grid_params).cuda()

    # Measurement loss term
    if config.representation.type == 'spline':
      y_output = measurement_rep()
      y_measured = spline_info['spline_target']
      if config.problem == 'radon':
        y_loss = ((y_output - y_measured)[0,:,:,2]**2).mean()
      elif config.problem == 'radon_3d':
        y_loss = ((y_output - y_measured)[0,:,:,:,3]**2).mean()
      else:
        raise ValueError('Inverse problem unrecognized.')
    else:
      y_output = measurement_rep(og_grid)
      y_output = torch.reshape(y_output, y_measured.shape)  
      y_loss = ((y_output - y_measured / y_max)**2).mean()

    # Data-fidelity loss term
    if config.representation.type == 'spline':
      if config.problem == 'radon':
        _, num_ctrl_pts_v = spline_info['y_measured'].shape
        dense_angles_spline_space = spline_info['dense_angles_spline_space'] + 0
        dense_angles_spline_space += spline_info['angles_shift']
        dense_angles_spline_space /= spline_info['angles_scale']
        detectors_spline_space = np.linspace(0, 1, num_ctrl_pts_v)
        dense_measurements_from_rep = measurement_rep.eval(dense_angles_spline_space, detectors_spline_space)
        dense_measurements_from_rep = dense_measurements_from_rep.reshape(1, len(dense_angles_spline_space), num_ctrl_pts_v, 3)[0,:,:,2]
      elif config.problem == 'radon_3d':
        _, num_ctrl_pts_v, num_ctrl_pts_w = spline_info['y_measured'].shape
        dense_angles_spline_space = spline_info['dense_angles_spline_space'] + 0
        dense_angles_spline_space += spline_info['angles_shift']
        dense_angles_spline_space /= spline_info['angles_scale']
        detectors_v_spline_space = np.linspace(0, 1, num_ctrl_pts_v)
        detectors_w_spline_space = np.linspace(0, 1, num_ctrl_pts_w)
        dense_measurements_from_rep = measurement_rep.eval(
          dense_angles_spline_space, detectors_v_spline_space, detectors_w_spline_space)
        dense_measurements_from_rep = dense_measurements_from_rep.reshape(
          1, len(dense_angles_spline_space), num_ctrl_pts_v, num_ctrl_pts_w, 4)[0,:,:,:,3]
      else:
        raise ValueError('Inverse problem unrecognized.')
    else:
      dense_measurements_from_rep = measurement_rep(dense_grid)

    if config.problem == 'radon':
      interpolated_fbp = dense_operator.pinv(
        torch.reshape(dense_measurements_from_rep, 
          (1, 1, dense_operator.num_angles, dense_operator.num_detectors)) * y_max)
    elif config.problem == 'radon_3d':
      interpolated_fbp = dense_operator.pinv(
        torch.reshape(dense_measurements_from_rep, 
          (1, 1, dense_operator.num_angles, dense_operator.num_detectors_x, dense_operator.num_detectors_y)) * y_max)
    else:
      raise ValueError('Inverse problem unrecognized.')

    unet.eval()
    model_output = unet(interpolated_fbp)
    unet.train()

    if config.problem == 'radon':
      recon_img_size = [config.img_size, config.img_size]
    elif config.problem == 'radon_3d':
      recon_img_size = [config.img_size, config.img_size, config.img_size]
    dense_measurements_from_op = torch.reshape(
      dense_operator(torch.reshape(model_output, recon_img_size)),
      dense_measurements_from_rep.shape)

    dense_loss = ((
      dense_measurements_from_rep - dense_measurements_from_op / y_max)**2).mean()

    loss = config.y_loss_weight * y_loss + config.data_fid_weight * dense_loss
    
    loss_dict = {
      'measurement_loss': y_loss, 
      'dense_loss': dense_loss, 
      'total_loss': loss
    }

    if return_model_out: 
      ret = (loss_dict, model_output)
    else:
      ret = loss_dict

    return ret

  return loss_fn      
