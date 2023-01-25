import torch
import numpy as np

from .implicit_neural import FourierNet
from .spline import NURBS2D, NURBS3D
from .unet_3d import UNet

def get_model_dict(config, input_parameters, guess_recon, spline_info):  
  if config.problem == 'radon_3d':
    unet = UNet(in_channels=1, 
      out_channels=1, 
      n_blocks=4, 
      start_filts=16,
      activation='relu',
      normalization='batch',
      conv_mode='same',
      dim=3)
    checkpoint = torch.load(f'unets/unet_3d/model_{config.unet_angles}angles_measurementsnr{int(config.measurement_snr)}dB.pt')
  elif config.problem == 'radon':
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 
      'unet',
      in_channels=1, 
      out_channels=1, 
      init_features=16, 
      pretrained=False)
    checkpoint = torch.load(f'unets/unet_2d/model_{config.unet_angles}angles_measurementsnr{int(config.measurement_snr)}dB.pt')
  else:
    raise ValueError('Did not recognize problem')
  unet.load_state_dict(checkpoint['model_state_dict'])
  unet.eval()
  model_dict={'unet': unet.cuda()}

  if config.representation.type == 'implicit_neural':
    measurement_rep = FourierNet(
      in_features=config.representation.input_size,
      out_features=config.representation.output_size,
      hidden_features=config.representation.hidden_size,
      hidden_blocks=config.representation.num_layers,
      L = config.representation.L)
  elif config.representation.type == 'spline':
    if config.problem == 'radon':
      _, num_ctrl_pts_v = spline_info['y_measured'].shape
      angles_spline_space = spline_info['angles_spline_space']
      angles_spline_space = np.concatenate(
        (angles_spline_space[-config.representation.deg_x:] - np.pi, 
          angles_spline_space, 
          angles_spline_space[:config.representation.deg_x] + np.pi), axis=-1)
      angles_shift = -(np.min(angles_spline_space)) + 1e-3
      angles_spline_space += angles_shift
      angles_scale = np.max(angles_spline_space) + 1e-3
      angles_spline_space /= angles_scale
      spline_info['angles_shift'] = angles_shift
      spline_info['angles_scale'] = angles_scale
      detectors_spline_space = np.linspace(0, 1, num_ctrl_pts_v)
      X, Y = np.meshgrid(angles_spline_space, detectors_spline_space, indexing='ij')
      Z = spline_info['y_measured'].detach().cpu().numpy()
      Z = np.concatenate(
        (np.flip(Z[-config.representation.deg_x:], axis=-1), 
          Z, 
          np.flip(Z[:config.representation.deg_x], axis=-1)), axis=0)
      inp_ctrl_pts = torch.from_numpy(np.array([X,Y,Z])).permute(1,2,0).unsqueeze(0).contiguous()
      weights = torch.ones(1, len(angles_spline_space), num_ctrl_pts_v, 1)
      measurement_rep = NURBS2D(
        inp_ctrl_pts,
        weights, 
        angles_spline_space,
        detectors_spline_space,
        config.representation.deg_x, 
        config.representation.deg_y)

      # Used in loss. Only measurements (Z) are actually used. Check if other dims are required.
      target = torch.FloatTensor(np.array([X,Y,Z])).permute(1,2,0).unsqueeze(0).cuda()
      spline_info.update({'spline_target': target})
    
    elif config.problem == 'radon_3d':
      _, num_ctrl_pts_v, num_ctrl_pts_w = spline_info['y_measured'].shape
      angles_spline_space = spline_info['angles_spline_space'] + 0
      angles_shift = -(np.min(angles_spline_space)) + 1e-3
      angles_spline_space += angles_shift
      angles_scale = np.max(angles_spline_space) + 1e-3
      angles_spline_space /= angles_scale
      spline_info['angles_shift'] = angles_shift
      spline_info['angles_scale'] = angles_scale
      detectors_v_spline_space = np.linspace(0, 1, num_ctrl_pts_v)
      detectors_w_spline_space = np.linspace(0, 1, num_ctrl_pts_w)
      W, X, Y = np.meshgrid(angles_spline_space, detectors_v_spline_space, detectors_w_spline_space, indexing='ij')
      Z = spline_info['y_measured'].detach().cpu().numpy()
      inp_ctrl_pts = torch.from_numpy(np.array([W,X,Y,Z])).permute(1,2,3,0).unsqueeze(0).contiguous()
      weights = torch.ones(1, len(angles_spline_space), num_ctrl_pts_v, num_ctrl_pts_w, 1)
      measurement_rep = NURBS3D(
        inp_ctrl_pts,
        weights,
        angles_spline_space,
        detectors_v_spline_space,
        detectors_w_spline_space,
        config.representation.deg_x,
        config.representation.deg_y,
        config.representation.deg_z)

      # Used in loss. Only measurements (Z) are actually used. Check if other dims are required.
      target = torch.FloatTensor(np.array([W,X,Y,Z])).permute(1,2,3,0).unsqueeze(0).cuda()
      spline_info.update({'spline_target': target})

    else:
      raise ValueError('Invalid inverse problem')
  else:
    raise ValueError('Invalid representation type')

  model_dict.update({'measurement_rep': measurement_rep.cuda()})

  input_parameters.requires_grad_(True)
  if config.representation.type == 'spline':
    input_parameters = model_dict['measurement_rep'].u_spline_space * np.pi

  model_dict.update({'input_parameters': input_parameters})
  model_dict.update({'guess_recon': guess_recon})

  return model_dict