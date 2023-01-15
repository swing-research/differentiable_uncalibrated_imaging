import os
import torch
import ml_collections
import numpy as np
from scipy.interpolate import interp2d
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from absl import logging
from absl import flags

import utils
import operators
import models
import train_lib


FLAGS = flags.FLAGS

def trainer(config, ground_truth):

  total_steps = config.total_steps

  operator_dict = operators.get_operator_dict(config)
  operator = operator_dict['original_operator']

  ground_truth = ground_truth.cuda()
  with torch.no_grad():
    ground_truth_size = [config.img_size, config.img_size]
    if config.problem == 'radon_3d':
      ground_truth_size = [config.img_size, config.img_size, config.img_size]
    y_measured = operator(torch.reshape(ground_truth, ground_truth_size))
    y_measured = utils.perturb_measurements(
      y_measured, config.measurement_snr)
    y_max = torch.max(y_measured)
    logging.info(f'Measurement shape: {y_measured.shape}')
    logging.info(f'Measured y max: {y_measured.max()}, min: {y_measured.min()}')
    
    original_operator_clean = operator_dict['original_operator_clean']
    x_fbp = original_operator_clean.pinv(y_measured)

  if config.representation.type == 'spline':
    spline_info = {
    'y_measured'                : y_measured / y_max,
    'angles_spline_space'       : original_operator_clean.angles,
    'dense_angles_spline_space' : operator_dict['dense_operator'].angles
    }
  else:
    spline_info = None
  model_dict = models.get_model_dict(config,
    input_parameters=operator.optimizable_params,
    guess_recon=x_fbp,
    spline_info=spline_info)
  optimizer = train_lib.get_optimizer(config, model_dict)

  logging.info('Beginning training.')
  writer = SummaryWriter(log_dir=config.workdir)
  store = utils.MetricsStore(total_steps)

  loss_fn = train_lib.get_loss_fn(config, operator_dict, model_dict, spline_info)

  ### Baseline
  unet = model_dict['unet']
  unet.eval()
  if config.problem == 'radon':
    recon_size = (1, 1, config.img_size, config.img_size)
  elif config.problem == 'radon_3d':
    recon_size = (1, 1, config.img_size, config.img_size, config.img_size)
  x_baseline = unet(torch.reshape(x_fbp, recon_size))
  x_baseline_snr = utils.SNR(ground_truth.cpu().detach().numpy().flatten(), x_baseline.cpu().detach().numpy().flatten())
  logging.info(f'Baseline SNR is {x_baseline_snr}')
  x_fbp_snr = utils.SNR(ground_truth.cpu().detach().numpy().flatten(), x_fbp.cpu().detach().numpy().flatten())
  logging.info(f'FBP SNR is {x_fbp_snr}')
  np.save(config.workdir + '/fbp.npy', x_fbp.cpu().detach().numpy().reshape(recon_size).squeeze())
  np.save(config.workdir + '/baseline.npy', x_baseline.cpu().detach().numpy().reshape(recon_size).squeeze())
  if config.problem == 'radon_3d':
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x_fbp.cpu().detach().numpy().reshape(recon_size).squeeze()[:,config.img_size//2], cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(x_baseline.cpu().detach().numpy().reshape(recon_size).squeeze()[:,config.img_size//2], cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(ground_truth.cpu().detach().numpy().reshape(recon_size).squeeze()[:,config.img_size//2], cmap='gray')
    plt.savefig(f'{config.workdir}/fbp_baseline.pdf')
  del unet

  loss_prev = np.inf
  for step in range(total_steps):
    loss_dict, model_output = loss_fn(y_measured, return_model_out=True)
    loss = loss_dict.get("total_loss")

    writer.add_scalars('losses', loss_dict, global_step=step+1)
    if config.problem == 'radon':
      writer.add_image('recon',
        torch.reshape(model_output, ground_truth_size).unsqueeze(0), global_step=step+1)
    elif config.problem == 'radon_3d':
      writer.add_image('recon1',
        torch.reshape(model_output, ground_truth_size)[config.img_size//2].unsqueeze(0), global_step=step+1)
      writer.add_image('recon2',
        torch.reshape(model_output, ground_truth_size)[:,config.img_size//2].unsqueeze(0), global_step=step+1)
      writer.add_image('recon3',
        torch.reshape(model_output, ground_truth_size)[:,:,config.img_size//2].unsqueeze(0), global_step=step+1)
    else:
      raise ValueError('Invalid inverse problem')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for k, v in loss_dict.items():
      store.update(step,
        {
          k: v.cpu().detach().numpy(),
        })

    with torch.no_grad():
      measurement_rep = model_dict['measurement_rep']
      _, model_output = loss_fn(y_measured, return_model_out=True)
      recon = model_output.cpu().detach().numpy().flatten()
      truth_flat = ground_truth.cpu().detach().numpy().flatten()
      snr_val = utils.SNR(truth_flat, recon)
      if config.representation.type == 'spline':
        if config.problem == 'radon':
          corrected_angles = model_dict['measurement_rep'].u_spline_space.cpu().detach().numpy().flatten()
          corrected_angles *= spline_info['angles_scale']
          corrected_angles -= spline_info['angles_shift']
          corrected_angles = corrected_angles[config.representation.deg_x: -config.representation.deg_x]
        elif config.problem == 'radon_3d':
          corrected_angles = model_dict['measurement_rep'].u_spline_space.cpu().detach().numpy().flatten()
          corrected_angles *= spline_info['angles_scale']
          corrected_angles -= spline_info['angles_shift']
        else:
          raise ValueError('Unknown inverse problem.')
        angle_error = utils.angles_w1_error(
          corrected_angles,
          operator.angles.flatten())
      else:
        angle_error = utils.angles_w1_error(
          model_dict['input_parameters'].cpu().detach().numpy().flatten(),
          operator.angles.flatten())

      store.update(step, {'SNR': snr_val})
      writer.add_scalar('SNR', snr_val, global_step=step+1)
      writer.add_scalar('SNR improvement', snr_val - x_baseline_snr, global_step=step+1)
      writer.add_scalar('angle_error', angle_error, global_step=step+1)

    loss_change = abs(loss - loss_prev)
    loss_prev = loss
    writer.add_scalar('loss_change', loss_change, global_step=step+1)
    
    if (step + 1) % config.summary_frequency == 0:
      # Do summaries here.
      for k, v in loss_dict.items():
        logging.info(f'{k} at {step+1} = {store.get(k)[step]}')
      logging.info(f'SNR at {step+1} = {store.get("SNR")[step]}')
      logging.info(f'Angle error at {step+1} = {angle_error}')
      logging.info(f'Loss change at {step+1} = {loss_change}')
      logging.info(f'SNR improvement at {step+1} = {snr_val - x_baseline_snr}')

      store.save_all(config.workdir)
      store.plot_all(config.workdir)

    if loss_change < config.termination_tol and step > int(0.4*total_steps):
      logging.info(f'Break training. Loss change at {step+1} = {loss_change}')
      break

  logging.info(f'Max SNR at {np.argmax(store.get("SNR")) + 1} of {np.max(store.get("SNR"))}')
  for k, v in loss_dict.items():
    logging.info(f'Min {k} at {np.argmin(store.get(k)) + 1} of {np.min(store.get(k))}')

  store.save_all(config.workdir)
  store.plot_all(config.workdir)

  _, model_output = loss_fn(y_measured, return_model_out=True)
  recon = model_output.cpu().detach().numpy().flatten()
  truth_flat = ground_truth.cpu().detach().numpy().flatten()

  writer.flush()
  writer.close()

  torch.save({
    'step': config.total_steps,
    'model_type': config.representation.type,
    'measurement_rep': measurement_rep.state_dict(),
    'optimizer': optimizer.state_dict(),
    'loss': loss
    }, os.path.join(config.workdir, 'model.pt'))

  return recon, truth_flat
