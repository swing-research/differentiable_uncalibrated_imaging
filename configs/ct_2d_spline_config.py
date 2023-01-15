import ml_collections
import numpy as np

def get_config():
  import configs.base_config as base_config
  config = base_config.get_config()

  # inverse problem parameters
  config.problem = 'radon'
  config.angles = 90
  config.dense_angles = config.angles
  config.unet_angles = -1
  config.angle_snr = 25.0
  config.measurement_snr = 30.0
  
  config.img_size = 128

  # representation parameters
  config.representation.type = 'spline'
  config.representation.deg_x = 18
  config.representation.deg_y = 2

  # logging parameters
  config.total_steps = 5000

  # training parameters
  config.opt_strat = 'broken_machine'
  config.y_loss_weight = 1.0
  config.data_fid_weight = 0.025
  config.consistency_weight = 0
  config.y_siren_learning_rate = 5e-2
  config.input_learning_rate = 2e-4
  config.unet_input_grad = False
  config.data_fidelity = True
  config.termination_tol = 1e-11

  return config