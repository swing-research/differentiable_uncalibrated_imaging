import ml_collections

def get_config():
  import configs.base_config as base_config
  config = base_config.get_config()

  # inverse problem parameters
  config.problem = 'radon_3d'
  config.angles = 60
  config.dense_angles = config.angles
  config.unet_angles = config.angles
  config.angle_snr = 25.0
  config.measurement_snr = 35.0

  config.img_size = 64

  # network parameters
  config.representation.type = 'spline'
  config.representation.deg_x = 15
  config.representation.deg_y = 2
  config.representation.deg_z = 2

  # logging parameters
  config.total_steps = 5000

  # training parameters
  config.y_loss_weight = 1.0
  config.data_fid_weight = 0.6
  config.y_siren_learning_rate = 5e-3
  config.input_learning_rate = 1e-4
  config.termination_tol = 1e-11

  return config