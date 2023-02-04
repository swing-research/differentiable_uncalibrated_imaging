import ml_collections

def get_config():
  import configs.base_config as base_config
  config = base_config.get_config()

  # inverse problem parameters
  config.problem = 'radon' # solve 2D CT
  config.angles = 90 # number of measurement view angles
  config.dense_angles = config.angles # number of angles to evaluate
  config.unet_angles = 90 # number of angles used to train Unet
  config.angle_snr = 25.0 # angle uncertainty
  config.measurement_snr = 30.0 # measurement uncertainty
  
  config.img_size = 128 # test image size

  # representation parameters
  config.representation.type = 'spline' # spline measurment representation
  config.representation.deg_x = 18
  config.representation.deg_y = 2

  # logging parameters
  config.total_steps = 5000 # total number of iterations

  # training parameters
  config.fitting_weight = 1.0
  config.consistency_weight = 0.025
  config.rep_learning_rate = 5e-2
  config.parameter_learning_rate = 2e-4
  config.termination_tol = 1e-11

  return config