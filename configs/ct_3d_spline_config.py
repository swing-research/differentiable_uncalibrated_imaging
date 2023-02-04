import ml_collections

def get_config():
  import configs.base_config as base_config
  config = base_config.get_config()

  # inverse problem parameters
  config.problem = 'radon_3d' # solve 3D CT
  config.angles = 60 # number of measurement tilt angles
  config.dense_angles = config.angles # number of angles to evaluate
  config.unet_angles = config.angles # number of angles used to train Unet
  config.angle_snr = 25.0 # angle uncertainty
  config.measurement_snr = 35.0 # measurement uncertainty

  config.img_size = 64 # test volume size

  # network parameters
  config.representation.type = 'spline' # spline measurment representation
  config.representation.deg_x = 15
  config.representation.deg_y = 2
  config.representation.deg_z = 2

  # logging parameters
  config.total_steps = 5000 # total number of iterations

  # training parameters
  config.y_loss_weight = 1.0
  config.data_fid_weight = 0.6
  config.y_siren_learning_rate = 5e-3
  config.input_learning_rate = 1e-4
  config.termination_tol = 1e-11

  return config