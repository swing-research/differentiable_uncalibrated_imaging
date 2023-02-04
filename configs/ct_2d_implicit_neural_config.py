import ml_collections

def get_config():
  import configs.base_config as base_config
  config = base_config.get_config()

  # inverse problem parameters
  config.problem = 'radon' # solve 2D CT
  config.angles = 90 # number of measurement view angles
  config.dense_angles = config.angles # number of angles to evaluate
  config.unet_angles = 90 # number of angles used to train Unet
  config.angle_snr = 45.0 # angle uncertainty
  config.measurement_snr = 30.0 # measurement uncertainty

  config.img_size = 128 # test image size

  # representation parameters
  config.representation.type = 'implicit_neural' # implicit neural measurment representation
  config.representation.input_size = 2
  config.representation.L = 10
  config.representation.hidden_size = 256
  config.representation.num_layers = 7
  config.representation.output_size = 1

  # logging parameters
  config.total_steps = 20000 # total number of iterations

  # training parameters
  config.fitting_weight = 10.0
  config.consistency_weight = 1.0
  config.rep_learning_rate = 5e-4
  config.parameter_learning_rate = 5e-4
  config.termination_tol = 1e-10

  return config