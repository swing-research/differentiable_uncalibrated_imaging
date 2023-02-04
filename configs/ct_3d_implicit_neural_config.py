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
  
  # representation parameters
  config.representation.type = 'implicit_neural' # implicit neural measurment representation
  config.representation.input_size = 3
  config.representation.L = 10
  config.representation.num_layers = 7
  config.representation.hidden_size = 256
  config.representation.output_size = 1

  # logging parameters
  config.total_steps = 20000 # total number of iterations

  # training parameters
  config.fitting_weight = 10.0
  config.consistency_weight = 1.0
  config.rep_learning_rate = 1e-3
  config.parameter_learning_rate = 1e-4
  config.termination_tol = 1e-11

  return config