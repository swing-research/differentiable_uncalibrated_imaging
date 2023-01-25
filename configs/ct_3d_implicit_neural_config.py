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
  
  # representation parameters
  config.representation.type = 'implicit_neural'
  config.representation.input_size = 3
  config.representation.L = 10
  config.representation.num_layers = 7
  config.representation.hidden_size = 256
  config.representation.output_size = 1

  # logging parameters
  config.total_steps = 20000

  # training parameters
  config.y_loss_weight = 10.0
  config.data_fid_weight = 1.0
  config.y_siren_learning_rate = 1e-3
  config.input_learning_rate = 1e-4
  config.termination_tol = 1e-11

  return config