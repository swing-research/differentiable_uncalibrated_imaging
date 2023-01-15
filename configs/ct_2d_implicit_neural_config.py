import ml_collections

def get_config():
  import configs.base_config as base_config
  config = base_config.get_config()

  # inverse problem parameters
  config.problem = 'radon'
  config.angles = 90
  config.dense_angles = config.angles
  config.unet_angles = -1
  config.angle_snr = 45.0
  config.measurement_snr = 30.0

  config.img_size = 128

  # representation parameters
  config.representation.type = 'implicit_neural'
  config.representation.input_size = 2
  config.representation.L = 10
  config.representation.hidden_size = 256
  config.representation.num_layers = 7
  config.representation.output_size = 1

  # logging parameters
  config.total_steps = 20000

  # training parameters
  config.y_loss_weight = 10.0
  config.data_fid_weight = 1.0
  config.consistency_weight = 0
  config.y_siren_learning_rate = 5e-4
  config.input_learning_rate = 5e-4
  config.y_amsgrad_flag = True
  config.unet_input_grad = False
  config.data_fidelity = True
  config.termination_tol = 1e-10

  return config