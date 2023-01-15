import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  config.workdir = ''
  
  config.seed = 0
  config.img_index = 0

  # logging parameters
  config.summary_frequency = 100

  # training parameters
  config.learning_rate = 1e-4

  # representation parameters
  config.representation = ml_collections.ConfigDict()
  
  return config