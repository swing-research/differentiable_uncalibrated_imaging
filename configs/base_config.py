import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  config.workdir = ''
  
  config.seed = 0 # random seed
  config.img_index = 0 # test image number

  # logging parameters
  config.summary_frequency = 100 # number of iterations before printing log output

  # representation parameters
  config.representation = ml_collections.ConfigDict()
  
  return config