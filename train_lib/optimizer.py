import torch

def get_optimizer(config, model_dict):
  
  if config.representation.type == 'spline':
    spline = model_dict.get('measurement_rep')

    if config.problem == 'radon_3d':
      weights_lr = 0
    elif config.problem == 'radon':
      weights_lr = config.y_siren_learning_rate

    optimizer = torch.optim.Adam([
      {"params": [spline.control_pts], "lr": config.y_siren_learning_rate},
      {"params": [spline.weights], "lr": weights_lr},
      {"params": [spline.u_spline_space], "lr": config.input_learning_rate},
      ])

  elif config.representation.type == 'implicit_neural':
    optimizer = torch.optim.Adam([
      {"params": model_dict.get('measurement_rep').parameters(),
       "lr": config.y_siren_learning_rate,
       "amsgrad": config.y_amsgrad_flag},
      {"params": [model_dict.get('input_parameters')],
       "lr": config.input_learning_rate}
      ])

  else:
    raise ValueError('Representation type unrecognized.')

  return optimizer