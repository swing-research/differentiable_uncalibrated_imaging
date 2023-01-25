"""
Sample command:
python main.py \
--config=configs/ct_2d_implicit_neural_config.py \
--config.workdir=./logs/trial

"""

from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags

import torch
import os
import numpy as np
import random

import utils
import trainer

from skimage.io import imread
from skimage.transform import resize

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config')

def main(argv):

  config = FLAGS.config
  # Setup logging
  logging.get_absl_handler().use_absl_log_file(
    'logs.out', config.workdir
  )

  logging.info(FLAGS.config)

  # Seed fixer: Fixes all sources of randomness
  seed = config.seed
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
  np.random.seed(seed)  # Numpy module.
  random.seed(seed)  # Python random module.
  torch.manual_seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

  # Check if working directory in config exists before training
  if (os.path.isdir(config.workdir) is False):
    raise FileNotFoundError(
      f'Working directory {config.workdir} does not exist')

  if config.problem == 'radon_3d':
    image = np.load(f'3d_volumes/{config.img_index}.npy').astype(np.float32)
  elif config.problem == 'radon':
    image = np.load('lodopab/test.npy').astype(np.float32)
    image = image[config.img_index, 0]
  else:
    raise ValueError('Did not recognize problem')

  recon, gt = trainer.trainer(
    config,
    torch.from_numpy(image))

  shape = config.img_size, config.img_size
  if config.problem == 'radon_3d':
    shape = config.img_size, config.img_size, config.img_size

  recon = recon.reshape(shape)
  gt = gt.reshape(shape)

  np.save(config.workdir + '/gt.npy', gt)
  np.save(config.workdir + '/recon.npy', recon)

  snr_val = utils.SNR(gt, recon)

  logging.info(f'Final SNR: {snr_val}')

  if config.problem == 'radon':
    fig = utils.plot_imgs([recon, gt],
      ['Reconstruction', 'Ground truth'],
      path=config.workdir + '/final.pdf')
  elif config.problem == 'radon_3d':
    fig = utils.plot_imgs([
      recon[config.img_size//2],
      recon[:,config.img_size//2],
      recon[:,:,config.img_size//2],
      gt[config.img_size//2],
      gt[:,config.img_size//2],
      gt[:,:,config.img_size//2]
      ],
      ['Reconstruction', 'Reconstruction', 'Reconstruction',
      'Ground truth', 'gt2', 'gt3'],
      path=config.workdir + '/final.pdf')


if __name__ == '__main__':
  app.run(main)
