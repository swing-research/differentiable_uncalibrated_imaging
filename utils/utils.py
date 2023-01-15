import torch
import numpy as np
import matplotlib.pyplot as plt

def perturb_measurements(truth, target_SNR):
  # Scale perturbation so that perturbation is of specified SNR
  if target_SNR > 100:
    return truth

  perturbation = torch.randn_like(truth)

  original_shape = perturbation.shape
  
  truth = truth.flatten()
  perturbation = perturbation.flatten()
  truth_norm = torch.linalg.norm(truth)
  perturbation_norm = torch.linalg.norm(perturbation)
  
  k = 1 / ((perturbation_norm / truth_norm)*(10**(target_SNR/20)))
  perturbation = k*perturbation
  
  return torch.reshape(truth + perturbation, original_shape)

def get_sinogram_mgrid(angles, num_detectors):
  """Generates a flattened grid of (t,theta_1,...) coordinates in a range of -1 to 1.

  Args:
    angles (1D torch.Tensor): Angles of the radon operator.
    num_detectors (int): The number of detectors.
  Returns:
    mgrid (torch.Tensor): 
  """
  detector_tensors = torch.linspace(-1, 1, steps=num_detectors)
  mgrid = torch.stack(torch.meshgrid(
    angles, detector_tensors, indexing='ij'), dim=2)
  mgrid = mgrid.reshape(-1, 2)

  return mgrid

def get_sinogram_3d_mgrid(angles, num_detectors_x, num_detectors_y):
  detector_x_tensors = torch.linspace(-1, 1, steps=num_detectors_x)
  detector_y_tensors = torch.linspace(-1, 1, steps=num_detectors_y)
  mgrid = torch.stack(torch.meshgrid(
    angles, detector_x_tensors, detector_y_tensors, indexing='ij'), dim=3)
  mgrid = mgrid.reshape(-1, 3)

  return mgrid

def plot_imgs(imgs, titles, path=None):
  """Saves images in a row plot with corresponding titles."""
  assert len(imgs) == len(titles), "Number of images do not match number of titles."

  N = len(imgs)
  fig, ax = plt.subplots(1, N, figsize = (7, 5*N))

  for i, ax_ in enumerate(ax):
    im = ax_.imshow(imgs[i], cmap='gray')
    ax_.axis('off')
    ax_.set_title(titles[i])
    # fig.colorbar(im, ax=ax_)

  if path is not None:
    plt.savefig(path, bbox_inches='tight')

  return fig

def SNR(x, xhat):
  diff = x - xhat
  return -20*np.log10(np.linalg.norm(diff)/ np.linalg.norm(x))

def angles_w1_error(true, estimate):
  return np.mean(np.abs(np.sort(true) - np.sort(estimate)))

class MetricsStore():
  ## Stores a metric in a numpy array of dimension [steps x metric size]

  def __init__(self, total_steps):
    self.data = {}
    self.total_steps = total_steps

  def __add_metric(self, name, metric_shape):
    metric_shape.insert(0, self.total_steps)
    item = np.empty(metric_shape)
    item[:] = np.nan

    self.data.update({name: item})

  def update(self, step, current_metrics):
    for m in current_metrics:
      val = current_metrics[m]
      if (m not in self.data):
        self.__add_metric(m, list(val.shape))

      self.data[m][step] = val

  def get(self, name):
    return self.data[name]

  def save_all(self, path):
    for m in self.data:
      np.save(path + '/' + m + '.npy', self.data[m])

  def plot_all(self, path):
    for m in self.data:
      plt.figure()
      plt.plot(self.data[m])
      plt.ylabel(m)
      plt.grid(True, which='major')
      plt.grid(True, which='minor', alpha=0.4, linestyle='--')
      plt.minorticks_on()
      plt.tight_layout()
      plt.savefig(path + '/' + m + '.pdf')
      plt.close()
