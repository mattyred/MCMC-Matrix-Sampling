import numpy as np
import tensorflow as tf
import scipy
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

def wishart_logpdf(L, P):
  l = np.diagonal(L)
  n = L.shape[0]
  return -np.sum(np.log(np.abs(l))) -0.5 * np.trace((1/n)*np.eye(n,n) @ P)

def invwishart_logpdf(L, P):
  P = tf.linalg.inv(P)
  l = np.diagonal(L)
  n = L.shape[0]
  return -(2*n + 1) * np.sum(np.log(np.abs(l))) - np.trace(P) / 2

def logdet_jacobian_neudecker(L, n):
  l = np.diagonal(L)
  exps = np.flip(np.arange(n)+1)
  return n*np.log(2) + np.sum(np.multiply(exps,np.log(np.abs(l))))

def rw_mh_step(Ls_prev_array, step_size, scale, K, jacobian=False):
    # Sampling L from Gaussian density
    Ls_tilde_array = np.random.multivariate_normal(mean=np.zeros(len(Ls_prev_array)), cov=np.eye(len(Ls_prev_array)))
    Ls_tilde_array = Ls_tilde_array * step_size + Ls_prev_array
    # Reshape Ls_tilde and Ls_prev into lower triangular matrices
    Ls_tilde = tfp.math.fill_triangular(Ls_tilde_array, upper=False)
    Ls_prev = tfp.math.fill_triangular(Ls_prev_array, upper=False)
    # Compute precision matrices
    Ps_tilde = Ls_tilde @ tf.transpose(Ls_tilde)
    Ps_prev = Ls_prev @ tf.transpose(Ls_prev)
    # Compute acceptance ratio
    log_gs_tilde = scipy.stats.invwishart.logpdf(Ps_tilde, df=K, scale=scale)
    log_gs_prev = scipy.stats.invwishart.logpdf(Ps_prev, df=K, scale=scale)
    if jacobian:
        logdet_gs_tilde = logdet_jacobian_neudecker(Ls_tilde, Ls_tilde.shape[0])
        logdet_gs_prev = logdet_jacobian_neudecker(Ls_prev, Ls_prev.shape[0])
        log_gs_tilde += logdet_gs_tilde
        log_gs_prev += logdet_gs_prev
    log_ps_prev = scipy.stats.multivariate_normal.logpdf(Ls_prev_array, mean=Ls_tilde_array, cov=np.eye(len(Ls_tilde_array)))
    log_ps_tilde = scipy.stats.multivariate_normal.logpdf(Ls_tilde_array, mean=Ls_prev_array, cov=np.eye(len(Ls_tilde_array)))
    log_ratio = (log_gs_tilde - log_gs_prev) + (log_ps_prev - log_ps_tilde)
    ratio = np.exp(log_ratio)
    if ratio >= 1 or ratio >= np.random.uniform(0, 1):
        return True, Ls_tilde
    else:
        return False, Ls_prev
    
def run_sampling(Ls_prev_array, n_samples, step_size, scale, K, jacobian=False):
    samples = []
    Ls_array = Ls_prev_array
    accepted_samples = []
    for _ in range(n_samples):
        accepted, Ls = rw_mh_step(Ls_array, step_size, scale, K, jacobian=jacobian)
        samples.append(Ls @ tf.transpose(Ls))
        accepted_samples.append(int(accepted))
        Ls_array = tfp.math.fill_triangular_inverse(Ls, upper=False)
    return accepted_samples, samples

def plot_mcmc_marginals(n, V, K, samples_mcmc, samples_actual):
  samples_mcmc_merged = np.empty((n, n), dtype=object)
  samples_actual_merged = np.empty((n, n), dtype=object)
  samples_mcmc_mean = np.empty((n, n))
  samples_mcmc_var = np.empty((n, n))
  samples_actual_mean = np.empty((n, n))
  samples_actual_var = np.empty((n, n))
  for i in range(n):
      for j in range(n):
          samples_mcmc_merged[i,j] = [mat[i, j].numpy() for mat in samples_mcmc]
          samples_actual_merged[i,j] = [mat[i, j] for mat in samples_actual]
          samples_mcmc_mean[i,j] = np.mean(samples_mcmc_merged[i, j])
          samples_mcmc_var[i,j] = np.var(samples_mcmc_merged[i, j])
          samples_actual_mean[i,j] = np.mean(samples_actual_merged[i,j])
          samples_actual_var[i,j] = np.var(samples_actual_merged[i,j])
  true_mean = V / (K-n-1)
  def moving_average(x, w):
      return np.convolve(x, np.ones(w), 'valid') / w
  fig, axes = plt.subplots(n, n, figsize=(20, 10), sharex=True, sharey=True)
  for i in range(n):
      for j in range(n):
          ax = axes[i, j]
          mv_avg = moving_average(samples_mcmc_merged[i,j],10)
          ax.plot(np.arange(0,len(mv_avg),1),mv_avg,linewidth=0.1, marker='o', markersize=.3)
          ax.axhline(true_mean[i,j], color='red')
          ax.axhline(samples_mcmc_mean[i,j], color='blue')
          ax.set_xlabel(r"iteration")
          ax.set_ylabel(r"$\Lambda_{%d%d}$"%(i,j))
          #ax.set_title(r"Samples from $p(\mathbf{\Lambda}|\mathbf{V},k) $")
  plt.show()
  return samples_mcmc_mean, samples_mcmc_var, samples_actual_mean, samples_actual_var