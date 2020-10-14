import numpy as np
import os
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

from ska_sdc import Sdc1Scorer
from ska_sdc.sdc1.utils.crossmatch import crossmatch_kdtree
from ska_sdc.sdc1.utils.prep import prepare_data
from ska_sdc.sdc1.utils.sieve import process_kdtree_cand_df

def add_ellipticity_cols(cat):
  ''' Add (in place) ellipticity columns to an SKA SDC1 catalogue,

  Ellipticity is calculated according to the GalSim reduced shear,
  as describe https://galsim-developers.github.io/GalSim/_build/html/shear.html

  g1 = (a - b) / (a + b)

  Args:
    cat : the sdc1_scorer data frame

  '''
    
  cat['q'] = cat['b_min']/cat['b_maj']
  cat['modg'] = (cat['b_maj'] - cat['b_min']) / (cat['b_maj'] + cat['b_min'])
  cat['g1'] = cat['modg'] * np.cos(2. * np.deg2rad(cat['pa']))
  cat['g2'] = cat['modg'] * np.sin(2. * np.deg2rad(cat['pa']))

  cat['q_t'] = cat['b_min_t']/cat['b_maj_t']
  cat['modg_t'] = (cat['b_maj_t'] - cat['b_min_t']) / (cat['b_maj_t'] + cat['b_min_t'])
  cat['g1_t'] = cat['modg_t'] * np.cos(2. * np.deg2rad(cat['pa_t']))
  cat['g2_t'] = cat['modg_t'] * np.sin(2. * np.deg2rad(cat['pa_t']))


def get_linear_bias(cat, cname='_t'):
  ''' Calculate a linear bias model for the truth vs submitted ellipticity values
  for an SKA SDC1 catalogue, with added ellipticity columns.

  Args:
    cat : the sdc1_scorer data frame

  Returns:
    a dictionary of linear bias model coefficients in g1 and g2 and their uncertainties,
    along with Pearson's R correlation coefficients between the truth and submitted
    g1 and g2

  '''

  def flin(x, m, c):
    return m*x + c
  
  cat = cat.dropna()

  popt_e1, pcov_e1 = curve_fit(flin, cat['g1'], cat['g1'+cname])
  popt_e2, pcov_e2 = curve_fit(flin, cat['g2'], cat['g2'+cname])
  perr_e1 = np.sqrt(np.diag(pcov_e1))
  perr_e2 = np.sqrt(np.diag(pcov_e2))

  m_e1 = popt_e1[0]
  c_e1 = popt_e1[1]
  sigma2_m_e1 = perr_e1[0]**2.
  sigma2_c_e1 = perr_e1[1]**2.

  m_e2 = popt_e2[0]
  c_e2 = popt_e2[1]
  sigma2_m_e2 = perr_e2[0]**2.
  sigma2_c_e2 = perr_e2[1]**2.

  R_e1 = np.corrcoef(cat['g1'], cat['g1'+cname])[0,1]
  R_e2 = np.corrcoef(cat['g2'], cat['g2'+cname])[0,1]

  return {'m_g1' : m_e1,
          'm_g2' : m_e2,
          'c_g1' : c_e1,
          'c_g2' : c_e1,
          'sigma2_m_g1' : sigma2_m_e1,
          'sigma2_m_g2' : sigma2_m_e2,
          'sigma2_c_g1' : sigma2_c_e1,
          'sigma2_c_g2' : sigma2_c_e2,
          'R_g1' : R_e1,
          'R_g2' : R_e2,
          }

def make_ellipticity_bias_plot(sub_cat_path, truth_cat_path,
                               freq,
                               train=False,
                               sub_skiprows=13,
                               out_dir='./plots/'):

  scorer = Sdc1Scorer.from_txt(
                             sub_cat_path,
                             truth_cat_path,
                             freq=9200,
                             sub_skiprows=sub_skiprows,
                             truth_skiprows=0
                            )

  sub_cat_name = os.path.splitext(os.path.basename(sub_cat_path))[0]
  truth_cat_name = os.path.splitext(os.path.basename(truth_cat_path))[0]

  scorer.run()

  sub_df_prep = prepare_data(scorer.sub_df, freq, train)
  truth_df_prep = prepare_data(scorer.truth_df, freq, train)

  cand_sub_df = crossmatch_kdtree(sub_df_prep, truth_df_prep, 0)

  sieved_sub_df = process_kdtree_cand_df(cand_sub_df, 0)

  add_ellipticity_cols(sieved_sub_df)
  mandc = get_linear_bias(sieved_sub_df)

  plt.figure(figsize=(4.5, 3.75))

  x = np.linspace(-1,1,64)
  y_e1 = mandc['m_g1'] * x + mandc['c_g1']
  y_e2 = mandc['m_g2'] * x + mandc['c_g2']

  plt.scatter(sieved_sub_df['g1_t'], sieved_sub_df['g1'], color='powderblue', alpha=0.6)
  plt.plot(x, y_e1, c='powderblue', label='$e_1, \lbrace m,c,R \\rbrace = \lbrace {0:.2f}\pm{3:.2f},{1:.2f}\pm{4:.2f},{2:.2f} \\rbrace $'.format(1-mandc['m_g1'], mandc['c_g1'], mandc['R_g1'], np.sqrt(mandc['sigma2_m_g1']), np.sqrt(mandc['sigma2_c_g1'])))

  plt.scatter(sieved_sub_df['g2_t'], sieved_sub_df['g2'], color='lightcoral', alpha=0.6)
  plt.plot(x, y_e2, c='lightcoral', label='$e_2, \lbrace m,c,R \\rbrace = \lbrace {0:.2f}\pm{3:.2f},{1:.2f}\pm{4:.2f},{2:.2f} \\rbrace $'.format(1-mandc['m_g2'], mandc['c_g2'], mandc['R_g2'], np.sqrt(mandc['sigma2_m_g2']), np.sqrt(mandc['sigma2_c_g2'])))

  plt.xlabel('Input $e^{\\rm in}$')
  plt.ylabel('Output $e^{\\rm out}$')
  plt.xlim([-1,1])
  plt.ylim([-1,1])
  plt.legend(fontsize='x-small', loc='upper left')
  plt.savefig(out_dir+'/{0}-ellipticitybias.png'.format(sub_cat_name), dpi=300, bbox_inches='tight')

  plt.figure(figsize=(2*4.5, 3.75))

  plt.subplot(121)
  plt.hist(sieved_sub_df['modg'], histtype='step', color='powderblue', label='Submission')
  plt.hist(sieved_sub_df['modg_t'], histtype='step', color='k', label='Truth')
  plt.xlabel('$|e|$')
  plt.legend(fontsize='x-small')
  plt.subplot(122)
  plt.hist(sieved_sub_df['pa'], histtype='step', color='powderblue')
  plt.hist(sieved_sub_df['pa_t'], histtype='step', color='k')
  plt.xlabel('PA [deg]')
  plt.savefig(out_dir+'/{0}-mod_e-pa.png'.format(sub_cat_name), dpi=300, bbox_inches='tight')
