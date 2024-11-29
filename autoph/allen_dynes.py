#!/usr/bin/env python

import os
import numpy as np

class AllenDynes(object):
  def __init__(self, a2f_file="a2F.dos10", mu=0.1):
    self.a2f_file = a2f_file
    self.mu = mu
    self.w = None
    self.a2f = None
    self._Tc = None
    self._la = None
    self._wlog = None
    self._w2 = None
    assert(os.path.isfile(self.a2f_file))
    self.read_a2F()
  
  @property
  def la(self):
    if self._la is None:
      self.get_epc()
    return self._la

  @property
  def wlog(self):
    if self._wlog is None:
      self.get_epc()
    return self._wlog

  @property
  def w2(self):
    if self._w2 is None:
      self.get_epc()
    return self._w2

  @property
  def Tc(self):
    if self._Tc is None:
      self.get_Tc()
    return self._Tc

  def read_a2F(self):
    """ Reads a2F file

    """  
    with open(self.a2f_file) as f:
      lines = f.readlines()

    lines = [[float(j) for j in i.split()] for i in lines[5:-1]]
    # remove negative frequencies
    lines = np.array([i for i in lines if i[0] > 0])
    self._w = lines[:, 0]
    self._a2f = lines[:, 1]
  
  def get_epc(self):
    """ Calculate EPC constants
    
    """
    from scipy.integrate import simpson
    K_BOLTZMANN_RY = (1.380649E-23/(4.3597447222071E-18/2))

    a2f = self._a2f
    w = self._w
    la = 2*simpson(a2f/w, x=w)
    sim = simpson(a2f/w*np.log(w), x=w)
    self._wlog = np.exp(2/la*sim)/K_BOLTZMANN_RY

    sim = max(0.0, simpson(2*a2f*w, x=w))
    self._w2 = np.sqrt(sim/la)/K_BOLTZMANN_RY
    self._la = la
    return self._wlog, self._w2, self._la

  def get_Tc(self):
    """ Computed Tc with both Allen-Dynes and improved formula
    
    """
    # if lambda is too small Allen-Dynes yields nonesense
    if self.la > 1.5*self.mu:
      f1 = (1 + (self.la/(2.46*(1+3.8*self.mu)))**(3/2))**(1/3)
      f2 = 1 + (self.la**2*(self.w2/self.wlog -1))/(self.la**2+(1.82*(1+6.3*self.mu)*(self.w2/self.wlog))**2)

      ad = self.wlog/1.2*np.exp(-(1.04*(1+self.la))/(self.la-self.mu*(1+0.62*self.la)))
      iad = f1*f2*ad
    else:
      ad = 0.0
      iad = 0.0
    self._Tc = [ad, iad]
    return ad, iad

if __name__ == "__main__":
  ad = AllenDynes("a2F.dos10")
  wlog, w2, la = ad.get_epc()
  tc1, tc2 = ad.Tc

  print("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(wlog, w2, la, tc1, tc2))
