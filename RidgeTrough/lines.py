from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from netCDF4 import Dataset
from tqdm import tqdm
import numpy as np
import datetime
import re
import os
import gc

def grads(hgts, sigma=(5,3), verbose=False, gc_freq=1000):
    grad_tensor = np.zeros(hgts.shape)
    for day in tqdm(range(hgts.shape[0]), desc='getting gradients'):
        if day % gc_freq == 0: gc.collect()
        for level in range(hgts.shape[1]):
            tmp_grad = ndimage.gaussian_gradient_magnitude(hgts[day, level,...], sigma=sigma)
            grad_tensor[day, level] = tmp_grad
    
    return grad_tensor

def norm(v):
    return (v - v.min()) / (v.max() - v.min())

def max_grads(grad_tensor, sigma=(0.5,0.1,1)):
    maxs = np.argmax(grad_tensor, axis=2).astype(float)
    return ndimage.gaussian_filter(maxs, sigma, mode='wrap')

def ridge_gif(hgts, smooth_max, fig_width=12, level=5):
    AR = hgts.shape[-1] / hgts.shape[-2]
    fig_height = fig_width / AR
    fig, ax = plt.subplots(figsize=(fig_width,fig_height))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    h = hgts[0, level, ...]
    im = ax.imshow(norm(h), cmap='YlOrRd', interpolation='bilinear')
    ln, = ax.plot(smooth_max[0, level, :])

    def init():
        hgts_i = hgts[0, level, ...]
        smooth_max_i = smooth_max[0, level, :]
        ln.set_data(range(0,len(smooth_max_i)),smooth_max_i)
        im.set_data(norm(hgts_i))
        return im,ln

    def update(frame):
        hgts_i = hgts[frame, level, ...]
        smooth_max_i = smooth_max[frame, level, :]
        ln.set_data(range(0,len(smooth_max_i)),smooth_max_i)
        im.set_data(norm(hgts_i))
        return im, ln

    return FuncAnimation(fig, update, frames=range(1,hgts.shape[0]), init_func=init, blit=True)


def year_range(hgt_path):
    files = os.listdir(hgt_path)
    min_year = float('inf')
    max_year =  float('-inf')
    hgt_re = r'^hgt.\d\d\d\d.nc$'
    for file in files:
        m = re.search(hgt_re, file)
        if m is not None:
            year = int(m.group(0).split('.')[1])
            if year < min_year:
                min_year = year
            if year > max_year:
                max_year = year
    
    return min_year, max_year

def retrieve_data(path):
        fh = Dataset(path, mode='r')
        lons = fh.variables['lon'][:].data
        lats = fh.variables['lat'][:].data
        times = fh.variables['time'][:].data
        times = times - times[0]
        data = fh.variables['hgt'][:].data
        time = datetime.datetime.now().strftime('%H:%M:%S')
        
        return (times, lats, lons, data)

def get_data(hgt_path, yr_range = None, verbose=True):
        if yr_range is None: yr_range = year_range(hgt_path)
        years = []
        times = []
        yearly_data = []
        lats, lons = None, None

        for year in tqdm(range(yr_range[0], yr_range[1]+1), desc='retrieving data'):
            t, _lats, _lons, data = retrieve_data(os.path.join(hgt_path, f'hgt.{year}.nc'))
            if lats is None: lats = _lats
            if lons is None: lons = _lons
            years.append(year)
            times.append(t)
            yearly_data.append(data)
        
        return years, times, lats, lons, yearly_data


class FullCDFData():
    def __init__(self, hgt_path, yr_range=None, grad_sigma=(5,3), smooth_sigma=(0.5,0.1,1)):
        
        years, times, lats, lons, yearly_data = get_data(hgt_path, yr_range)
        self.years = years
        self.times = np.concatenate(times)
        _hgts = np.concatenate(yearly_data)
        self.hgts = _hgts[...,:(len(lats)//2), :]
        self.lats = lats[:(len(lats)//2)]
        self.lons = lons
        
        self.grads = grads(self.hgts, sigma=grad_sigma)
        self.grad_lines = max_grads(self.grads, sigma=smooth_sigma)
    
    def update_max_lines(self, sigma):
        self.grad_lines = max_grads(self.grads, sigma=sigma)


class PeakFinder():
    def __init__(self):
        pass