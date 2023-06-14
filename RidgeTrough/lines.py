from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from netCDF4 import Dataset
from tqdm import tqdm
import numpy as np
import datetime
from scipy import signal
import re
import os
import gc

def grads(hgts, sigma=(5,3), verbose=False, gc_freq=1000):
    grad_tensor = np.zeros(hgts.shape)
    for day in tqdm(range(hgts.shape[0]), desc='getting gradients'):
        if day % gc_freq == 0: gc.collect() # temporary memory leak solution?
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
    '''
    get data from a single year
    '''
    fh = Dataset(path, mode='r')
    lons = fh.variables['lon'][:].data
    lats = fh.variables['lat'][:].data
    times = fh.variables['time'][:].data
    times = times - times[0]
    data = fh.variables['hgt'][:].data
    time = datetime.datetime.now().strftime('%H:%M:%S')
    
    return (times, lats, lons, data)

def get_data(hgt_path, yr_range = None, verbose=True):
    '''
    get data from multiple years
    '''
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


class CritPointFinder():
    def __init__(self, prominence=1.5, distance=4):
        self.prominence = prominence
        self.distance = distance

    def __call__(self, data): # convenience method
        return self.find(data)
    
    def find(self, data):
        # [day, level, lat]
        if isinstance(data, FullCDFData):
            return self._find_3d(data.grad_lines)
        elif len(data.shape) == 3:
            return self._find_3d(data)
        elif len(data.shape) == 2:
            return self._find_2d(data)
        elif len(data.shape) == 1:
            return self._find_1d(data)
        else:
            assert False
    
    def _find_1d(self, data):
        assert len(data.shape) == 1
        troughs = signal.find_peaks(-data, distance=self.distance, prominence=self.prominence)[0]
        peaks = signal.find_peaks(data, distance=self.distance, prominence=self.prominence)[0]
        return peaks, troughs

    def _find_2d(self, data):
        assert len(data.shape) == 2
        peaks = []
        troughs = []
        for i in range(data.shape[0]):
            _peaks, _troughs = self._find_1d(data[i,:])
            peaks.append(_peaks)
            troughs.append(_troughs)
        
        return peaks, troughs
    
    def _find_3d(self, data):
        peaks = []
        troughs = []
        for i in range(data.shape[0]):
            _peaks, _troughs = self._find_2d(data[i,:,:])
            peaks.append(_peaks)
            troughs.append(_troughs)
        
        return peaks, troughs

def plot_crit_points(peaks, troughs, data, day, level):
    fig, ax = plt.subplots()
    ax.contourf(data.hgts[day,level,...], cmap='YlOrRd', levels=10)
    ax.scatter(troughs[day][level], data.grad_lines[day, level][troughs[day][level]], marker='x',s=100)
    ax.scatter(peaks[day][level], data.grad_lines[day, level][peaks[day][level]], marker='x', s=100,color='purple')
    ax.axis('off')
    return fig, ax