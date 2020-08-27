# -*- coding: utf-8 -*-

"""Wavelet tessellation.

Copyright © Geolearn, Jerome Simon
https://github.com/GeolearnAI/wavelet_tesselation
"""

from os import listdir

import pywt
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage.color import rgb2grey, rgb2hsv


# Play with one contact and try to see what works better

im1 = np.array(Image.open(r'LEM-37\Export_10cm_slices\245.2m-LEM-37.jpg'))
im2 = np.array(Image.open(r'LEM-37\Export_10cm_slices\251.1m-LEM-37.jpg'))
im = np.concatenate((im1, im2), axis=0)
im = im[:, 150:500, :]
im.shape

plt.imshow(im1[:1000, 150:500, :])


# Moving average

im = rgb2grey(im)
mean_value = np.mean(im, axis=1)
mean_df = pd.DataFrame(mean_value, columns=['mean'])
# mean_df['mean'] = mean_value
mean_df['1cm'] = mean_df['mean'].rolling(window=100, min_periods=0, center=True).mean()
mean_df['10cm'] = mean_df['mean'].rolling(window=1000, min_periods=0, center=True).mean()
mean_df['100cm'] = mean_df['mean'].rolling(window=10000, min_periods=0, center=True).mean()
mean_df['500cm'] = mean_df['mean'].rolling(window=50000, min_periods=0, center=True).mean()

fig, axes = plt.subplots(1, 5, figsize=(5, 6))
axes[0].plot(mean_df['mean'], mean_df.index.values/100, linestyle='-')
axes[1].plot(mean_df['1cm'], mean_df.index.values/100, linestyle='-')
axes[2].plot(mean_df['10cm'], mean_df.index.values/100, linestyle='-')
axes[3].plot(mean_df['100cm'], mean_df.index.values/100, linestyle='-')
axes[4].plot(mean_df['500cm'], mean_df.index.values/100, linestyle='-')
for i in range(len(axes)):
    axes[i].invert_yaxis()
    axes[i].get_xaxis().set_visible(False)
for i in range(len(axes)-1):
    axes[i+1].get_yaxis().set_visible(False)
plt.show()


# Moving standard deviation

mean_df['10cm_var'] = mean_df['mean'].rolling(window=1000, min_periods=0, center=True).std()
mean_df['100cm_var'] = mean_df['mean'].rolling(window=10000, min_periods=0, center=True).std()
mean_df['500cm_var'] = mean_df['mean'].rolling(window=50000, min_periods=0, center=True).std()

fig, axes = plt.subplots(1, 4, figsize=(6, 6))
axes[0].plot(mean_df['mean'], mean_df.index.values / 100, linestyle='-')
axes[1].plot(mean_df['10cm_var'], mean_df.index.values / 100, linestyle='-')
axes[2].plot(mean_df['100cm_var'], mean_df.index.values / 100, linestyle='-')
axes[3].plot(mean_df['500cm_var'], mean_df.index.values / 100, linestyle='-')
for i in range(len(axes)):
    axes[i].invert_yaxis()
    axes[i].get_xaxis().set_visible(False)
for i in range(len(axes) - 1):
    axes[i + 1].get_yaxis().set_visible(False)


# Discrete wavelet

wp = pywt.WaveletPacket(data=mean_df['mean'], wavelet='db1', mode='symmetric', maxlevel=5)
fig = plt.figure()
plt.set_cmap('bone')
ax = fig.add_subplot(wp.maxlevel + 1, 1, 1)
ax.plot(mean_df['mean'], 'k')
ax.set_xlim(0, len(mean_df['mean'] - 1))
ax.set_title("Wavelet packet coefficients")

for level in range(1, wp.maxlevel + 1):
    ax = fig.add_subplot(wp.maxlevel + 1, 1, level + 1)
    nodes = wp.get_level(level, "freq")
    nodes.reverse()
    labels = [n.path for n in nodes]
    values = -abs(np.array([n.data for n in nodes]))
    ax.imshow(values, interpolation='nearest', aspect='auto')
    ax.set_yticks(np.arange(len(labels) - 0.5, -0.5, -1), labels)
    plt.setp(ax.get_xticklabels(), visible=False)

plt.show()

wp = pywt.WaveletPacket(data=mean_df['mean'], wavelet='db1', mode='symmetric')

# Continuous Gaussian wavelet and its derivative (Mexican Hat wavelet)

coef, freqs = pywt.cwt(data=mean_df['mean'], scales=np.arange(1, 200), wavelet='gaus1')
coef = np.rot90(coef, 3)

fig, axes = plt.subplots(1, 4, figsize=(8, 6))
axes[0].plot(mean_df['mean'], mean_df.index.values/100, linestyle='-')
axes[1].plot(coef[:, 5], range(len(coef[:, 5])), linestyle='-')
axes[2].plot(coef[:, 10], range(len(coef[:, 10])), linestyle='-')
axes[3].matshow(coef, cmap='inferno', extent = (0, 25000, 0, len(mean_df['mean'])))
for i in range(len(axes) - 1):
    axes[i].set_ylim([0,axes[i].lines[0].get_ydata().max()])
    axes[i+1].get_yaxis().set_visible(False)
for i in range(len(axes)):
    axes[i].invert_yaxis()
    axes[i].get_xaxis().set_visible(False)

coef, freqs = pywt.cwt(data=mean_df['mean'], scales=np.arange(1,200), wavelet='mexh')
coef = np.rot90(coef, 3)

fig, axes = plt.subplots(1, 4, figsize=(6, 6))
axes[0].plot(mean_df['mean'], mean_df.index.values/100, linestyle='-')
axes[1].plot(coef[:, 5], range(len(coef[:, 5])), linestyle='-')
axes[2].plot(coef[:, 10], range(len(coef[:, 10])), linestyle='-')
axes[3].matshow(coef, cmap='inferno', extent = (0, 20000, 0, len(mean_df['mean'])))
for i in range(len(axes)-1):
    axes[i].set_ylim([0,axes[i].lines[0].get_ydata().max()])
    axes[i+1].get_yaxis().set_visible(False)
for i in range(len(axes)):
    axes[i].invert_yaxis()
    axes[i].get_xaxis().set_visible(False)

x = np.arange(512)
y = np.sin(2*np.pi*x/32)
coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
plt.matshow(coef)
plt.show()


# ## Entire LEM-18 drillhole

# ### Get geological log

# In[ ]:


def make_litho_grid(From, To, Litho, step):
    ###### using data to make an empty numpy array
    length = (np.max(To) - np.min(From)) / step
    grid = np.zeros(shape=(int(length), 1))
    ##### transformation of litho names into numbers
    Litho = Litho.replace(to_replace=Litho.unique(), value=np.arange(1, len(Litho.unique()) + 1, 1), inplace=False)
    ##### combining log data arrays into single dataframe
    log_litho = pd.concat((From, To, Litho), axis=1)
    log_litho.columns = ['From', 'To', 'Litho']
    for i in range(int(length)):
        depth_step = np.min(From) + i * step
        litho_step = log_litho.loc[(log_litho['From'] < depth_step) & (log_litho['To'] >= depth_step), ['Litho']]
        if litho_step.empty:
            grid[i, :] = np.nan
        else:
            litho_step = litho_step.iloc[0, 0]
            grid[i, :] = litho_step
    return grid


# In[ ]:


class_set = pd.read_csv('LEM-18\Geological_log_LEM-18.csv', sep=';')
litho_colors = ['#000000', '#0066ff', '#ffff00','#663300','#ff9900']
cmap_litho = colors.ListedColormap(litho_colors[0:len(litho_colors)], 'indexed')
grid = make_litho_grid(class_set['from'], class_set['to'], class_set['litho_simplified'], 1)


# Concatenate image while converting and applying mean

onlyfiles = listdir('LEM-37\Export_10cm_slices')

im = np.empty([1000 * len(onlyfiles), 1], dtype=np.float32)

for i, image in enumerate(onlyfiles):
    temp = np.array(Image.open('LEM-37/Export_10cm_slices/' + image))
    temp = temp[:, 150:500, :]
    temp = rgb2grey(temp)
    temp = np.mean(temp, axis=1).reshape([-1, 1])
    im[i * 1000:(i + 1) * 1000] = temp

fig, axes = plt.subplots(1, 1, figsize=(1, 20))
axes.imshow(im, interpolation='none', aspect='auto', cmap=mpl.cm.gray)
plt.show()


# Discrete wavelets

mean_df = pd.DataFrame(mean_array, columns=['mean'])
# mean_df['mean'] = mean_value
mean_df['1cm']   = mean_df['mean'].rolling(window=100,   min_periods=0, center=True).mean()
mean_df['10cm']  = mean_df['mean'].rolling(window=1000,  min_periods=0, center=True).mean()
mean_df['100cm'] = mean_df['mean'].rolling(window=10000, min_periods=0, center=True).mean()
mean_df['500cm'] = mean_df['mean'].rolling(window=50000, min_periods=0, center=True).mean()

fig, axes = plt.subplots(1, 6, figsize=(5, 6))
axes[0].imshow(grid, interpolation='none', aspect='auto', cmap=cmap_litho, vmin=1, vmax=len(litho_colors))
axes[1].plot(mean_df['mean'],  mean_df.index.values / 100, linestyle='-')
axes[2].plot(mean_df['1cm'],   mean_df.index.values / 100, linestyle='-')
axes[3].plot(mean_df['10cm'],  mean_df.index.values / 100, linestyle='-')
axes[4].plot(mean_df['100cm'], mean_df.index.values / 100, linestyle='-')
axes[5].plot(mean_df['500cm'], mean_df.index.values / 100, linestyle='-')
for i in range(len(axes)-1):
    axes[i + 1].set_ylim([0,axes[i + 1].lines[0].get_ydata().max()])
    axes[i + 1].get_yaxis().set_visible(False)
    # axes[i+1].invert_yaxis()
for i in range(len(axes)):
    axes[i].get_xaxis().set_visible(False)
plt.show()

indexes = [5, 14, 15, 16, 17, 18, 19, 20, 21, 22]

wp = pywt.WaveletPacket(data=mean_df['mean'], wavelet='db1', mode='symmetric')

fig, axes = plt.subplots(1, len(indexes) + 3, figsize=(10, 6))
axes[0].imshow(grid, interpolation='none', aspect='auto', cmap=cmap_litho, vmin=1, vmax=len(litho_colors))
axes[1].plot(mean_df['mean'], mean_df.index.values / 100, linestyle='-')
axes[2].plot(mean_df['500cm'], mean_df.index.values / 100, linestyle='-')
for n, i in enumerate(indexes):
    axes[n + 3].plot(wp['a' * i].data, range(len(wp['a' * i].data)), linestyle='-')
for i in range(len(axes) - 1):
    axes[i+1].set_ylim([0,axes[i + 1].lines[0].get_ydata().max()])
    axes[i+1].get_yaxis().set_visible(False)
    # axes[i+1].invert_yaxis()
for i in range(len(axes)):
    axes[i].get_xaxis().set_visible(False)
plt.show()

indexes = [0, 4, 8, 12, 17, 18, 19, 20]

wp = pywt.WaveletPacket(data=mean_df['mean'], wavelet='db1', mode='symmetric')

fig, axes = plt.subplots(1, len(indexes) + 3, figsize=(10, 6))
axes[0].imshow(grid, interpolation='none', aspect='auto', cmap=cmap_litho, vmin=1, vmax=len(litho_colors))
axes[1].plot(mean_df['mean'], mean_df.index.values / 100, linestyle='-')
axes[2].plot(mean_df['500cm'], mean_df.index.values / 100, linestyle='-')
for n, i in enumerate(indexes):
    axes[n + 3].plot(wp['a' * i + 'd'].data, range(len(wp['a' * i + 'd'].data)), linestyle='-')
for i in range(len(axes) - 1):
    axes[i + 1].set_ylim([0,axes[i + 1].lines[0].get_ydata().max()])
    axes[i + 1].get_yaxis().set_visible(False)
    # axes[i + 1].invert_yaxis()
for i in range(len(axes)):
    axes[i].get_xaxis().set_visible(False)
plt.show()


# Multiscale hierarchical domaining using continuous wavelets

# Padding to avoid border effects (optional)

PAD_FRACTION = .2  # .1 is sufficient

im = im.reshape([-1])
pad = int(PAD_FRACTION * im.shape[0])
im = np.hstack([im[pad:0:-1], im, im[-2:-pad - 2:-1]])


# Continuous wavelets

# scales = np.logspace(1, 3.5, 10)
scales = np.logspace(2., 3., 5)

coefs = pywt.cwt(data=im, scales=scales, wavelet='mexh')
coefs = np.swapaxes(coefs[0], 0, 1);

# Unpad

im = im[pad:-pad]
pad = int(PAD_FRACTION * coefs.shape[0] /(1+PAD_FRACTION*2))
coefs = coefs[pad:-pad]

coef_max = np.amax(coefs)
coefs = coefs / coef_max

fig, axes = plt.subplots(1, 1, figsize=(3, 100))

coef_max = np.amax(coefs)
axes.imshow(coefs, interpolation='nearest', aspect='auto', cmap=mpl.cm.bwr, vmin=-coef_max, vmax=coef_max)
cont = axes.contour(coefs, levels=[0])

plt.show()


# Tessellation

# Children with intensities smaller than FILTER_THRESHOLD are merged (default: 0.15):
FILTER_THRESHOLD = .15
# Accelerates computation by reducing the initial quantity of rectangles (default: 0):
MIN_WIDTH_FILTER = 3

# Get rectangles from contours

paths = cont.collections[0].get_paths()
rectangles = np.empty([len(paths), 4], dtype = np.int32)

for i, path in enumerate(paths):
    v = path.vertices
    x = v[:, 0]
    y = v[:, 1]
    if y[0] > y[-1]:
        y[0], y[-1] = y[-1], y[0]
    rectangles[i] = [np.amin(x), y[0], np.amax(x), y[-1] - y[0]]  # ['x', 'y', 'width', 'height']

rectangles = rectangles[(rectangles[:, 0] == 0) & (rectangles[:, 2] != 0) & (rectangles[:, 3] != 0)]
rectangles = rectangles[rectangles[:, 2] > MIN_WIDTH_FILTER]
rectangles = np.vstack([[0, 0, coefs.shape[1] - 1, coefs.shape[0] - 1], rectangles])

del paths

# Delete overlapping rectangles  # SLOW

bool_ = np.ones(rectangles.shape[0], dtype=bool)

for i1, r1 in enumerate(rectangles):

    for i2, r2 in enumerate(rectangles):

        if r2[2] > r1[2] or i1 == i2 or not bool_[i2]:
            continue

        upper_in = r2[1] >= r1[1] and r2[1] < r1[1] + r1[3]
        bottom_in = r2[1] + r2[3] > r1[1] and r2[1] + r2[3] <= r1[1] + r1[3]

        if upper_in != bottom_in:
            bool_[i2] = False

rectangles = rectangles[bool_]

# Filtering the rectangles
# If a rectangle does not contain an abs(value) greater than FILTER_THRESHOLD, it is deleted.

bool_ = np.ones(rectangles.shape[0], dtype=bool)

for i, r in enumerate(rectangles):
    if not (np.absolute(coefs[r[1]:r[1] + r[3] + 1, r[0]:r[0] + r[2] + 1]) > FILTER_THRESHOLD * coef_max).any():
        bool_[i] = False

shape_ = rectangles.shape[0]
rectangles = rectangles[bool_]
rect_backup = np.copy(rectangles)
print("Rectangles left:", rectangles.shape[0], '/', shape_)

# Restore rectangles array

rectangles = rect_backup

# Create new rectangles for every child

import time
now = time.time()

flag = True
ver_rect = []  # Added vertical rectangles, the parents
non_parents = np.zeros(rectangles.shape[0], dtype=bool)  # Array which increase computation effectiveness
temp = np.copy(rectangles)  # The set of children doesn't need to be modified, as new children won't be wider than the
                            # original children and as parents' max_x remain at their original width.
                            # Also note that temp[n, 0] + temp [n, 2] == temp[n, 2] for every n, as temp[n, 0] == 0.

while flag:  # While there are still horizontal rectangles that are added
    flag = False
    bool_ = np.ones(rectangles.shape[0], dtype=bool)  # Keeps track of elements to delete after the loops
    hor_rect = []  # Added horizontal rectangles, the children

    for i1, r1 in enumerate(rectangles):
        if non_parents[i1]:  # If the rectangle does not have any child
            continue

        children = []

        for i2, r2 in enumerate(temp):

            if r2[1] >= r1[1] and r2[1] < r1[1] + r1[3] and r2[3] < r1[3] and r2[2] <= r1[2]:
                if not (r1 == r2).all():  # Time consuming, add unique_id?
                    children.append(i2)  # r2 is a children

        if len(children) == 0:  # If there is no children
            non_parents[i1] = True
            continue

        upper_x = [temp[i][2] for i in children]
        max_x = max(upper_x)

        children = [temp[i] for i, j in zip(children, upper_x) if j == max_x]  # Gets the eldest children
        children.sort(key=lambda x: x[1])

        if r1[2] - max_x != 0:
            ver_rect += [(max_x, r1[1], r1[2] - max_x, r1[3])]
        bool_[i1] = False

        children = np.vstack([np.array([[0, r1[1], max_x, 0]]),
                              *children,
                              np.array([[0, r1[1] + r1[3], max_x, 0]])])  # Add borders

        for top, bottom in zip(children[:-1], children[1:]):  # Adds a rectangle between each pair of children
            if bottom[1] - top[1] - top[3] != 0:
                hor_rect += [(0, top[1] + top[3], max_x, bottom[1] - top[1] - top[3])]
                flag = True

    rectangles = rectangles[bool_]
    rectangles = np.vstack([rectangles, *hor_rect])  # The new children could be parents; those must be iterated over

    non_parents = non_parents[bool_]
    non_parents = np.hstack([non_parents, np.zeros(len(hor_rect), dtype=bool)])  # Must remain the same length as rectangles

    prec, now = now, time.time()
    print('Time elapsed:      ', int(now - prec), 'seconds')
    print('ver_rect length:   ', len(ver_rect))
    print('rectangles length: ', rectangles.shape[0])
    print('hor_rect length:   ', len(hor_rect), '\n')

rectangles = np.vstack([rectangles, *ver_rect])

del non_parents, ver_rect, hor_rect, bool_, temp

print('Done!')

# Plot rectangles

fig, axes = plt.subplots(1, 1, figsize=(3, 200))

axes.imshow(coefs, interpolation='nearest', aspect='auto', cmap=mpl.cm.bwr, vmin=-coef_max, vmax=coef_max)

p = mpl.collections.PatchCollection([mpl.patches.Rectangle((r[0], r[1]), r[2], r[3]) for r in rectangles])
p.set_facecolor('none')
p.set_edgecolor('k')
axes.add_collection(p)

plt.show()


# Geological log

# Color rectangles

value = np.zeros(rectangles.shape[0])

for i, r in enumerate(rectangles):
    array_ = coefs[r[1]:r[1] + r[3] + 1, r[0]:r[0] + r[2] + 1]
    extremums = (np.amin(array_), np.amax(array_))

    if -extremums[0] > extremums[1]:
        value[i] = extremums[0]
    else:
        value[i] = extremums[1]

fig, axes = plt.subplots(1, 1, figsize=(3, 200))

axes.axis([0, coefs.shape[1] - 1, 0, coefs.shape[0] - 1])
axes.invert_yaxis()

p = mpl.collections.PatchCollection([mpl.patches.Rectangle((r[0], r[1]), r[2], r[3]) for r in rectangles], cmap=mpl.cm.bwr)
p.set_clim([-coef_max, coef_max])
p.set_array(value)
p.set_edgecolor('k')
axes.add_collection(p)

plt.show()

# Plot with geological log

fig, axes = plt.subplots(1, 2, figsize=(10, 60))

axes[0].imshow(grid, interpolation='none', aspect='auto', cmap=cmap_litho, vmin=1, vmax=len(litho_colors))

axes[1].imshow(coefs, interpolation='nearest', aspect='auto', cmap=mpl.cm.bwr, vmin=-coef_max, vmax=coef_max)

p = mpl.collections.PatchCollection([mpl.patches.Rectangle((r[0], r[1]), r[2], r[3]) for r in rectangles], cmap=mpl.cm.bwr)
p.set_clim([-coef_max, coef_max])
p.set_array(value)
p.set_edgecolor('k')
p.set_alpha(0.5)
axes[1].add_collection(p)

plt.show()


# Get boundaries intersecting the x == slice_ axis.

slice_ = 25

boundaries = rectangles[(rectangles[:, 0] <= slice_) & (slice_ < rectangles[:, 0] + rectangles[:, 2])][:, 1]
boundaries.sort()
print(list(boundaries) + [coefs.shape[0]])

fig, axes = plt.subplots(1, 2, figsize=(10, 60))

axes[0].imshow(grid, interpolation='none', aspect='auto', cmap=cmap_litho, vmin=1, vmax=len(litho_colors))

axes[1].imshow(coefs, interpolation='nearest', aspect='auto', cmap=mpl.cm.bwr, vmin=-coef_max, vmax=coef_max)

lines = [[(0, b), (coefs.shape[1] - 1, b)] for b in boundaries]
p = mpl.collections.LineCollection(lines)
p.set_color('k')
axes[1].add_collection(p)

plt.show()


# Get boundaries intersecting the x == slice_ axis.

slice_frac = .5
slice_ = int(coefs.shape[1] * slice_frac)

# boundaries = rectangles[(rectangles[:, 0] <= slice_) & (slice_ < rectangles[:, 0] + rectangles[:, 2])][:, 1]
# boundaries.sort()
# print(list(boundaries) + [coefs.shape[0]])

fig, axes = plt.subplots(1, 2, figsize=(10, 60))

axes[0].imshow(grid, interpolation='none', aspect='auto', cmap=cmap_litho, vmin=1, vmax=len(litho_colors))

# axes[1].imshow(coefs, interpolation='nearest', aspect='auto', cmap=mpl.cm.bwr, vmin=-coef_max, vmax=coef_max)

boundaries = (abs(coefs[:, slice_]) < 1e-3).astype(int).reshape([-1, 1])

axes[1].imshow(boundaries, interpolation='none', aspect='auto', cmap=mpl.cm.binary, vmin=0, vmax=1)

plt.tight_layout()
plt.show()
