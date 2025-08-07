###############################################################################
# Parameters to set up before the analysis.
# Hard coded for debug purposes
###############################################################################

# Draw related
#NoDraw = False              # If true, 2D graphs are not plot.
fading_time_us = 20E6		# update time for the plot on the right.
refresh_time_us = 5         # Plot refresh time.

# Cluster related
#cluster_analysis = False    # Enable clustering analysis (CARE: VERY SLOW, ONLY COSMICS)
#cluster_time_us = 1E3
#hamming = 5
#blob_min_radius = 2         # Minimum radius for blobs.

# Hard coded parameters.
resolution = 0.25E-6		# Time resolution (multiple of 0.125E-6 that is Tclk)
fe_alpide = True		    # True if ASIC is ALPIDE; False otherwise.

# Read and save
#speed = 1
save_min = 30			    # How often (mins) an *.eps figure is saved

# Set the Bias value.
vcasn = 25			        # higher vcasn corresponds to lower threshold (inversely proportional)
                            # 0: FE off
                            # 1: Highest possible threshold 835 e-
                            # 63: minimum value

# Sync window (timestamp units).
sync = 2048

# For Trigger mode.
#setTriggered = False        # Enable triggered acquisition after initialization
#TrigDelay = 4000            # setup of trigger parameters. Expressed in clock cycles.
#TrigWindowWidth = 0x8       # Useless in current implementation.

# Other
#already_hit_analysis = False

# Automasking
setAutomask = False         # Enable automasking of noisy pixels.
mask_time_s = 100           # How often (s) automask is performed.
mask_sigma = 5              # How many sigmas are necessary to mask a pixel.
mask_min = 25               # How many hits (minimum) a noisy pixel has.
mask_only_sw = False        # If true, masking is only for visualization, not actual pixel masking.



###############################################################################

"""
Threshold (electrons):
 VCASN  |   e- 
----------------
   0    | fe off
   1    |  835
   4    |  625
   7    |  475
   10   |  400
   16   |  360
   25   |  290
   31   |  220
   37   |  145
   ...
   63   |  min
"""

"""
res_incr 	-> Matrice incrementale: contiene tutti gli hit dall'inizio dell'acquisizione (MATRICE SINISTRA)
res_last	-> Matrice che mostra gli hit negli ultimi 20 secondi - variabile fading_time_us (MATRICE DESTRA)
res_cleaned 	-> Matrice di clustering che è attivata dalla variabile cluster_analysis = True	
"""
import os.path
import sys
import time
import logging
import math
import random
import threading
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import warnings
from pyarcadia.test_multiDevice import Test_multiDevice
from pyarcadia.sequence import Sequence, SubSequence
from pyarcadia.data import ChipData, TriggerStartWord, Pixel, TestPulse
from pyarcadia.FEB_setup import SetupFEBMD3, SetupFEBMD3_1, SetupFEBMD3_3, SetupFEBMD3_4
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson
from scipy.stats import norm
#plt.style.use('seaborn-pastel')
#matplotlib.use("Qt5agg")
warnings.simplefilter('ignore')


past = None



# A cluster contains all the information of the cluster, including whether it
# is a blob or not.
'''
class Cluster:
    time = None
    swtime = None
    pixels = []
    patch = None
    to_remove = False
    width = 0
    height = 0
    blob = False
    bbox = None

    def update_size(self):
        min_x = 511; max_x = 0; min_y = 511; max_y = 0
        for y, x in self.pixels:
            min_x = x if x < min_x else min_x
            min_y = y if y < min_y else min_y
            max_x = x if x > max_x else max_x
            max_y = y if y > max_y else max_y

        self.height = max_y - min_y
        self.width = max_x - min_x
        self.bbox = ( (min_x, min_y), (max_x, max_y) )
        self.blob = self.has_blob()

    def has_blob(self):
        for pixel in self.pixels:
            neighbors = 0

            for neighbor in self.pixels:
                if pixel == neighbor:
                    continue

                if (
                        abs(pixel[0] - neighbor[0]) <= blob_min_radius and
                        abs(pixel[1] - neighbor[1]) <= blob_min_radius
                ):
                    neighbors += 1

            if neighbors > 3.14*blob_min_radius*blob_min_radius:
                return True

        return False

'''
# Color maps for data plotting.
incr_cmap = matplotlib.cm.get_cmap("cool")#.copy()
incr_cmap.set_bad('white',1.)
last_cmap = matplotlib.cm.get_cmap("cool")#.copy()
last_cmap.set_bad('white',1.)

incr_cmap1 = matplotlib.cm.get_cmap("cool")#.copy()
incr_cmap1.set_bad('white',1.)
last_cmap1 = matplotlib.cm.get_cmap("cool")#.copy()
last_cmap1.set_bad('white',1.)

sync_cmap0 = matplotlib.cm.get_cmap("cool")#.copy()
sync_cmap0.set_bad('white',1.)
sync_cmap1 = matplotlib.cm.get_cmap("cool")#.copy()
sync_cmap1.set_bad('white',1.)

# Enable matplotlib interactive mode to zoom.
plt.ion()

# Overwrites test methods to serialize and deserialize words.
def deserialize(self, words):
    if not hasattr(self, "packets") or self.packets is None:
        self.packets = SubSequence()

    print("Deserializing %d words" % len(words))

    for word in words:
        if len(word) == 0:
            continue

        word = word[1:-2]
        word = word.split(", ")

        packet = ChipData()
        try:
            packet.bottom = int(word[0])
            packet.hitmap = int(word[1])
            packet.corepr = int(word[2])
            packet.col = int(word[3])
            packet.sec = int(word[4])
            packet.ser = int(word[5])
            packet.falling = (word[6] == 'True')
            packet.ts = int(word[7])
            packet.ts_fpga = int(word[8])
            packet.ts_sw = int(word[9])
            packet.ts_ext = int(word[10])
            packet.triggerNo = int(word[11])
        except ValueError:
            continue

        self.packets.append(packet)

def serialize(self, detectorNo = None):
    words = []
    for packet in self.packets:
        # Skip trigger start words, right now.
        if isinstance(packet, TriggerStartWord): continue
        words.append( (
            packet.bottom,
            packet.hitmap,
            packet.corepr,
            packet.col,
            packet.sec,
            packet.ser,
            packet.falling,
            packet.ts,
            packet.ts_fpga,
            packet.ts_sw,
            packet.ts_ext,
            packet.triggerNo,
            detectorNo
        ) )

    return words

Test_multiDevice.serialize = serialize
Test_multiDevice.deserialize = deserialize


# Old Version from Andrea. Davide developed an alternative below (probably working less).
"""
def automask(data):
    squashed = data.squash_data()

    masked = 0
    for data in squashed:
        slave_hitmap = data.hitmap & 0xf
        if slave_hitmap != 0:
            self.chip.pcr_cfg(0b11, [data.sec], [data.col], [data.corepr], [0], slave_hitmap)

        master_hitmap = (data.hitmap >> 4) & 0xf
        if master_hitmap != 0:
            self.chip.pcr_cfg(0b11, [data.sec], [data.col], [data.corepr], [1], master_hitmap)

        masked += len(data.get_pixels())

    print(f"Masked {masked} pixels")
"""


def autoscale(num, pad=3, prec=2):
    scales = {
        1e9 : "G",
        1e6 : "M",
        1e3 : "k",
        1 : "",
        1e-3 : "m",
        1e-6 : "u",
        1e-9 : "n",
        1e-12 : "p"
    }

    for scale in scales:
        if num >= scale:
            break

    return ("%0"+str(pad+prec+1)+"."+str(prec)+"f %s") % ( (num/scale), scales[scale] )

# New acquisition or replay?
#filename = False
#if len(sys.argv) > 2:
#    filename = sys.argv[2]
#    if not os.path.isfile(filename):
#        print("File not found.")
#        sys.exit(-1)

# Results
# incremental and last plot for each one of the devices, indepentently
res_incr_0 = np.full((512, 512), 0.0)
res_last_0 = np.full((512, 512), 0.0)
res_incr_1 = np.full((512, 512), 0.0)
res_last_1 = np.full((512, 512), 0.0)

# Histogram to save synced data.
res_sync_0 = np.full((512, 512), 0.0)
res_sync_1 = np.full((512, 512), 0.0)

times = []
last_time = None
count = 0

# parameters for automasking.
average = 0.
RMS = 0.


# WIP! FIXME
def automask(detectorNo = None):
    if detectorNo == 0:
      dataToMask = res_incr_0
    elif detectorNo == 1:
      dataToMask = res_incr_1
    else:
      raise ValueError('Cannot mask detector number: %i' % detectorNo)     

    #print("Entering automask()")
    # average and RMS global to change them properly and not copy.
    global average, RMS
    # Compute sum of all entries, sum of all squares and number of non-zero entries.
    sum = np.nansum(dataToMask)
    sumSq = np.nansum(np.square(dataToMask))
    goodPixs = np.count_nonzero(dataToMask) - np.sum(np.isnan(dataToMask))
    
    # I considered the idea to not use sqrt, but at the end more expensive in terms of cycles.
    # Calculate RMS, average and threshold level for masking.
    average = sum / goodPixs
    RMS = math.sqrt(sumSq / goodPixs - average * average)
    maskThreshold = max(average + mask_sigma * RMS, mask_min)
    
    
    toMask = np.transpose(np.nonzero(dataToMask > maskThreshold))
    #toMask = np.array(toMask.astype(int)
    if len(toMask) > 0:
        print(f"Masking {len(toMask)} pixels in detector {detectorNo}:")
        print(toMask)
        if not mask_only_sw: x.chip[detectorNo].pixel_cfg(toMask.tolist(), mask=True)
        dataToMask[tuple(np.transpose(toMask))] = np.nan
    else:
        print("No noisy pixels in detector %i. Avg = %3.3f, RMS = %3.3f" %(detectorNo, average, RMS))
    


# Plots
# Detector 0
fig, (ax_incr, ax_last) = plt.subplots(1, 2)

ax_incr.get_shared_x_axes().join(ax_incr, ax_last)
ax_incr.get_shared_y_axes().join(ax_incr, ax_last)

ax_incr.set_title("Incremental map - Detector 0")
ax_last.set_title("Most recent hits - Detector 0")
img_last = ax_last.imshow(res_last_0, interpolation='none', origin='lower', vmin=0, vmax=fading_time_us, cmap=last_cmap)
img_incr = ax_incr.imshow(res_incr_0, interpolation='none', origin='lower', cmap=incr_cmap)
cbar_last = plt.colorbar(img_last, orientation='horizontal', ax=ax_last)
cbar_incr = plt.colorbar(img_incr, orientation='horizontal', ax=ax_incr)
plt.show()

# Detector 1
fig1, (ax_incr1, ax_last1) = plt.subplots(1, 2)

ax_incr1.get_shared_x_axes().join(ax_incr1, ax_last1)
ax_incr1.get_shared_y_axes().join(ax_incr1, ax_last1)

ax_incr1.set_title("Incremental map - Detector 1")
ax_last1.set_title("Most recent hits - Detector 1")
img_last1 = ax_last1.imshow(res_last_1, interpolation='none', origin='lower', vmin=0, vmax=fading_time_us, cmap=last_cmap1)
img_incr1 = ax_incr1.imshow(res_incr_1, interpolation='none', origin='lower', cmap=incr_cmap1)
cbar_last1 = plt.colorbar(img_last1, orientation='horizontal', ax=ax_last1)
cbar_incr1 = plt.colorbar(img_incr1, orientation='horizontal', ax=ax_incr1)
plt.show()

# Sync plot.
figsync, (ax_sync0, ax_sync1) = plt.subplots(1, 2)

ax_sync0.get_shared_x_axes().join(ax_sync0, ax_sync1)
ax_sync0.get_shared_y_axes().join(ax_sync0, ax_sync1)

ax_sync0.set_title("Sync map - Detector 0")
ax_sync1.set_title("Sync map - Detector 1")
sync_img_1 = ax_sync1.imshow(res_sync_1, interpolation='none', origin='lower', cmap=sync_cmap1)
sync_img_0 = ax_sync0.imshow(res_sync_0, interpolation='none', origin='lower', cmap=incr_cmap1)
cbar_sync1 = plt.colorbar(sync_img_1, orientation='horizontal', ax=ax_sync1)
cbar_sync0 = plt.colorbar(sync_img_0, orientation='horizontal', ax=ax_sync0)
plt.show()
"""
    # Cluster data only if clustering is enabled
    if cluster_analysis:
        fig_hist, (ax_cltime, ax_clsize) = plt.subplots(1, 2, tight_layout=True)
        _, _, times_hist = ax_cltime.hist([0 for _ in range(20)], bins=30)
        times_fit = ax_cltime.plot(np.arange(20), [0 for _ in range(20)])
        plt.ylim([0, 1])
        ax_cltime.set_ylabel('Probability')
        ax_cltime.set_xlabel(f"Time between events (s)")
        ax_clsize.set_ylabel('Number of clusters')
        ax_clsize.set_xlabel(f"Cluster size in pixels")
        timing_fit = ""

    # If automask is enabled, plot the current population of pixels.
    if setAutomask:
        figPop, axPop = plt.subplots()
        axPop2 = axPop.twinx()
        axPop.hist(res_incr.flatten(), bins = list(range(1, 51)))
        axPop.set_title('Papaya')
        #plt.show()
        #plt.figure()
        

    plt.show()
"""

def visualize(new_clusters, clusters_removed, elapsed):
    """
    if cluster_analysis:
        # Remove old clusters
        for cluster in clusters_removed:
            if cluster.patch:
                cluster.patch.remove()

        # Update existing clusters
        fading_delta = elapsed*1E6/fading_time_us
        for p in reversed(ax_last.patches):
            alpha = p.get_alpha() or 1
            p.set_alpha(alpha-fading_delta) if alpha > fading_delta else p.remove()

        # Add new clusters
        for i, cluster in enumerate(new_clusters):
            color = "limegreen" if not cluster.blob else "red"
            rect = plt.Rectangle((cluster.bbox[0][0]-.5, cluster.bbox[0][1]-.5), cluster.width+1, cluster.height+1, fill=False, color=color, linewidth=2)
            ax_last.add_patch(rect)
            cluster.patch = rect
    """
    # Animate - should be in separate thread
    img_last.set_data(res_last_0)
    cbar_last.update_normal(img_last)

    img_incr.set_data(res_incr_0)
    img_incr.set_clim(vmin=0, vmax=np.nanmax(res_incr_0)+1)
    cbar_incr.update_normal(img_incr)
    
    fig.canvas.draw()
    fig.canvas.start_event_loop(refresh_time_us*1E-6)

    img_last1.set_data(res_last_1)
    cbar_last1.update_normal(img_last1)

    img_incr1.set_data(res_incr_1)
    img_incr1.set_clim(vmin=0, vmax=np.nanmax(res_incr_1)+1)
    cbar_incr1.update_normal(img_incr1)
    
    fig1.canvas.draw()
    fig1.canvas.start_event_loop(refresh_time_us*1E-6)
    #plt.pause(refresh_time_us*1E-6)

    sync_img_1.set_data(res_sync_1)
    cbar_sync1.update_normal(sync_img_1)
    sync_img_1.set_clim(vmin=0, vmax=max(np.nanmax(res_sync_1), 1))
    sync_img_0.set_data(res_sync_0)
    cbar_sync0.update_normal(sync_img_0)
    sync_img_0.set_clim(vmin=0, vmax=max(np.nanmax(res_sync_0),1))

    """
        # If automask is set, update also automask figure
        if setAutomask:
            axPop.cla()
            axPop.hist(res_incr.flatten(), bins = list(range(1, 51)))
            
            # Plot also the gaussian
            axPop2.cla()
            x_axis = np.arange(0, 51, 0.01)
            axPop2.plot(x_axis, norm.pdf(x_axis, average, RMS), color='red')
            #print(f'Average = {average}, RMS = {RMS}')
            #figPop.canvas.draw()
            #figPop.canvas.start_event_loop(refresh_time_us*1E-6)

        if cluster_analysis:
            histo_times()
    """
    global tstHist, timeStampDist, fakePlot, TimestampBins       
        
"""
def histo_times():
    global times_hist, times, times_fit

    # Remove old plots
    ax_cltime.clear()
    ax_clsize.clear()

    sizes = [len(cl.pixels) for cl in cluster_history]
    heights, bins, times_hist = ax_clsize.hist(sizes, bins=np.arange(30))

    #
    # Time between clusters
    #

    # New plot
    heights, bins, times_hist = ax_cltime.hist(times, bins=np.arange(20), density=True)

    # calculate bin centres
    bin_middles = 0.5 * (bins[1:] + bins[:-1])

    # calculate bin centres
    def fit_function(x, mu):
        return poisson.pmf(x, mu)

    # fit with curve_fit
    try:
        parameters, cov_matrix = curve_fit(fit_function, bin_middles, heights)

        # plot poisson-deviation with fitted parameter
        ax_cltime.plot(
            np.arange(20)+0.5,
            fit_function(bins, *parameters),
            marker='o', linestyle='',
            label='Fit result',
        )

        timing_fit = "Timing dispersion fit w/ poissonian distribution: mu = %ss +- %ss" % (autoscale(parameters[0]), autoscale(math.sqrt(cov_matrix[0][0])))
    except (RuntimeError, ValueError, scipy.optimize.OptimizeWarning):
        pass

    ax_cltime.set_ylabel('Probability')
    ax_cltime.set_xlabel(f"Time between events (s)")
    ax_clsize.set_ylabel('Number of clusters')
    ax_clsize.set_xlabel(f"Cluster size in pixels")

    fig_hist.canvas.draw()
    fig_hist.canvas.start_event_loop(refresh_time_us*1E-6)

"""

# Test
#chip_id = 0 if len(sys.argv) < 2 else int(sys.argv[1])
x = Test_multiDevice(chipID1 = 0, chipID2 = 1)

#if filename is False:
#    if(setTriggered):
#        t.chip.send_controller_command('setDaqMode', 0x1)
#        t.chip.send_controller_command('setTrigDelay', TrigDelay)
#        t.chip.send_controller_command('setTrigWindow', TrigWindowWidth)


x.initialize(auto_read=False)
x.set_timestamp_resolution(resolution, False, clk_freq=80E6)
    
    # Switch off the very noisy pixels. They are:
    #   494, 452
    #   32, 0
    #   32, 1
    #   33, 0
    #   33, 1 

#    if(setTriggered):
#        print("Setting Triggered acquisition...")
#        t.chip.send_controller_command('setDaqMode', 0x1)
#else:
#.set_timestamp_resolution(resolution, False, clk_freq=80E6)

for sec in x.lanes_excluded[0]:
    res_incr_0[:, sec*32:(sec+1)*32-1] = np.nan
    res_last_0[:, sec*32:(sec+1)*32-1] = np.nan

for sec in x.lanes_excluded[1]:
    res_incr_1[:, sec*32:(sec+1)*32-1] = np.nan
    res_last_1[:, sec*32:(sec+1)*32-1] = np.nan


# Setup of both devices. Easy one for now.
#SetupFEB7()
#SetupFEB8()
SetupFEBMD3_3(x, 0, vcasn = vcasn)
SetupFEBMD3_4(x, 1, vcasn = vcasn) 

# Single packets that will be overwritten in save mode.
x.packets = SubSequence()

# Needed to initialize filenames. 3 files will be saved Save sync to get the name.
last_save_file0 = None
last_save_file1 = None
last_save_filesync = x.savecsv(last_save_file0, detectorNo = "sync")

"""
else:
    t.loadcsv(filename)
    replay = [SubSequence()]

    epoch = 0
    for p in t.packets:
        if p.ts_ext*resolution > epoch+0.1:
            epoch += 0.1
            print(f"New epoch:  {epoch}. Collected {len(replay[-1])} packets")
            replay.append(SubSequence())

        replay[-1].append(p)

    print(f"Arrived to {epoch*10} s.") 
"""
"""
    fig_hist, (ax_cltime, ax_clsize) = plt.subplots(1, 2, tight_layout=True)
    _, _, times_hist = ax_cltime.hist(times, bins=30)
    times_fit = ax_cltime.plot(np.arange(20), [0 for _ in range(20)])
    plt.ylim([0, 1])
    ax_cltime.set_ylabel('Probability')
    ax_cltime.set_xlabel(f"Time between events (s)")
    ax_clsize.set_ylabel('Number of clusters')
    ax_clsize.set_xlabel(f"Cluster size in pixels")
    timing_fit = ""
    plt.show()
"""

alive = True
slept = 0
past_time = time.time()

past_ts_ext0 = None
past_ts_ext1 = None

clusters = 0
new_clusters = []
clusters_removed = []
cluster_history = []
last_report = time.time()
last_save = time.time()
last_mask0 = time.time()
last_mask1 = time.time()
start_time = time.time()
last_time = time.time()
last_pixels = []
current_pixels = []

"""
if filename is not False:
    init_ts = t.packets[0].ts_ext
    init_time = time.time()
"""

last_ts_sw0 = None
last_triggered0 = None

last_ts_sw1 = 1
last_triggered1 = None

# Initialize last_sync_time
#last_sync_time = 0

newData0 = []
newData1 = []

# Used for sync.
id0 = 0
id1 = 0

# Running parameter to recognize synced events.
runningSync = 0

while True:
    # Readout of chip 0
    try:
        new_pixels = []
        new_pixels_times = {}

        #if filename is False:
        x.chip[0].injection_digital()
        x.chip[0].injection_analog()
        overflow0 = x.chip[0].fifo_overflow_count()
        if overflow0 > 0:
            print(f"Warning: lost {overflow0} packets in chip 0!")
            x.chip[0].fifo_overflow_counter_reset()


        #newData0 = []
        current = SubSequence(x.chip[0].readout(), init_ts_sw=last_ts_sw0, init_triggered = last_triggered0, sw_delta_ts = x.ts_delta_sw[0])

        # Save new data acquired in this readout.
        newData0 += current

        # Update already triggered.
        last_triggered0 = current.timesTriggered
        last_ts_sw0 = current.ts_sw
        this_time = time.time()

        x.packets = current
#            if(setTriggered):
#                current.dump(500)

        if len(current) > 0:
            last_save_file0 = x.savecsv(last_save_file0, detectorNo = 0)

        #else:
        """
            if not hasattr(t, 'packets') or len(t.packets) == 0:
                break

            elapsed_real = time.time() - init_time
            elapsed_sim = (t.packets[0].ts_ext - init_ts)*resolution/speed
        """
        """

            if len(replay) == 0:
                break

            current = replay.pop(0)
        """
        """
            if time.time() - last_time < 0.
            last_time = time.time()
            this_time = elapsed_sim
        """

        #   this_time = time.time()

        # How many ends there have been.
        #howManyEnds = 0

        # Trigger start timestamp
        #TriggerStartTS = -1

        # Give a testpulse just to try.
        #x.chip[0].force_injection()
        #x.chip[0].force_nomask()
        #x.chip[0].send_tp()
        #x.chip[1].force_injection()
        #x.chip[1].force_nomask()
        #x.chip[1].send_tp()
        #time.sleep(1)
        #print("TPs sent.")
        
        if len(current) > 0:
            current_pixels = []
            already_hit = {}
            for packet in current:
                # Do not analyze trigger start words.
                if isinstance(packet, TriggerStartWord): continue

                # Also skip testpulse words (done for debug purposes).
                if isinstance(packet, TestPulse): continue
                
                if past_ts_ext0 is not None:
                    times.append(packet.ts_ext - past_ts_ext0)
                past_ts_ext0 = packet.ts_ext

                pixels = packet.get_pixels()
                

                for pixel in pixels:
                    pixel.ts = packet.ts_ext
                    """
                    if already_hit_analysis:
                        for other in (last_pixels + current_pixels):
                            if pixel.row == other.row and pixel.col == other.col:
                                if (pixel.row, pixel.col) not in already_hit:
                                    already_hit[ (pixel.row, pixel.col) ] = [other.ts]

                                already_hit[ (pixel.row, pixel.col) ].append( pixel.ts )
                     """         
                    # Update data plots.
                    #if(NoDraw == False):
                    res_incr_0[pixel.row][pixel.col] += 1
                    res_last_0[pixel.row][pixel.col] = fading_time_us

                    current_pixels.append(pixel)
                    new_pixels.append( (pixel.row, pixel.col) )
                    new_pixels_times[(pixel.row, pixel.col)] = packet.ts_ext
                        
                          
            """
            if already_hit_analysis:
                for pixel in already_hit:
                    ots = already_hit[pixel].pop(0)
                    print("Pixel %s was originally hit @ %ss and then: " % (pixel, autoscale(resolution*ots)), end="")
                    print(", ".join([autoscale(resolution*(ts-ots))+"s ago" for ts in already_hit[pixel]]))
            """
            last_pixels = current_pixels
            
        
        elapsed_step = this_time - past_time 
        past_time = this_time
        elapsed = this_time - start_time

        # Update res_last
        res_last_0[res_last_0 > 0] -= int(elapsed_step*1E6)

        """
        if cluster_analysis:
            clsearch_start = time.time()
            # Clusterize
            result = np.where(res_last > fading_time_us-cluster_time_us)
            old_pixels = list(zip(result[0], result[1]))

            # Until the pixels have all been checked
            old_clusters = new_clusters
            new_clusters = []
            total_pixels = new_pixels + old_pixels
            while len(new_pixels) > 0:
                cluster_pixels = [new_pixels[0]]
                pixel_idx = 0

                replacement = False
                # Build the new cluster
                while pixel_idx < len(cluster_pixels):
                    for row_delta in range(-hamming, hamming+1):
                        for col_delta in range(-hamming, hamming+1):
                            row, col = cluster_pixels[pixel_idx]
                            needle = (row+row_delta, col+col_delta)

                            # If this pixel is alive, include in current cluster and scan around it
                            if needle in total_pixels:
                                cluster_pixels.append(needle)
                                total_pixels.remove(needle)

                                if needle in new_pixels:
                                    new_pixels.remove(needle)

                                # If it was part of a previous cluster, destroy it
                                for old_cluster in old_clusters:
                                    if needle in old_cluster.pixels:
                                        old_cluster.to_remove = True
                                        replacement = old_cluster

                    pixel_idx += 1

                # Got it!
                newcl = Cluster()
                newcl.swtime = time.time()
                newcl.time = None
                newcl.pixels = list(set(cluster_pixels))
                newcl.update_size()

                if replacement:
                    newcl.time = replacement.time
                else:
                    clusters += 1
                    for pixel in newcl.pixels:
                        if pixel in new_pixels_times:
                            tt = new_pixels_times[pixel]
                            if newcl.time is None or tt < newcl.time:
                                newcl.time = tt

                    # Add for histogram
                    curr = packet.ts_ext*resolution
                    if last_time is not None:
                        diff = curr - last_time
                        if diff == 0.0 and len(times) > 0:
                            times.append(times[-1])
                        else:
                            times.append(diff)
                    last_time = curr

                new_clusters.append(newcl)
                cluster_history.append(newcl)

            to_replace = []
            to_remove = []
            for idx, cluster in enumerate(old_clusters):
                # Remove clusters marked for substitution
                if cluster.to_remove:
                    #print("Cluster %d marked for replacement. Deleting." % idx)
                    to_replace.append(idx)
                    continue

                # Removed clusters with no visible pixels
                delete = True
                for pixel in cluster.pixels:
                    if res_last[pixel[0]][pixel[1]] > 0:
                        delete = False
                        break

                if delete:
                    #print("Cluster %d has no more visible pixels. Deleting." % idx)
                    to_remove.append(idx)

            to_remove = list(set(to_remove + to_replace))

            clusters_removed = []
            for idx in reversed(to_remove):
                clusters_removed.append(old_clusters[idx])
                del old_clusters[idx]

            for idx in reversed(to_replace):

                del cluster_history[idx]


            clsearch_time = time.time() - clsearch_start
            print("Cluster search took %07.3f s", clsearch_time)

        """

        # Update view! (useless?)
        visualize(new_clusters, clusters_removed, elapsed_step)
        
        # Eventually mask
        if setAutomask and this_time - last_mask0 > mask_time_s:
            automask(0)
            last_mask0 = this_time
         

        # Report rates
        """
        if cluster_analysis and this_time - last_report > 30:
            rate_1m = 0; rate_1m_pix = 0
            rate_5m = 0; rate_5m_pix = 0
            rate_1h = 0; rate_1h_pix = 0
            rate_1m_f = 0; rate_1m_f_pix = 0
            rate_5m_f = 0; rate_5m_f_pix = 0
            rate_1h_f = 0; rate_1h_f_pix = 0

            for cluster in reversed(cluster_history):
                if cluster.swtime < this_time-60*60:
                    break

                if cluster.swtime > this_time-60:
                    rate_1m += 1
                    rate_1m_pix += len(cluster.pixels)
                    if not cluster.blob:
                        rate_1m_f += 1
                        rate_1m_f_pix += len(cluster.pixels)

                if cluster.swtime > this_time-60*5:
                    rate_5m += 1
                    rate_5m_pix += len(cluster.pixels)
                    if not cluster.blob:
                        rate_5m_f += 1
                        rate_5m_f_pix += len(cluster.pixels)

                rate_1h += 1
                rate_1h_pix += len(cluster.pixels)
                if not cluster.blob:
                    rate_1h_f += 1
                    rate_1h_f_pix += len(cluster.pixels)

            area_cm2 = (25e-6*25e-6*512*512)/(0.01*0.01)

            time_factor = elapsed if elapsed < 60 else 60
            rate_1m = rate_1m/time_factor/area_cm2
            rate_1m_pix = rate_1m_pix/time_factor/area_cm2
            rate_1m_f = rate_1m_f/time_factor/area_cm2
            rate_1m_f_pix = rate_1m_f_pix/time_factor/area_cm2

            time_factor = elapsed if elapsed < 60*5 else 60*5
            rate_5m = rate_5m/time_factor/area_cm2
            rate_5m_pix = rate_5m_pix/time_factor/area_cm2
            rate_5m_f = rate_5m_f/time_factor/area_cm2
            rate_5m_f_pix = rate_5m_f_pix/time_factor/area_cm2

            time_factor = elapsed if elapsed < 60*60 else 60*60
            rate_1h_err = math.sqrt(rate_1h)/time_factor/area_cm2
            rate_1h = rate_1h/time_factor/area_cm2
            rate_1h_pix_err = math.sqrt(rate_1h_pix)/time_factor/area_cm2
            rate_1h_pix = rate_1h_pix/time_factor/area_cm2
            rate_1h_f_err = math.sqrt(rate_1h_f)/time_factor/area_cm2
            rate_1h_f = rate_1h_f/time_factor/area_cm2
            rate_1h_f_pix_err = math.sqrt(rate_1h_f_pix)/time_factor/area_cm2
            rate_1h_f_pix = rate_1h_f_pix/time_factor/area_cm2

            print("Report @ %.3f s" % (this_time - start_time))
            print("1-minute rates  w/ big clusters. Particle rate: %sHz/cm2 - Pixel rate: %sHz/cm2" % (autoscale(rate_1m), autoscale(rate_1m_pix)))
            print("1-minute rates w/o big clusters. Particle rate: %sHz/cm2 - Pixel rate: %sHz/cm2" % (autoscale(rate_1m_f), autoscale(rate_1m_f_pix)))
            print("5-minute rates  w/ big clusters. Particle rate: %sHz/cm2 - Pixel rate: %sHz/cm2" % (autoscale(rate_5m), autoscale(rate_5m_pix)))
            print("5-minute rates w/o big clusters. Particle rate: %sHz/cm2 - Pixel rate: %sHz/cm2" % (autoscale(rate_5m_f), autoscale(rate_5m_f_pix)))
            print("  1-hour rates  w/ big clusters. Particle rate: %sHz/cm2 - Pixel rate: %sHz/cm2" % (autoscale(rate_1h), autoscale(rate_1h_pix)))
            print("  1-hour rates w/o big clusters. Particle rate: %sHz/cm2 - Pixel rate: %sHz/cm2" % (autoscale(rate_1h_f), autoscale(rate_1h_f_pix)))
            print("")

            print("Report @ %.3f s" % (this_time - start_time))
            print("W/ blooms: Particle rate: %sHz/cm2 +- %sHz/cm2| Pixel rate: %sHz/cm2 +- %sHz/cm2" % (autoscale(rate_1h), autoscale(rate_1h_err), autoscale(rate_1h_pix), autoscale(rate_1h_pix_err)))
            print("W/o blooms:  Particle rate: %sHz/cm2 +- %sHz/cm2| Pixel rate: %sHz/cm2 +- %sHz/cm2" % (autoscale(rate_1h_f), autoscale(rate_1h_f_err), autoscale(rate_1h_f_pix), autoscale(rate_1h_f_pix_err)))
            print("")
            print(timing_fit)
            last_report = this_time

        """

        # FIXME wanna save?
        """
        if not filename and this_time - last_save > 60*save_min:
            print("\n\nSaving\n")
            plt.savefig(t._filename(last_save_file + ".eps"), dpi=200.0, format='eps')
            last_save = this_time
        """ 

    except ValueError: #KeyboardInterrupt:
        try:
            """
            if cluster_analysis:
                rate = 0; rate_pix = 0
                rate_f = 0; rate_f_pix = 0

                for cluster in cluster_history:
                    rate += 1
                    rate_pix += len(cluster.pixels)

                    if not cluster.blob:
                        rate_f += 1
                        rate_f_pix += len(cluster.pixels)

                area_cm2 = (25e-6*25e-6*512*512)/(0.01*0.01)

                time_factor = elapsed
                rate = rate/time_factor/area_cm2
                rate_pix = rate_pix/time_factor/area_cm2
                rate_f = rate_f/time_factor/area_cm2
                rate_f_pix = rate_f_pix/time_factor/area_cm2

                print("Report @ %.3f s" % (this_time - start_time))
                print(" w/ big clusters. Particle rate: %sHz/cm2 - Pixel rate: %sHz/cm2" % (autoscale(rate), autoscale(rate_pix)))
                print("w/o big clusters. Particle rate: %sHz/cm2 - Pixel rate: %sHz/cm2" % (autoscale(rate_f), autoscale(rate_f_pix)))
                print(timing_fit)

            not filename and t.savecsv(last_save_file)

            """
            time.sleep(1)
        except KeyboardInterrupt:
            break



    # Readout of chip 1. Copied only not commented stuff.
    try:
        new_pixels = []
        new_pixels_times = {}

        x.chip[1].injection_digital()
        x.chip[1].injection_analog()
        overflow1 = x.chip[1].fifo_overflow_count()
        if overflow1 > 0:
            print(f"Warning: lost {overflow1} packets in chip 1!")
            x.chip[1].fifo_overflow_counter_reset()

        current = SubSequence( x.chip[1].readout(), init_ts_sw=last_ts_sw1, init_triggered = last_triggered1, sw_delta_ts = x.ts_delta_sw[1] )


        #newData1 = []
        # Save new data acquired in this readout.
        newData1 += current

        # Update already triggered.
        last_triggered1 = current.timesTriggered
        last_ts_sw1 = current.ts_sw
        this_time = time.time()

        x.packets = current

        if len(current) > 0:
            last_save_file1 = x.savecsv(last_save_file1, detectorNo = 1)


        if len(current) > 0:
            current_pixels = []
            already_hit = {}
            for packet in current:
                # Do not analyze trigger start words.
                if isinstance(packet, TriggerStartWord): continue

                # Also skip testpulse words (done for debug purposes).
                if isinstance(packet, TestPulse): continue

                """
                if past_ts_extq is not None:
                    times.append(packet.ts_ext - past_ts_extq)
                past_ts_extq = packet.ts_ext
                """
                pixels = packet.get_pixels()
                

                for pixel in pixels:
                    pixel.ts = packet.ts_ext
                    # Update data plots.
                    res_incr_1[pixel.row][pixel.col] += 1
                    res_last_1[pixel.row][pixel.col] = fading_time_us

                    current_pixels.append(pixel)
                    new_pixels.append( (pixel.row, pixel.col) )
                    new_pixels_times[(pixel.row, pixel.col)] = packet.ts_ext
                        
                          
            last_pixels = current_pixels
        
        elapsed_step = this_time - past_time 
        past_time = this_time
        elapsed = this_time - start_time

        # Update res_last
        res_last_1[res_last_1 > 0] -= int(elapsed_step*1E6)

        # Update view! (useless?)
        visualize(new_clusters, clusters_removed, elapsed_step)

        # Eventually mask
        if setAutomask and this_time - last_mask1 > mask_time_s:
            automask(1)
            last_mask1 = this_time
         

        



    except ValueError: #KeyboardInterrupt:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break

    # Update sync plot.
    if len(newData0) > 0 and len(newData1) > 0:
        #print("Two data in same readout")
    
        # Sorting should not be needed, as I expect data will be ordered.
        #FIXME check.



        # Look while ids are still within the data.
        while id0 < len(newData0) and id1 < len(newData1):
            #print(f"DBG: id0 is {id0} and id1 is {id1}")
            #print(f"DBG: len(newData0) is {len(newData0)} and len(newData1) is {len(newData1)}")
            #print(f"id:{id0}/{len(newData0)} and {id1}/{len(newData1)}")
            #print(f"Timestamps:{newData0[id0].ts_ext} and {newData1[id1].ts_ext}")
            #print("~~~~~~~~")
            #for data in newData0: print(data.ts_ext)
            #print("~~~~~~~~")
            #for data in newData1: print(data.ts_ext)

            # If the distance between the two ids is minor than sync.
            if abs(newData0[id0].ts_ext - newData1[id1].ts_ext) < sync:

                print(f"found sync @ {id0} ; {id1} with diff {newData0[id0].ts_ext - newData1[id1].ts_ext}")
                
                # Initialize current packets for saving.
                current0 = [newData0[id0]]
                current1 = [newData1[id1]]

                runningSync += 1

                # Look for the earliest one.
                earliest_one = newData0[id0] if newData0[id0].ts_ext < newData1[id1].ts_ext else newData1[id1]
 
                #update the plots.
                for pixel in newData0[id0].get_pixels():
                    res_sync_0[pixel.row][pixel.col] += runningSync

                for pixel in newData1[id1].get_pixels():
                    res_sync_1[pixel.row][pixel.col] += runningSync

                # Look for other data in this window.
                while id0 + 1 < len(newData0):
                    if newData0[id0 + 1].ts_ext < earliest_one.ts_ext + sync:
                        id0 += 1
                        current0.append(newData0[id0])
                        for pixel in newData0[id0].get_pixels():
                            res_sync_0[pixel.row][pixel.col] += runningSync
                    else: break

                # Look for other data in this window.
                while id1 + 1 < len(newData1):
                    if newData1[id1 + 1].ts_ext < earliest_one.ts_ext + sync:
                        id1 += 1
                        current1.append(newData1[id1])
                        for pixel in newData1[id1].get_pixels():
                            res_sync_1[pixel.row][pixel.col] += runningSync
                    else: break

                # Save on file
                if len(current0) > 0:
                  x.packets = current0
                  last_save_filesync = x.savecsv(last_save_filesync, detectorNo = 0)
                  x.packets = current1
                  last_save_filesync = x.savecsv(last_save_filesync, detectorNo = 1)


                # They are sync, I want to update the indexes.
                if id0 < len(newData0) or id1 < len(newData1):
                  if id0 < len(newData0): id0 += 1
                  if id1 < len(newData1): id1 += 1
                  continue

            # Update the IDs.
            sumID0 = 0
            sumID1 = 0
            #print(f"DBG: newData0[id0].ts_ext is {newData0[id0].ts_ext}; newData1[id1].ts_ext is {newData1[id1].ts_ext}")
            if newData0[id0].ts_ext <= newData1[id1].ts_ext:
                while id0 + sumID0 < len(newData0) and newData0[id0 + sumID0].ts_ext < newData1[id1].ts_ext:
                    sumID0 += 1 
            else:
                while id1 + sumID1 < len(newData1) and newData1[id1 + sumID1].ts_ext < newData0[id0].ts_ext:
                    sumID1 += 1

            id0 += sumID0
            id1 += sumID1

            
            # If they are not synced, manually sync sw timestamp as we lost some packets.
            if id0 < len(newData0) and id1 < len(newData1):
              if newData0[id0].ts_ext - newData1[id1].ts_ext > 1 << 24:
                last_ts_sw1 += 1
                print("Updating SW timestamp 1")
              elif newData1[id1].ts_ext - newData0[id0].ts_ext > 1 << 24:
                last_ts_sw0 += 1
                print("Updating SW timestamp 0")
            


