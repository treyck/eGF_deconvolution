import numpy as np
from obspy import read, read_inventory
import obspy
from scipy import signal
from scipy import stats
from scipy import sparse
from scipy import linalg
from scipy.optimize import curve_fit
from obspy.signal.invsim import (cosine_taper, cosine_sac_taper,
                                         invert_spectrum)

from scipy.integrate import simps

from obspy.core import UTCDateTime

import traceback
import os
import glob
import pickle
import sys

# import modules from the attenuation calculations

import scipy as sp
from scipy import sparse
import copy
import math

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


from scipy.signal import butter,sosfilt, sosfiltfilt, sosfreqz
from scipy.interpolate import interp1d

from scipy.integrate import simps


def spectrum(data, win, nfft, n1=0, n2=0,ver='real'):
    """
    Spectrum of a signal.

    Computes the spectrum of the given data which can be windowed or not. The
    spectrum is estimated using the modified periodogram. If n1 and n2 are not
    specified the periodogram of the entire sequence is returned.

    The modified periodogram of the given signal is returned.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to make spectrum of.
    :param win: Window to multiply with given signal.
    :param nfft: Number of points for FFT.
    :type n1: int, optional
    :param n1: Starting index, defaults to ``0``.
    :type n2: int, optional
    :param n2: Ending index, defaults to ``0``.
    :return: Spectrum.
    """
    if (n2 == 0):
        n2 = len(data)
    n = n2 - n1
    u = pow(np.linalg.norm([win]), 2) / (n)
    xw = data * win
    
    if ver=='real':
        
        fftval = sp.fftpack.rfft(xw, nfft)
        px = pow(abs(fftval), 2) / (n * u)
        px[0] = px[1]

        freq = sp.fftpack.rfftfreq(nfft,d=1/100)
        return px,freq
    elif ver=='standard':
        fftval = sp.fftpack.fft(xw, nfft)
        px = pow(abs(fftval), 2) / (n * u)
        px[0] = px[1]
        
#         print('Standard returns:', fftval[0:10])
        freq = sp.fftpack.fftfreq(nfft,d=1/100)
        return px[:int(len(px)/2)],freq[:int(len(px)/2)]
    

def running_average(x_values,data,n_elements=5):

    # Define the window for running average
    window = np.ones(n_elements) / n_elements  # Running average over 3 elements

    # Calculate running average using convolution
    running_avg = sp.signal.convolve(data, window, mode='valid')
    
    new_x_values = x_values[(n_elements - 1) // 2 : -(n_elements - 1) // 2]
    
    return new_x_values,running_avg

def window_time_series(inc, tr, time_variables, preS = 0.1, postS = 3,Verbose=False):
    data = tr.data
    
    ppick = time_variables[0]
    spick = time_variables[1]
    event_start = time_variables[2]

    record_start = tr.stats.starttime
    sample_rate = tr.stats.sampling_rate

    difference_sec = event_start - record_start
    
    if Verbose:
        print(inc)

    start = np.where(tr.times() >= (spick+difference_sec-preS+inc))[0][0]; 
    end = np.where(tr.times() >= (postS+spick+difference_sec-preS+inc))[0][0];
    # start = 0; end=50;
    
    if Verbose:
        print(start,end)

    timesn = tr.times()[start:end]; datan = data[start:end]
    
    return timesn, datan

def window_noise(tr, time_variables, preS = 0.1, postS = 3,Verbose=False):
    data = tr.data
    
    ppick = time_variables[0]
    spick = time_variables[1]
    event_start = time_variables[2]

    record_start = tr.stats.starttime
    sample_rate = tr.stats.sampling_rate

    difference_sec = event_start - record_start

    ## Logic here: If there is a p-pick, use that as the end of noise. Otherwise conservatively use the S
    if ppick != 0:
        try:
            start = np.where(tr.times() >= (ppick+difference_sec-preS - postS))[0][0]; 
            end = np.where(tr.times() >= (ppick+difference_sec-preS))[0][0];
        except:
            start = 0
            end = sample_rate*postS
    else:
        try:
            start = np.where(tr.times() >= (spick+difference_sec-preS - postS))[0][0]; 
            end = np.where(tr.times() >= (spick+difference_sec-preS))[0][0];
        except:
            start = 0
            end = sample_rate*postS
     
    
    times_noise = tr.times()[start:end]; data_noise = data[start:end]
    
    return times_noise, data_noise
def multiwindow_spectra(tr,time_variable,length_window,num_windows=5):
    fEn_multiwindow = 0

    ratiosNet = 0
    divisor = 0
    increment = length_window/2
    SENet = np.array([])
    for i in range(num_windows):

        timesn,datan = window_time_series(i*increment,tr,time_variable,postS = length_window)

        SE,fE = spectrum(datan,cosine_taper(len(datan),0.05,sactaper=True, halfcosine=False),len(datan),
                     ver='standard')
        
        fE[0] = 0.01

        fEn, SEn = interpolate_data(fE,SE,0.025)
        fEn, SEn = running_average(fEn, SEn)
        fEn_multiwindow, SEn_multiwindow = raise_to_power(fEn,SEn,10)

    #     plt.semilogx(fEn1,SEn1)

        SENet = np.concatenate((SENet,SEn_multiwindow))
    
    times_noise,noise = window_noise(tr,time_variable,postS = length_window)
    
    SE_noise,fE_noise = spectrum(noise,cosine_taper(len(noise),0.05,sactaper=True, halfcosine=False),len(noise),
                     ver='standard')
    
    fE_noise[0] = 0.01
    
    fE_noise, SE_noise = interpolate_data(fE_noise,SE_noise,0.025)
    fE_noise, SE_noise = running_average(fE_noise, SE_noise)
    fE_noise, SE_noise = raise_to_power(fE_noise,SE_noise,10)
    
    return fEn_multiwindow,SENet,SE_noise



def interpolate_data(fE,SE, spacing=0.025):
    interp_func = interp1d(np.log10(fE), np.log10(SE), kind='linear')  # linear interpolation
    
    f_desired = np.arange(np.log10(0.025),np.log10(49),spacing)
    
    interpolated_values = interp_func(f_desired)
    
    
    return f_desired, interpolated_values

def raise_to_power(fE,SE,power):
    
    new_frequencies = np.power(power,fE)
    new_SE = np.power(power,SE)
    
    return new_frequencies, new_SE

def preprocess_trace(file,plot=False):
    
    st = read(file)


    event = file.split('/')[-2]
    code = file.split('/')[-1]
    network = code.split('.')[1]
    station = code.split('.')[0]
    comp = code.split('.')[2]
    folder = file.split('/')[-3]

    path = '/oak/stanford/schools/ees/beroza/tknudson/ridgecrest/{}/{}/{}.phase'.format(folder,event,event)
    ppick,spick,event_start,dist,one,two = retrieve_phases(path,network,station)
    response_path = '/oak/stanford/schools/ees/beroza/tknudson/ridgecrest/{}/{}/{}.iris.xml'.format(folder,event,event)  
    inv = read_inventory(response_path)
    #print(ppick,spick,event_start,dist)
    tr = st[0]


    record_start = tr.stats.starttime
    sample_rate = tr.stats.sampling_rate

#     tr = st[0].copy()
    data = tr.data
    times = tr.times()
    data = (data - np.mean(data))
    npts = len(data)
    data = (data * cosine_taper(npts,0.05,sactaper=True, halfcosine=False))


    tr.data = data
    
    if plot:
        plt.figure()
        difference_sec = event_start - record_start
        plt.plot(times,data)
        plt.axvline(difference_sec+spick,c='r')
        plt.axvline(difference_sec+spick+10,c='r')
    
    time_variables = (ppick,spick,event_start)
    
    return tr, time_variables

def retrieve_phases(path, network, station):
    ppick = 0
    spick = 0
    distance = 0
    lat = ''
    lon = ''

    with open(path, 'r') as f:

        lines = f.readlines()
        event_start = lines[0].split()[3]
        for i in range(1,len(lines)-1):
            lines_break = lines[i].split()

            if lines_break[0] == network and lines_break[1] == station:
                if lines_break[7] == "P" and ppick == 0:
                    ppick = float(lines_break[12])
                elif lines_break[7] == "S" and spick == 0:
                    spick = float(lines_break[12])
                    
                if distance == 0:
                    distance = float(lines_break[11])
                    
                if lat == '' and lon=='':
                    lat = float(lines_break[4])
                    lon = float(lines_break[5])

    #print(event_start)
    #print(ppick, spick)
    event_start = UTCDateTime(event_start)
    
    return ppick, spick, event_start, distance, lat, lon

def retrieve_coords_ev(path_ev):
    with open(path_ev,'r') as f:
        lines = f.readlines()
        lat = float(lines[0].split()[3])
        lon = float(lines[0].split()[6])
        
    return lat, lon

def retrieve_coords_ev_phase(path_ev):
    '''
    This is assuming .phase as the input file
    '''
    with open(path_ev,'r') as f:
        lines = f.readlines()
        lat = float(lines[0].split()[4])
        lon = float(lines[0].split()[5])
        
    return lat, lon

# def testing(path )


################################# Begin the function ####################################

loc = '/oak/stanford/schools/ees/beroza/tknudson/ridgecrest/' 

directory = sys.argv[1]

start_chunk = int(sys.argv[2])
start_index = start_chunk*80
end_index = (start_chunk+1)*80

path = loc+directory+'/*'
 
directories = glob.glob(path)

if end_index >= len(directories):
     end_index = len(directories)

ids = list()

for i in range(len(directories)):
    ids.append(directories[i].split('/')[-1])
    
# assign directory

print("Number of events:", len(directories))
indice_list = list()

s_ampsE = list()
s_ampsN = list()
noise_ampsE = list()
noise_ampsN = list()
distances = list()
frequency = list()
net_list = list()
sta_list = list()
sta_lat = list()
sta_lon = list()

attempts = 0
good_attempts = 0

# for i in range(len(directories)):
for i in range(start_index,end_index):
    plotcount = 0
    print(i,'/55')
    print('id:',ids[i])
    s_ampsE_e = list()
    s_ampsN_e = list()
    noise_ampsE_e = list()
    noise_ampsN_e = list()
    distances_e = list()
    net_list_e = list()
    sta_list_e = list()
    sta_lat_e = list()
    sta_lon_e = list()
    netcodes = list()
    
    
    response_path = glob.glob(directories[i]+'/*.iris.xml')[0]
    print(response_path)
    phase_path = glob.glob(directories[i]+'/*.phase')[0]
    print(phase_path)
    inv = read_inventory(response_path)
    seeds = glob.glob(directories[i]+'/*.ms')

    
#    event_loc_path = directories[i]+'/taup.out'
#    ev_la,ev_lo = retrieve_coords_ev(event_loc_path)
    for j in range(len(seeds)):  
#     for j in range(10):
        attempts += 1
        strings = seeds[j].split('.')
        if strings[2][1] == 'N':
            continue
        network1 = strings[1]
        if network1 == "GS":
            continue
        station1 = strings[0].split('/')[-1]
        channel1 = strings[2]
        if 'Z' in channel1:
            continue
        code1 = network1+'.'+station1+'.'+channel1
        netcode = network1+'.'+station1
        if netcode not in netcodes and ((channel1 == "HHE" or channel1 == "HHN") or
                                        (channel1 == "HH1" or channel1 == "HH2") or
                                        (channel1 == "EH1" or channel1 == "EH2") or
                                        (channel1 == "EHN" or channel1 == "EHE")):
            netcodes.append(netcode)
            for k in range(len(seeds)):
                strings = seeds[k].split('.')
                if strings[2][1] == 'N':
                    continue
                network2 = strings[1]
                if network2 == "GS":
                    continue
                station2 = strings[0].split('/')[-1]
                channel2 = strings[2]
                if 'Z' in channel2:
                    continue
                code2 = network2+'.'+station2+'.'+channel2
                netcode2 = network2+'.'+station2
                
                if netcode2 == netcode and channel2!=channel1 and ((channel1 == "HHE" or channel1 == "HHN") or
                                                                   (channel1 == "HH1" or channel1 == "HH2") or
                                                                   (channel1 == "EH1" or channel1 == "EH2") or
                                                                   (channel1 == "EHN" or channel1 == "EHE")):
                    try:
                        ppick,spick,event_start,distance_,lat_sta,lon_sta = retrieve_phases(phase_path,network1,station1)
                        
                        if channel1[-1] == 'E' or channel1[-1] == '1':
                            tr1,time_variables1 = preprocess_trace(seeds[j],False)
                            tr2,time_variables2 = preprocess_trace(seeds[k],False)
                        else:
                            tr1,time_variables1 = preprocess_trace(seeds[k],False)
                            tr2,time_variables2 = preprocess_trace(seeds[j],False)
                            
                        # Break if the S-pick for either case is not good:
                        if time_variables1[1] == 0 or time_variables2[1] == 0:
                            #print('No S picks')
                            break

        
                        freq1,spectra1,noise1 = multiwindow_spectra(tr1,time_variables1,length_window=4)
                        freq2,spectra2,noise2 = multiwindow_spectra(tr2,time_variables2,length_window=4)
                        # Assign a placeholder p amplitude for now
                    
                        if np.any(spectra1 != 0) and np.any(spectra2 != 0):

                            s_ampsE_e.append(spectra1)
                            s_ampsN_e.append(spectra2)
                            noise_ampsE_e.append(noise1)
                            noise_ampsN_e.append(noise2)
                            distances_e.append(distance_)
                            frequency = freq1
                            net_list_e.append(network1)
                            sta_list_e.append(station1)
                            sta_lat_e.append(lat_sta)
                            sta_lon_e.append(lon_sta)

                            
                        else: # if there aren't valid values, we just forget these components
                            break


                        good_attempts += 1
                        break

                    except Exception as e:
                        print(e)
                        continue
                else: # If the kth station isn't the respective N or E component
                    continue
        else: # if the station isn't the right channel or if it's already in netcodes
            continue
    
    print("Netcodes :", netcodes)
    s_ampsE.append(s_ampsE_e)
    s_ampsN.append(s_ampsN_e)
    noise_ampsE.append(noise_ampsE_e)
    noise_ampsN.append(noise_ampsN_e)
    distances.append(distances_e)
    net_list.append(net_list_e)
    sta_list.append(sta_list_e)
    sta_lat.append(sta_lat_e)
    sta_lon.append(sta_lon_e)
    
            
print(good_attempts,'/',attempts,' successful attempts')



########################################## Save the Data ################################################
loc = '/oak/stanford/schools/ees/beroza/tknudson/spectral_ratios/data_chunks_full/'

with open(loc+'s_ampsE'+directory+'_'+sys.argv[2]+'.data', 'wb') as f:
    pickle.dump(s_ampsE, f)
with open(loc+'s_ampsN'+directory+'_'+sys.argv[2]+'.data', 'wb') as f:
    pickle.dump(s_ampsN, f)
with open(loc+'noise_ampsE'+directory+'_'+sys.argv[2]+'.data', 'wb') as f:
    pickle.dump(noise_ampsE, f)
with open(loc+'noise_ampsN'+directory+'_'+sys.argv[2]+'.data', 'wb') as f:
    pickle.dump(noise_ampsN, f)
with open(loc+'distances'+directory+'_'+sys.argv[2]+'.data', 'wb') as f:
    pickle.dump(distances, f)
with open(loc+'net_list'+directory+'_'+sys.argv[2]+'.data', 'wb') as f:
    pickle.dump(net_list, f)
with open(loc+'sta_list'+directory+'_'+sys.argv[2]+'.data', 'wb') as f:
    pickle.dump(sta_list, f)
with open(loc+'sta_latitudes'+directory+'_'+sys.argv[2]+'.data', 'wb') as f:
    pickle.dump(sta_lat, f)
with open(loc+'sta_longitudes'+directory+'_'+sys.argv[2]+'.data', 'wb') as f:
    pickle.dump(sta_lon, f)
with open(loc+'frequency'+directory+'_'+sys.argv[2]+'.data', 'wb') as f:
    pickle.dump(frequency, f)
with open(loc+'ids'+directory+'_'+sys.argv[2]+'.data', 'wb') as f:
    pickle.dump(ids, f)   

