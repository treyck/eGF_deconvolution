import numpy as np
import matplotlib.pyplot as plt  # Use for plotting if you plan to plot inside this file
import scipy as sp
from scipy import signal
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import simps
from obspy import read, read_inventory
from obspy.core import UTCDateTime
from obspy.signal.invsim import cosine_taper


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
    id = code.split('.')[-2]

    path = '{}/{}.phase'.format(event,id)
    ppick,spick,event_start,dist,one,two = retrieve_phases(path,network,station)
    response_path = '{}/{}.iris.xml'.format(event,id)  
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


def common_station_bookwork(ev1,ev2):
    '''
    Get the common stations and return the ordered s amplitudes for these stations using event indices ev1
    and ev2
    '''
    # Make the pairs only use data from overlapping stations:

    # Create a set of new amplitudes

    # Find the common stations to both
    common_list = [c for c in sta_list[ev1] if c in sta_list[ev2]]

    # find the indices for each of the station list which contain the common list
    ind1 = list()
    ind2 = list()
    for j in range(len(common_list)):
        ind1.append(np.where(np.asarray(sta_list[ev1])==common_list[j])[0][0])
        ind2.append(np.where(np.asarray(sta_list[ev2])==common_list[j])[0][0])
    # Adjust the lists according to only the common stations
    sE_amps_ev1 = [sE_amps[ev1][index] for index in ind1]
    sN_amps_ev1 = [sN_amps[ev1][index] for index in ind1]
    sE_amps_ev2 = [sE_amps[ev2][index] for index in ind2]
    sN_amps_ev2 = [sN_amps[ev2][index] for index in ind2]
    noiseE_amps_ev1 = [noiseE_amps[ev1][index] for index in ind1]
    noiseN_amps_ev1 = [noiseN_amps[ev1][index] for index in ind1]
    noiseE_amps_ev2 = [noiseE_amps[ev2][index] for index in ind2]
    noiseN_amps_ev2 = [noiseN_amps[ev2][index] for index in ind2]
    sta_list_ev1 = [sta_list[ev1][index] for index in ind1]
#     net_list_ev1 = [net_list[ev1][index] for index in ind1]
#     net_list_ev2 = [net_list[ev2][index] for index in ind2]

    
    return sE_amps_ev1,sN_amps_ev1,sE_amps_ev2,sN_amps_ev2,noiseE_amps_ev1,noiseN_amps_ev1,noiseE_amps_ev2,noiseN_amps_ev2,sta_list_ev1

def common_station_distances(ev1,ev2):
    '''
    Get the common stations and return the ordered s amplitudes for these stations using event indices ev1
    and ev2
    '''
    # Make the pairs only use data from overlapping stations:

    # Create a set of new amplitudes

    # Find the common stations to both
    common_list = [c for c in sta_list[ev1] if c in sta_list[ev2]]

    # find the indices for each of the station list which contain the common list
    ind1 = list()
    ind2 = list()
    for j in range(len(common_list)):
        ind1.append(np.where(np.asarray(sta_list[ev1])==common_list[j])[0][0])
        ind2.append(np.where(np.asarray(sta_list[ev2])==common_list[j])[0][0])
        
    distances_ev1 = [distances[ev1][index] for index in ind1]
    
    return distances_ev1

def spectral_ratio(freq, moment_R, fc1, fc2):
    numerator = moment_R*(1+(freq/fc2)**2)
    denominator = (1+(freq/fc1)**2)
    return numerator/denominator

def brune_model(freq,moment_R,fc1):
    numerator = moment_R
    denominator = (1+(freq/fc1)**2)
    return numerator/denominator
