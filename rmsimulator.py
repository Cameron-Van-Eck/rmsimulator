
import sys
from RMutils.util_RM import do_rmsynth_planes,get_rmsf_planes
import numpy as np

class RMspectrum:
    """Class for an RM spectrum (also called a Faraday depth function. This is an object that has both a frequency-space spectrum and a Faraday-depth space spectrum.
    To initialize the RMspectrum, it's necessary to specify the sampling in Faraday depth (the Faraday depth range to be covered and the step size (all in units of rad/m^2)), and the frequency sampling (the channel frequencies of the associated observation, in Hz). The frequency sampling can either be specified from a string (see the set_sampling method) for one of the pre-programmed observation configurations or as an array of channel frequencies (in Hz).
    
    Once a sampling is initialized, it's possible to add Faraday-depth space features using the add_to_spectrum method. Since everything is linear, you can add as many features as you like (invoking the method adds another feature; there is no way of removing a feature except to re-initialize the RMspectrum). To add a feature, the type of feature must be specified (as one of the pre-defined strings), and a list of parameters (which is different for each type of feature) must be supplied. See the method docstring for details about the different options.
    
    Once the features are added, it is required to run the calculate_spectrum method. This performs the actual RM synthesis and populates the relevant class properties.
    
    Class properties:
    phi: array of Faraday depth positions for the RM spectrum.
    spectrum: array of complex RM spectrum values (at the positions specified by phi)
    idealspectrum: the physically 'true' shape of the RM spectrum, for the features that have been added. (at the positions specified by phi). See note below about normalization.
    freq: frequency array for all channels
    sampling: lambda^2 array for all channels
    P_lambda: array of 'measured' (complex) polarization as a function of frequency/lambda^2 for all channels
    Q_lambda: the real part of P_lambda
    U_lambda: the imaginary part of P_lambda
    RMSF_Phi: array of Faraday depth positions for the RMSF
    RMSF: array of complex RMSF values (at the positions specified by RMSF_Phi)
    RMSF_width: width the RMSF
    
    Normalization note:
    There is a problem of normalization between the RM spectrum and the ideal spectrum, because of units. The RM spectrum is always calculated in units of intensity per RMSF; this has the effect that a Faraday-thin ('peak') feature is observed to have the correct intensity. But this means that Faraday-complex features have a measured intensity that isn't directly mappable to true intensity. The ideal spectrum has nominal units of intensity per (rad/m^2). In principle, it should be possible to approximately normalize by the RMSF width, but this has not been tested and is likely to have problems.
    """

    def add_to_spectrum(self,mode,parms):
        """This method will add a feature to an RM spectrum (more accurately, to P(lambda)). Also adds feature to ideal spectrum.
        Inputs:
        mode: a string with the type of spectrum desired. Must be one of the programmed types.
        parms: array/list of parameters that define the spectrum. Number of and meanings of elements
                will vary depending on mode.
        spectrum (self): The input spectrum, to which the new feature will be added. Of RMSpectrum type.
                The spectrum must have lambda^2 values in self.sampling.
        Modes:
            peak: [phi,amplitude,angle(deg)]  (this is a perfectly Faraday-thin peak)
            slab: [phi_min,phi_max,amplitude,angle(deg)] (classic Burn slab)
            gaussian: [amplitude, angle(deg), phi_0,sigma]  (Gaussians are associated with foreground turbulent Faraday screens)
            noise: [Amplitude] (pure Gaussian noise of specified amplitude in Q and U; ideal spectrum counterpart should also be noise of amplitude 1/N_channels)/
        """

        if mode == 'peak': #spectrum with 1 peak in it.
            #parms=[phi,amplitude,angle(degrees)]
            self.P_lambda+=parms[1]*np.exp(2j*parms[0]*self.sampling+np.radians(parms[2])*2j)
            w=np.argmin(np.abs(self.phi-parms[0]))
            self.idealspectrum[w]+=parms[1]*np.exp(2j*np.radians(parms[2]))
        elif mode == 'slab': #Uniform Faraday slab (tophat function)
            #parms=[phi_min,phi_max,amplitude,angle(degrees)]
            self.P_lambda+=parms[2]*np.exp(2j*np.radians(parms[3]))*(np.exp(2j*parms[1]*self.sampling)-np.exp(2j*parms[0]*self.sampling))/(2j*self.sampling)/(parms[1]-parms[0])
            w=np.where((self.phi >= parms[0]) & (self.phi <= parms[1]))
            self.idealspectrum[w]+=parms[2]*np.exp(2j*np.radians(parms[3]))/(parms[1]-parms[0])
        elif mode == 'gaussian': #Gaussian Faraday 'slab'
            #parms=[Amplitude, angle(degrees),phi_0 (center of Gaussian), sigma (width of Gaussian)]
            self.P_lambda+=parms[0]*np.exp(2j*np.radians(parms[1]))*np.exp(
                2j*parms[2]*self.sampling)*np.exp(-1*parms[3]**2*self.sampling**2)
            self.idealspectrum+=parms[0]*np.sqrt(1/np.pi)/parms[3]*np.exp(2j*np.radians(parms[1]))*np.exp(-1*(self.phi-parms[2])**2/(parms[3]**2))
        elif mode == 'noise':
            #parms=[Amplitude]
            self.P_lambda+=parms[0]*(np.random.randn(self.sampling.size)+1.j*np.random.randn(self.sampling.size))
        else:
            print("Not valid spectrum feature!")
            raise



    def set_sampling(self,sampling):
        """Defines a frequency/lambda^2 sampling, based on an input array or string.
        If numpy array supplied, will be used as channel frequencies (in Hz).
        If string supplied, will be checked against pre-defined defaults.
        Inputs: sampling: array or string with the name of the sampling desired.
        Outputs: l2_array: An array of channel wavelength squares.
                 freq_array: Array of channel frequencies.
        Currently coded sampling string options: 
             'equal': log-spaced 1e4--1e9 Hz
             'LOFAR': IC342-like 
             'LOTSS': LOTSS-like
             'LOFAR_alt: Jennifer's data?
             'VLA': 1-2 GHz with 1000 channels
             'fixed': equal spacing in lambda^2, 1e-12--8 m^2
             'VLASS_coarse': frequencies from early VLASS coarse cube"""

        c=299792458  #speed of light in m/s
        if type(sampling)==np.ndarray:
            freq_array=sampling.copy()
            l2_array=(c/freq_array)**2
        elif sampling == 'equal':
#            freq_array=np.concatenate((np.arange(1e5,1e7,1e5),np.arange(1e7,1e9,1e6),np.arange(1e9,1e11,1e9)))
            freq_array=np.logspace(4,9,num=10000)
            l2_array=(c/freq_array)**2
        elif sampling == 'LOFAR': #more specifically, my IC342 data
            num_SB=324   #number of SubBands observed
            chan_per_SB=8   #channels in a SB
            SB_bandwidth=195312.   #Subband bandwidth (in Hz)
            SB0_frequency=114.952E6  #Lowest frequency (in Hz)
            freq_array=np.asarray(range(num_SB*chan_per_SB))*SB_bandwidth/chan_per_SB+SB0_frequency
            l2_array=(c/freq_array)**2
        elif sampling == 'LOTSS': #frequency coverage of typical LOTSS data.
            num_SB=244   #number of SubBands observed
            chan_per_SB=2   #channels in a SB
            SB_bandwidth=195312.   #Subband bandwidth (in Hz)
            SB0_frequency=120.0E6  #Lowest frequency (in Hz)
            freq_array=np.asarray(range(num_SB*chan_per_SB))*SB_bandwidth/chan_per_SB+SB0_frequency
            l2_array=(c/freq_array)**2
        elif sampling == 'LOFAR_alt': #experimenting with high averaging as demonstration for Jennifer
            num_SB=20   #number of SubBands observed
            chan_per_SB=1   #channels in a SB
            SB_bandwidth=2e6   #Subband bandwidth (in Hz)
            SB0_frequency=120.952E6  #Lowest frequency (in Hz)
            freq_array=np.asarray(range(num_SB*chan_per_SB))*SB_bandwidth/chan_per_SB+SB0_frequency
            l2_array=(c/freq_array)**2
        elif sampling == 'VLA': #1-2 GHz. For talk figure.
            num_chan=1000
            freq_array=1.e9+1.e9/num_chan*np.array(range(num_chan))
            l2_array=(c/freq_array)**2
        elif sampling=='fixed': #equally space in lambda^2
            l2_array=np.linspace(1e-12,20.0,10000)
            freq_array=c/np.sqrt(l2_array)
        elif sampling=='VLASS_coarse':
            freq_array=np.array([2027866895.13, 2155858491.06, 2283850087.0, 2411841682.94, 2539833278.88, 2667824874.81, 2795816470.75, 2923808066.68, 3051799662.62, 3179791258.56, 3307782854.5, 3435774450.43, 3563766046.37, 3691757642.31, 3819749238.24, 3947740834.18])
            l2_array=(c/freq_array)**2
        else:
            print("Not valid sampling!")
            raise
        return l2_array,freq_array

    def calculate_spectrum(self):
        """
        Runs RM synthesis. Takes no arguments. Run after adding a feature to the (theoretical) spectrum.
        """
        spectrum, lambda2_0=do_rmsynth_planes(np.real(self.P_lambda),np.imag(self.P_lambda), self.sampling, self.phi, 
                      weightArr=None, lam0Sq_m2=None, nBits=32, verbose=False)
        self.spectrum=spectrum
        self.Q_spectrum=self.spectrum.real
        self.U_spectrum=self.spectrum.imag
        self.l2_0=lambda2_0

    def __init__(self,phi_min,phi_max,dphi,sampling):
        """RMspectrum(phi_min,phi_max,dphi)
        Creates a blank RM spectrum, which runs from phi_min to phi_max in
        steps of dphi. Frequency sampling is set by entering a valid string in the sampling variable."""
        self.dphi=dphi  #Faraday depth sampling spacing.
        self.phi=np.arange(phi_min,phi_max,dphi)  #array of Faraday depths sampled
        self.spectrum=np.zeros(self.phi.shape,dtype=np.complex)  #Faraday spectrum, initially empty
        self.idealspectrum=np.zeros(self.phi.shape,dtype=np.complex)
        self.sampling,self.freq=self.set_sampling(sampling)  #lambda^2 sampling and frequency sampling
        self.P_lambda=np.zeros(self.sampling.shape,dtype=np.complex)  #Array of 'measured' polarizations as function of lambda^2.
        self.Q_lambda=self.P_lambda.real
        self.U_lambda=self.P_lambda.imag

        self.l2_0=np.mean(self.sampling) #Mean value of lambda^2, used in pyRMsynth to reduce phase/polangle wrapping.
        RMSFcube, phi2Arr, fwhmRMSFArr, statArr=get_rmsf_planes(self.sampling, self.phi, weightArr=None, mskArr=None, 
                    lam0Sq_m2=None, double=True, fitRMSF=True,
                    fitRMSFreal=False, nBits=32, verbose=False)
        self.RMSF=RMSFcube
        self.RMSF_Phi=phi2Arr
        self.RMSF_width=fwhmRMSFArr
        
    
    




