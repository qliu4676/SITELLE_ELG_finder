import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy.optimize import curve_fit
from scipy import interpolate, signal, integrate, stats
from scipy.spatial.distance import pdist

import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic

from astropy.stats import SigmaClip, sigma_clip, mad_std, gaussian_fwhm_to_sigma
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel, MexicanHat2DKernel
from astropy.modeling import models, fitting

import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, ICRS, FK5
from astropy.visualization import make_lupton_rgb, AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
norm = ImageNormalize(stretch=AsinhStretch())

from photutils import Background2D, SExtractorBackground
from photutils import EllipticalAnnulus, EllipticalAperture, aperture_photometry

from photutils import centroid_com, centroid_2dg
from photutils import detect_sources, detect_threshold, deblend_sources, source_properties


def vmax_3sig(img):
    # upper limit of visual imshow defined by 3 sigma above median
    return np.median(img)+3*np.std(img)

def vmax_5sig(img):
    # upper limit of visual imshow defined by 5 sigma above median
    return np.median(img)+5*np.std(img)

def vmin_5sig(img):
    # lower limit of visual imshow defined by 5 sigma below median
    return np.median(img)-5*np.std(img)

def printsomething():
    print("Print")
    
def coord_ImtoArray(X_IMAGE, Y_IMAGE):
    # Convert image coordniate to numpy array coordinate
    x_arr, y_arr = Y_IMAGE-1, X_IMAGE-1
    return x_arr, y_arr

def background_sub_SE(field, mask=None, b_size=64, f_size=3, n_iter=5):
    # Subtract background using SE estimator with mask
    Bkg = Background2D(field, mask=mask, bkg_estimator=SExtractorBackground(),
                       box_size=(b_size, b_size), filter_size=(f_size, f_size),
                       sigma_clip=SigmaClip(sigma=3., maxiters=n_iter))
    back = Bkg.background * ~mask
    field_sub = field - back
    return field_sub, back

def display_background_sub(field, back):
    # Display and save background subtraction result with comparison 
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(16,5))
    ax1.imshow(field, origin="lower",cmap="gray", vmin=0., vmax=vmax_3sig(field))
    ax2.imshow(back, origin='lower', cmap='gray', vmin=0., vmax=vmax_3sig(field))
    ax3.imshow(field - back, origin='lower', cmap='gray', vmin=0., vmax=vmax_3sig(field))
    plt.tight_layout()
    return fig

def measure_pa_offset(pos1, pos2, wcs, obj_SE):
    (x1, y1) = pos1 
    (x2, y2) = pos2
    ra1, dec1 = wcs.all_pix2world(x1 + obj_SE.y_min, obj_SE.img_thumb.shape[1] - obj_SE.x_max + y1, 1)
    ra2, dec2 = wcs.all_pix2world(x2 + obj_SE.y_min, obj_SE.img_thumb.shape[1] - obj_SE.x_max + y2, 1)
    c1 = SkyCoord(ra1, dec1, frame='icrs', unit="deg")
    c2 = SkyCoord(ra2, dec2, frame='icrs', unit="deg")

    pa = np.mod(270 + np.arctan2(y2-y1,x2-x1)*180/np.pi, 360)
    offset = np.sqrt((x1-x2)**2+(y1-y2)**2)

    return pa, offset

def measure_EW(spec, wavl, z, line_stddev=3, plot=False):
    w_l = line_stddev * 2
    
    ln_range = ((6563.-w_l)*(1+z), (6563.+w_l)*(1+z))
    ln = (wavl>ln_range[0])&(wavl<ln_range[1]) & (wavl>wavl[0]+20) & (wavl<wavl[-1]-20) 

    con = (~ln) & (wavl>wavl[0]+30) & (wavl<wavl[-1]-30) & ((wavl<(6548.-20)*(1+z))|(wavl>(6584.+20)*(1+z)))
    cont = np.median(spec[con])
    EW = np.sum((spec[ln]-cont)/cont)
    if plot:
        plt.step(wavl,spec, where="mid")
        plt.step(wavl[ln],spec[ln], where="mid")
        plt.scatter(wavl[con],spec[con], color="seagreen",alpha=0.9)
        plt.axhline(cont,color="seagreen",alpha=0.8)
    return EW
    
# SE Object Pipeline start from here
class Object_SE:
    def __init__(self, obj, k_wid=8, cube=None, img_seg=None, deep_frame= None, mask_field=None):
        self.num = obj["NUMBER"][0]
        self.X_IMAGE, self.Y_IMAGE = obj["X_IMAGE"][0], obj["Y_IMAGE"][0]
        self.x_c, self.y_c = coord_ImtoArray(self.X_IMAGE, self.Y_IMAGE)
        self.width = np.max([obj["A_IMAGE"][0],3])
        self.a_image = obj["A_IMAGE"][0]
        self.b_image = obj["B_IMAGE"][0]
        self.theta = obj["THETA_IMAGE"][0]*np.pi/180.
        
        self.R_petro = obj["PETRO_RADIUS"][0]
        self.R_kron = obj["KRON_RADIUS"][0]
        self.flux_auto = obj["FLUX_AUTO"][0]
        
        self.x_min, self.x_max = (np.max([int(self.x_c - k_wid*self.width), 0]), 
                                 np.min([int(self.x_c + k_wid*self.width), cube.shape[1]]))
        self.y_min, self.y_max = (np.max([int(self.y_c - k_wid*self.width), 0]), 
                                 np.min([int(self.y_c + k_wid*self.width), cube.shape[2]]))
        self.center_pos=(self.y_c-self.y_min, self.x_c-self.x_min)
            
        self.mask_thumb = mask_field[self.x_min:(self.x_max+1), self.y_min:(self.y_max+1)]
        
        self.seg_thumb = img_seg[self.x_min:(self.x_max+1), self.y_min:(self.y_max+1)]
        self.mask_seg = (self.seg_thumb==obj["NUMBER"]) | (self.seg_thumb==0)
        self.mask_seg_obj = (self.seg_thumb==obj["NUMBER"])
        
        self.cube_thumb = cube[:,self.x_min:(self.x_max+1), self.y_min:(self.y_max+1)]
        
        if deep_frame is not None:
            self.deep_thumb = deep_frame[self.x_min:(self.x_max+1), self.y_min:(self.y_max+1)]
        else:
            self.deep_thumb = None
        
    def img_display(self, vmin=0., vmax=None, norm=norm):
        fig,ax=plt.subplots(1,1)
        if self.deep_thumb is None:
            img = self.img_thumb
        else:
            img = self.deep_thumb
        if vmax==None: 
            vmax=vmax_5sig(img)
        s = ax.imshow(img, norm=norm, origin="lower",cmap="hot", vmin=vmin, vmax=vmax)
        ax.plot(self.center_pos[0], self.center_pos[1], "lime", marker="+", ms=15, mew=2, alpha=0.7)
        ax.set_xticklabels(ax.get_xticks().astype('int64')+self.y_min)
        ax.set_yticklabels(ax.get_yticks().astype('int64')+self.x_min)
        plt.colorbar(s)
        plt.tight_layout()
        try:
            self.apers[0].plot(color='lime', linewidth=2.5, alpha=0.8)
            for aper in self.apers[1:]:
                aper.plot(color='lightgreen', linewidth=2 ,alpha=0.7)
            return ax
        except AttributeError:
            return ax
     
    def aper_photometry(self, k, k1, k2, ext_type="opt", print_out=False):
        apertures = EllipticalAperture(positions=self.center_pos, 
                                       a=k*self.a_image, b=k*self.b_image, theta=self.theta)
        
        annulus_apertures = EllipticalAnnulus(positions=self.center_pos, 
                                              a_in=k1*self.a_image, a_out=k2*self.a_image, 
                                              b_out=k2*self.b_image, theta=self.theta)

        apers = [apertures, annulus_apertures]
        m0 = apers[0].to_mask(method='exact')[0]
        area_aper = m0.to_image(self.img_thumb.shape)
        area_aper[~self.mask_thumb] = 0
        m1 = apers[1].to_mask(method='exact')[0]
        area_annu = m1.to_image(self.img_thumb.shape)
        area_annu[~self.mask_thumb] = 0

        bkg = self.img_thumb[np.logical_and(area_annu>=0.5,self.mask_seg)]
        signal = self.img_thumb[np.logical_and(area_aper>=0.5,self.mask_seg)]
        n_pix = len(signal)
        
        if (len(signal)==0)|(len(bkg)==0):
            if print_out: print("No signal/background!")
            return np.nan, np.nan, None
        if n_pix<=5: 
            if print_out: print("Too few pixels!")
            return np.nan, np.nan, None
        
#         S =  np.max([np.sum(signal) - n_pix * np.median(bkg), 0])
        S =  np.max([np.sum(signal) - n_pix * mad_std(bkg), 0])
        N_sky = n_pix/(n_pix-1) * np.sqrt(n_pix*mad_std(bkg)) 
        N_tot = np.sqrt(S + N_sky**2)
        
        if ext_type == "sky": N = N_sky
        else: N = N_tot
            
        if print_out: print ("Aperture: %.1f Rp; Signal: %.4f, Noise: %.4f, SNR = %.4f"%(k, S, N, S/N))
        return S, N, apers 
                   
    def compute_aper_opt(self, ks = np.arange(1.0,4.5,0.2), k1 = 5., k2 = 8., ext_type="opt",
                         print_out=True, plot=False):
        if self.R_petro==0:
            if print_out: print("Error in measuring R_petro, dubious source")
            return (1., 0)
        
        snr_ks = np.empty_like(ks)
        k_opt, snr_opt = 1, 0
        for i, k in enumerate(ks):
            S, N, apers = self.aper_photometry(k=k, k1=k1, k2=k2, ext_type=ext_type)
            snr_ks[i] = S/N
        try:
            if ext_type=='sky':
                k_opt, snr_opt = ks[np.nanargmax(snr_ks)], np.nanmax(snr_ks)
            elif ext_type=='opt':
                k_opt, snr_opt = ks[np.nanargmax(snr_ks)], np.nanmax(snr_ks)
            else: print('Extraction type: "sky" or "opt"?')
        except ValueError:
            return (1., 0)  # No available aperture, all k is None
        
        if print_out: print ("Optimal Aperture: %.1f Rp, SNR = %.4f"%(k_opt, snr_opt))
        if plot:
            plt.figure(figsize=(7,5))
            plt.plot(ks, snr_ks, color="mediumseagreen",lw=2.5,alpha=0.9)
            plt.axhline(snr_opt, color='indianred', lw=2,ls='--', alpha=0.8)
            plt.xlabel("Radius (R$_{P}$)", fontsize=16)
            plt.ylabel("S/N", fontsize=16)
#             plt.title("Extraction: %s"%ext_type, fontsize=14)
            plt.tight_layout()
#             plt.show()
            
        return k_opt, snr_opt
    
    def extract_spec(self, k, ext_type="opt", k1 = 5, k2 = 8, wavl=None, plot=False):
        
        try:
            S, N, apers = self.aper_photometry(k=k, k1 = k1, k2 = k2, ext_type=ext_type)
        except TypeError:
            self.apers = None
        if apers is None:
            print("Failed extraction... Few pixels or saturated stars")
            return np.zeros_like(wavl)
        self.apers = apers
        m0 = apers[0].to_mask(method='exact')[0]
        area_aper = m0.to_image(self.img_thumb.shape)
        
        spec = self.cube_thumb[:,np.where(area_aper>=0.5)[0],np.where(area_aper>=0.5)[1]].sum(axis=1)
#         if plot:
#             fig,ax=plt.subplots(1,1)
#             ax.plot(wavl,spec/spec.max(),lw=1)
        return spec  
    

def fit_cont(spec, wavl, model='GP', 
             kernel_scale=100, kenel_noise=1e-3, edge_ratio=0.15,
             print_out=True, plot=True, fig=None, ax=None):
    """Fit normalized continuum with Trapezoid1D model or Gaussian Process"""
    
    # rebin wavelength linearly in log scale
    logwavl = np.log(wavl)
    n_wavl = int(len(wavl))
    logwavl_intp = np.linspace(logwavl[0],logwavl[-1],n_wavl+1)
    f_interp = interpolate.interp1d(logwavl, spec, kind="cubic", fill_value="extrapolate")
    
    spec_intp = f_interp(logwavl_intp)
    wavl_rebin = np.e**logwavl_intp
    y_intp = spec_intp/spec_intp.max() # normalized spectra
    
    if model=='Trapz':
        # Fit continuum with Trapezoid 1D model in astropy
        cont = np.ma.masked_array(y_intp, mask=np.zeros_like(y_intp))
        fit_range = (y_intp - np.mean(y_intp)) <= mad_std(y_intp)
        cont.mask[~fit_range] = True
        model_init = models.Trapezoid1D(amplitude=np.mean(cont), x_0=(wavl[0]+wavl[-1])/2., width=250, slope=0.03,
                                        bounds={"amplitude":(0,1),"x_0":((wavl[0]+wavl[-1])/2.-5, (wavl[0]+wavl[-1])/2.+5),
                                                "width":(200,300),"slope":(-0.01,0.2)},
                                        fixed={"amplitude":False, "slope":False})
        fitter = fitting.SLSQPLSQFitter()
        model_fit = fitter(model_init, wavl_rebin, cont)
        cont_fit = model_fit(wavl_rebin)
        cont_range = (cont_fit==cont_fit.max())
        if print_out:
            print(model_fit)
            
    elif model=='GP':
        # Fit continuum with Gaussian Process
        if np.sum((y_intp - np.median(y_intp) <= -2.5*mad_std(y_intp)))> edge_ratio*n_wavl:  # For some special source with sharp edge
            # fit continuum where no drop or emission exist
            fit_range = abs(y_intp - np.median(y_intp)) <= mad_std(y_intp) 
        else:
            # fit continuum where no emission exist
            fit_range = (y_intp - np.median(y_intp)) <= mad_std(y_intp)
        cont = y_intp.copy()
        cont[~fit_range] = np.median(y_intp) 
        
        # continuum range to exclude edge
        cont_range = np.full(fit_range.shape, False)
        cont_range[np.argwhere(fit_range)[0][0]:np.argwhere(fit_range)[-1][0]] = True

        kernel = RBF(length_scale=kernel_scale) + WhiteKernel(noise_level=kenel_noise)
        gcr = GaussianProcessRegressor(kernel=kernel, random_state=0, optimizer=None)
        gcr.fit(wavl_rebin.reshape(-1,1), cont)
        cont_fit = gcr.predict(wavl_rebin.reshape(-1,1))
        
        try:
            y_intp[~cont_range] = cont_fit[~cont_range]
        except ValueError:
            pass
        
    res = (y_intp-cont_fit)
    
    if plot:
        # Plot the data with the best-fit model
        if fig==None:  
            fig, ax = plt.subplots(figsize=(8,3))
        ax.step(wavl, spec/spec.max(), "k", label="spec.", where="mid", alpha=0.9)
        ax.plot(np.e**logwavl_intp[fit_range], cont[fit_range], "s", color='indianred', ms=4, label="cont.",alpha=0.7)
        ax.plot(np.e**logwavl_intp[~fit_range], cont[~fit_range], "s", mec='indianred', mew=1.5, mfc="white", ms=4, alpha=0.7)
        ax.plot(wavl_rebin, cont_fit, "orange", label="cont. fit", alpha=0.9)        
        ax.step(wavl_rebin, res, "royalblue",label="residual", where="mid",zorder=5, alpha=0.9)
        ax.set_xlabel('wavelength ($\AA$)')
        ax.set_ylabel('norm Flux')
        ax.legend(ncol=2)
        plt.tight_layout()
          
    res -= np.median(res)
    res /= res.max()
    
    return res, wavl_rebin, cont_fit


def generate_template(wavl, z=0, n_intp=2, 
                      line_ratio=[1,1,1], box_wid=4, sigma=3,
                      temp_type="Ha-NII", temp_model="gauss", plot=True):
    """Generate a template for with line ratio [NII6548, Ha, NII6584]/[Hb, OIII4959, OIII5007] and width [box_wid]
    # For Gaussian models sigma(stddev)=sigma
    # For delta function width=box_wid"""
    wav_wid = wavl[-1] - wavl[0]
    n_temp = int(len(wavl)) * n_intp +1  # +1 from rebinning wavelength during continuum fitting
    if temp_type=="Ha-NII":
        lam_0 = 6563.
        line_pos = [6548.*(1+z), 6563.*(1+z), 6584.*(1+z)]
    elif temp_type=="Hb-OIII":
        lam_0 = 5007.
        line_pos = [4959.*(1+z), 5007.*(1+z)]
    elif temp_type=="OII":
        lam_0 = 3727.
        line_pos = [3727.*(1+z)]
        
    wavl_temp = np.e**np.linspace(np.log(lam_0*(1+z)-wav_wid/2.),np.log(lam_0*(1+z)+wav_wid/2.), n_temp)
        
    if temp_model == "box":
        s = np.sum([models.Box1D(amplitude=lr, x_0=lp, width=box_wid) for (lr, lp) in zip(line_ratio, line_pos)])
        temp = s(wavl_temp)
    elif (temp_model == "gauss")|(temp_model == "sincgauss"):
        s = np.sum([models.Gaussian1D(amplitude=lr, mean=lp, stddev=sigma) for (lr, lp) in zip(line_ratio, line_pos)])
        temp = s(wavl_temp)
        if temp_model == "sincgauss":
            ILS = np.sinc((wavl_temp-lam_0*(1+z))/5.)
            temp = np.convolve(temp,ILS, mode="same")
            temp[(wavl_temp<lam_0*(1+z)-50)|(wavl_temp>lam_0*(1+z)+50)]=0
    else: 
        print('Model type: "box" or "gauss"?')
        
    temp /= np.max([1, temp.max()])
    
    if plot:
        plt.step(wavl_temp, temp,"g",where="mid", alpha=0.5)
        for lp in line_pos:
            plt.axvline(lp, color="k", ls="-.", alpha=0.7) 
        
    return temp, wavl_temp  


def xcor_SNR(res, wavl_rebin, temps, wavl_temp, 
             d_wavl, n_intp=1, kind_intp="linear",
             z_sys=0.228, rv=None, edge=20,
             temp_type="Ha-NII", temp_model="gauss",
             temps_stddev=3, temps_ratio=None,
             plot=False, fig=None, axes=None, h_contrast=0.1):
    """Cross-correlate with template and return rative velocity to z0 and corr function
    
    res : normalized residual spectra
    wavl_rebin : rebinned wavelength
    rv : relative redshift and velocity to z=z0, None if not known (the first object in batch)
    """ 
    
    # interpolation data before cross-correlation to match template
    Interp_spec = interpolate.interp1d(wavl_rebin, res, kind=kind_intp)
    log_x = np.linspace(np.log(wavl_rebin)[0],np.log(wavl_rebin)[-1], int(len(wavl_rebin)-1) * n_intp +1)
    x = np.e**log_x
    y = Interp_spec(x)
    
    temps = np.atleast_2d(temps)
    
    if temp_type=="Ha-NII":
        line_pos = np.array([6548, 6563, 6584])
        line_name = ["[NII]6548", "H$\\alpha$", "[NII]6584"]
        lam_0 = 6563.
    elif temp_type=="Hb-OIII":
        line_pos = np.array([4959, 5007])
        line_name = ["[OIII]4959",  "[OIII]5007"]
        lam_0 = 5007.
    elif temp_type=="OII":
        line_pos = np.array([3727.])
        line_name = [" "]
        lam_0 = 3727.
      
    z0 = 6563.*(1+z_sys)/lam_0 - 1  # reference redshift, the matched line by default is Ha
    
    # Compute relative redshift and velocity to z=z0, if rv not given 
    if rv is None:
        cc = signal.correlate(y, temps[0], mode="full", method="direct")    
        
        rz = np.linspace((x[0] - d_wavl/n_intp * temps.shape[1]/2.) / lam_0 - 1, 
                         (x[-1] + d_wavl/n_intp * temps.shape[1]/2.) / lam_0 - 1, 
                         len(cc)) - z0    

        rv = rz * 3e5
    else:
        rz = rv/3e5
        
    # possible z range computed from filter range
    zmin, zmax = wavl_rebin[0]/lam_0 - 1., wavl_rebin[-1]/lam_0 - 1.

    # For all template
    ccs = np.zeros((temps.shape[0],len(rv)))
    z_ccs = np.zeros(temps.shape[0])
    SNR_ps = np.zeros(temps.shape[0])
    SNRs = np.zeros(temps.shape[0])
    Contrasts = np.zeros(temps.shape[0])
    
    rv_edge = ((np.max([x[x!=0][0], x[0]+edge])/lam_0-1-z0) * 3e5,   # x=0 is mock edge during the continuum fitting
               (np.min([x[x!=0][-1], x[-1]-edge])/lam_0-1-z0) * 3e5) 
    
    z_range = (rv>(zmin-z0)*3e5) & (rv<(zmax-z0)*3e5)
    find_range = z_range & (rv > rv_edge[0]) & (rv < rv_edge[1])
    
    for i, (temp, sigma, ratio) in enumerate(zip(temps, temps_stddev, temps_ratio)):
        # vicinity around lines to be excluded when measuring noise RMS
        if temp_model=="sincgauss":
            sigma *= 3 
        w_l = sigma * 3.
        w_l2 = sigma * 8.
        
        if temp_type=="Ha-NII":   #For strong AGN/composite use a broader window, the transform is to make it continous
            if ratio.max()/3. < 3.:  #<2:1=>1.5w_l, >3:1=>w_l
                if sigma > 4:  
                    if SNRs.max()>=50:  
                        w_l2 *= 1.5
        
        cc = signal.correlate(y, temp, mode="full", method="direct")
        cc = cc/cc.max()
        ccs[i] = cc
        
        # Peak redshift (via quadratic interpolation around the peak)
        # note: difference with a gaussian fitting to the peak is <0.0001 in z
        
        rv_p = rv[find_range][np.argmax(cc[find_range])]
        Interp_Peak = interpolate.interp1d(rv, cc, kind='linear')
       
        # rv_intp = np.linspace(np.max([rv_edge[0],rv_p - 3*w_l/lam_0*3e5]), np.min([rv_edge[1],rv_p + 3*w_l/lam_0*3e5]), 300)
        rv_intp = np.linspace(np.max([rv_edge[0],rv_p - 1500]), np.min([rv_edge[1],rv_p + 1500]), 150)
        cc_intp = Interp_Peak(rv_intp)
        rv_match = rv_intp[np.argmax(cc_intp)]
        rz_match = rv_intp[np.argmax(cc_intp)] / 3e5
        
        z_cc = z0 + rz_match
        z_ccs[i] = z_cc
        
        S_p = cc_intp.max()
        
        # peak_range = (abs(rv-rv_match)/3e5 < 1.5*w_l/lam_0)
        peak_range = (abs(rv-rv_match)/3e5 < w_l/lam_0)
        if temp_type=="Ha-NII":
            noise_peak_range = (~peak_range) & (rv > rv_edge[0]) & (rv < rv_edge[1])  # not use edge
           
            signal_range = ((rv > (rv_match - (w_l2+lam_0-6548.)/6548.*3e5)) &\
                            (rv < (rv_match + (w_l2+6584.-lam_0)/6584.*3e5)))
            noise_range =  ~signal_range & (rv > rv_edge[0]) & (rv < rv_edge[1])  # not use edge
            
        elif temp_type=="Hb-OIII":
            noise_peak_range = (~peak_range) & (rv > rv_edge[0]) & (rv < rv_edge[1])  # not use edge
            
            signal_range = (abs(rz-rz_match) < w_l/lam_0) | (abs(rz-4959.*(1+z_cc)/5007.-z0-1)<w_l/lam_0)
            noise_range = ~signal_range& (rv > rv_edge[0]) & (rv < rv_edge[1]) 
        
        elif temp_type=="OII":
            signal_range = peak_range
            noise_range = (~signal_range) & (rv > rv_edge[0]) & (rv < rv_edge[1])  # not use edge
            noise_peak_range = noise_range

#         N = np.std(sigma_clip(cc[z_range & noise_range], sigma=3, maxiters=10))
#         N = mad_std(cc[z_range & noise_range])
        N = np.std(cc[z_range & noise_range])
        SNRs[i] = S_p/N  # detection S/N
        
#         N_p = np.std(sigma_clip(cc[z_range & noise_peak_range], sigma=3, maxiters=10))
#         N_p = mad_std(cc[z_range & noise_peak_range])
        N_p = np.std(cc[z_range & noise_peak_range])
        SNR_ps[i] = S_p/N_p
        
    
    ###-------##
        if (temp_type=="Ha-NII"):
            try:
                ind_peak, _ = signal.find_peaks(cc_intp, height=h_contrast, distance = 50)
                peaks = np.sort(cc_intp[ind_peak])[::-1]

                if len(peaks)>=2:
                    Contrasts[i] = peaks[0]/peaks[1]
                else: Contrasts[i] = 1.0 #peaks[0]*(ratio.max()/3.)
            except ValueError:
                Contrasts[i] = -0.1
        
        elif (temp_type=="Hb-OIII"):
            rv_intp_b = np.linspace(np.max([rv_edge[0],(4959.*(1+z_cc)/5007.-z0-1)*3e5 - w_l/lam_0*3e5]), 
                                    np.min([rv_edge[1],(4959.*(1+z_cc)/5007.-z0-1)*3e5 + w_l/lam_0*3e5]), 100)
            rv_intp_all = np.concatenate([rv_intp_b,rv_intp])
            
            
            cc_intp_all = Interp_Peak(rv_intp_all)
            ind_peak, _ = signal.find_peaks(cc_intp_all, height=h_contrast, distance = 100)
            peaks = np.sort(cc_intp_all[ind_peak])[::-1]
            if len(peaks)>=2:
                Contrasts[i] = peaks[0]/peaks[1]
            else: Contrasts[i] = 1.0
            
        elif temp_type=="OII":
            Contrasts[i] = 1.0
            
    Rs = Contrasts*SNRs/np.max(SNRs)     #Significance
#     z_loc = (z_ccs>np.percentile(z_ccs,2.5)) & (z_ccs<np.percentile(z_ccs,97.5))
    r_max = np.argmax(Rs)
#     r_max = np.argmax(Contrasts)
    
    ###-------##
    z_best = z_ccs[r_max]
    line_at_edge = ((1+z_best)*lam_0<(x[0]+edge)) | ((1+z_best)*lam_0>(x[-1]-edge))   
    if line_at_edge: 
        flag_edge = 1  # near the filter edge, raise caution
    else: 
        flag_edge = 0

    if plot:
        if fig==None:
            fig = plt.figure(figsize=(8,6))
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)
        else:
            ax1, ax2 = axes
        ax1.step(wavl_rebin, res, c="royalblue", lw=2, where="mid", label="Residual", alpha=0.9, zorder=4)
#         ax1.plot(x, y, c="steelblue", linestyle='--', alpha=0.9, zorder=4)
        if (temps_ratio is None):
            ax1.step(wavl_temp*(1+z_ccs[r_max]),temps[r_max], color="seagreen", where="mid",
                     alpha=0.7, label="Template: z=%.3g"%z_ccs[r_max], zorder=3)
        else:
            if (temp_model == "gauss")|(temp_model == "sincgauss"):
                
                #Plot ideal model template
                sig_best, ratio_best = temps_stddev[r_max], temps_ratio[r_max]
                                
                print("Best z:",z_best, "Best sigma:",sig_best)
                line_pos = line_pos*(1+z_best)
                
                s = np.sum([models.Gaussian1D(amplitude=lr, mean=lp, stddev=sig_best) for (lr, lp) in zip(ratio_best, line_pos)])
                wavl_new = np.linspace(wavl_temp[0], wavl_temp[-1], 400)/(1+z0)*(1+z_best)
                temp_best = s(wavl_new)
                if (temp_model == "sincgauss"):
                    ILS = np.sinc((wavl_new-lam_0*(1+z_best))/5.)
                    temp_best = np.convolve(temp_best, ILS, mode="same")
                    temp_best[(wavl_new<lam_0*(1+z_best)-50)|(wavl_new>lam_0*(1+z_best)+50)]=0  # let higher-order lobe to be 0
                    
#                 ax1.step(wavl_temp/(1+z0)*(1+z_best), temps[r_max], color="g", where="mid",
#                          alpha=0.5, label="Template: z=%.3g"%z_best, zorder=3)  
                ax1.plot(wavl_new, temp_best/temp_best.max(), color="mediumseagreen", 
                     alpha=0.7, lw=2, linestyle='--',  zorder=5)    
                
                if temp_model=="sincgauss":
                    sig_best *= 3
                w_l, w_l2 = sig_best * 3., sig_best * 8.
                
                if temp_type=="Ha-NII":   #For strong AGN/composite use a broader window, the transform is to make it continous
                    if ratio_best.max()/3. < 3.:  #<2:1=>1.5w_l, >3:1=>w_l
                        if sig_best > 4:  
                            if SNRs[r_max]>=50:  
                                w_l2 *= 1.5                
                
                           
        for k, (lp, ln) in enumerate(zip(line_pos, line_name)):
            if ln!=" ":
                if (lp<x[0]+edge-5) | (lp>x[-1]-edge+5):
                    continue
                y_lab = res.max() if (k==1) else 0.85 * res.max()
                ax1.vlines(x=lp, ymin=y_lab+0.1, ymax=y_lab+0.18, color='k', lw=1.5)
                ax1.text(x=lp, y=y_lab+0.28, s=ln, color='k', ha="center", va="center", fontsize=10)
                ax1.set_ylim(res.min()-0.05, res.max()+0.375)
            else:
                ax1.set_ylim(res.min()-0.05, res.max()+0.25)
        ax1.set_xlim(x[0]-10, x[-1]+10)
        
        x_text = 0.75 if z_best < 0.24 else 0.12
        ax1.text(x_text, 0.85, "z = %.3g"%(z_best),
                 color='k',fontsize=14, transform=ax1.transAxes,alpha=0.95, zorder=4)
        ax1.set_xlabel('Wavelength ($\AA$)',fontsize=16)
        ax1.set_ylabel('Normed Flux',fontsize=16)
        
        for i, temp in enumerate(temps):    
            ax2.plot(rv, ccs[i], "firebrick", alpha=np.min([0.9, 0.9/len(temps)*3]))
        ax2.plot(rv, ccs[r_max], "red", lw=1, alpha=0.7, zorder=4)
        
        rv_left, rv_right = ((z_ccs[r_max]-z0 - w_l/lam_0)*3e5, 
                             (z_ccs[r_max]-z0 + w_l/lam_0)*3e5)  
        
        if temp_type=="Ha-NII":
            rv_left2, rv_right2 = (np.max([rv_edge[0],(z_ccs[r_max]-z0 - (w_l2+lam_0-6548.)/6548.) * 3e5]), 
                                   np.min([rv_edge[1],(z_ccs[r_max]-z0 + (w_l2+6584.-lam_0)/6584.) * 3e5]) )
            
        elif temp_type=="Hb-OIII":
            rv_left2, rv_right2 = rv_left, rv_right
            rv_left2b, rv_right2b = (np.max([rv_edge[0],((4959.*(1+z_ccs[r_max])/5007.-z0-1) - w_l/lam_0) * 3e5]),
                                     np.min([rv_edge[1],((4959.*(1+z_ccs[r_max])/5007.-z0-1) + w_l/lam_0) * 3e5]) )
            ax2.axvline((4959.*(1+z_best)/5007.-z0-1)*3e5, color="orangered", lw=1.5, ls="--", alpha=0.8)
            ax2.axvspan(rv_left2b, rv_right2b, alpha=0.15, color='indianred', zorder=2)
            
        elif temp_type=="OII":
            rv_left2, rv_right2 = rv_left, rv_right
            
        ax2.axvspan(rv_left, rv_right, alpha=0.3, color='indianred')
        ax2.axvspan(rv_left2, rv_right2, alpha=0.15, color='indianred')
        
        if temp_type=="Hb-OIII":
            if rv_edge[0]<rv_left2b:  
                ax2.axvspan(rv_edge[0], rv_left2b, alpha=0.2, color='gray', zorder=1)
                ax2.axvspan(rv_right2b, rv_left2, alpha=0.2, color='gray', zorder=1)
            if rv_edge[1]>rv_right2: 
                ax2.axvspan(rv_right2, rv_edge[1], alpha=0.2, color='gray', zorder=1)
        else:
            if rv_edge[0]<rv_left2:  
                ax2.axvspan(rv_edge[0], rv_left2, alpha=0.2, color='gray', zorder=1)
            if rv_edge[1]>rv_right2:  
                ax2.axvspan(rv_right2, rv_edge[1], alpha=0.2, color='gray', zorder=1)
        
        ##--Label the CC peak--##
        try:
            if temp_type=="Ha-NII":
                rv_intp = np.linspace(rv_left2, rv_right2, 300)
                d_min = 30
            elif temp_type=="Hb-OIII":
                rv_intp = np.concatenate([np.linspace(rv_left2b, rv_right2b, 150),
                                          np.linspace(rv_left2, rv_right2, 150)]) 
                d_min = 100
            elif temp_type=="OII":
                rv_intp = np.linspace(rv_left2, rv_right2, 150)
                d_min = 30
            Interp_Peak = interpolate.interp1d(rv, ccs[r_max], kind='quadratic')
            cc_intp = Interp_Peak(rv_intp)

            ind_peak, _ = signal.find_peaks(cc_intp, height=h_contrast, distance = d_min)
            plt.plot(rv_intp[ind_peak], cc_intp[ind_peak], "x", ms=5, color='orange',zorder=4) 
            
        except ValueError:
            pass
        ##----##
        
        ax2.axhline(0, color="k", lw=1.5, ls="--", alpha=0.7)
        ax2.axvline((z_ccs[r_max]-z0)*3e5, color="orangered", lw=1.5, ls="--", alpha=0.8)
        print("Peak wavl",x[np.argmax(y)])
                           
        ax2.text(x_text, 0.85, r"$\rm S/N: %.1f$"%(SNRs[r_max]),
                 color='k',fontsize=14, transform=ax2.transAxes,alpha=0.95, zorder=4)
        if temp_type!="OII":
            ax2.text(x_text, 0.7, r"$\rm R: %.1f$"%(Rs[r_max]),
                     color='k',fontsize=14, transform=ax2.transAxes,alpha=0.95, zorder=4)
        ax2.set_xlim((zmin-z0)*3e5, (zmax-z0)*3e5)
        ax2.set_xlabel('Relative Velocity (km/s) to z=%.3g'%z0,fontsize=16)
        ax2.set_ylabel('CCF',fontsize=16)
        axb = ax2.twiny()
        axb.plot(rv/3e5 + z0, np.ones_like(rv)) # Dummy plot
        axb.cla()
        axb.set_xlim(zmin,zmax)
        axb.set_xlabel("Redshift",fontsize=16)      
        
    return ccs, rv, z_ccs, Rs, Contrasts, SNRs, SNR_ps, flag_edge

    
def compute_centroid_offset(obj_SE, k_aper, spec, wavl, 
                            z_cc, pos_BCG=None, coord_BCG=None, 
                            niter_aper=15, ctol_aper=0.01, iso_snr=1.2, 
                            n_rand=199, aper_size = [0.7,0.8,0.9,1.1,1.2,1.],
                            centroid_type="APER", coord_type="angular",
                            wcs=None, affil_map=None, deep_img=None, 
                            emission_type="subtract", sum_type="weight", 
                            aperture_type="separate", multi_aper=False,
                            line_stddev=3, line_ratio=None, mag0=25.2, 
                            plot=True, print_out=True):    
    # Compute the centroid of emission and continua, and plot
    w_l = line_stddev * 3
#     if line_ratio<3:
#         em_range = ((6548.-w_l)*(1+z_cc), (6584.+w_l)*(1+z_cc))
#     else: 
#         em_range = ((6563.-w_l)*(1+z_cc), (6563.+w_l)*(1+z_cc))
    em_range = ((6548.-w_l)*(1+z_cc), (6584.+w_l)*(1+z_cc))
        
    em = (wavl>em_range[0])&(wavl<em_range[1]) & (wavl>wavl[0]+15) & (wavl<wavl[-1]-15) 
#     img_em = obj_SE.cube_thumb[em,:,:].sum(axis=0)/np.sum(em)

    w2 = line_stddev * 5
    con = (~em) & (wavl>wavl[0]+25) & (wavl<wavl[-1]-25) & ((wavl<(6548.-w2)*(1+z_cc))|(wavl>(6584.+w2)*(1+z_cc)))

    if sum_type=="mean":
        img_em = np.mean(obj_SE.cube_thumb[em,:,:],axis=0)
        img_con = np.mean(obj_SE.cube_thumb[con,:,:], axis=0)
        
    elif sum_type=="median":    
        img_con = np.median(obj_SE.cube_thumb[con,:,:], axis=0)
        img_em = np.mean(obj_SE.cube_thumb[em,:,:],axis=0)
        
    elif sum_type=="weight":
        img_con = np.median(obj_SE.cube_thumb[con,:,:], axis=0)
#         img_con_std = mad_std(obj_SE.cube_thumb[con,:,:], axis=0)
        img_em0 = obj_SE.cube_thumb[em,:,:]
        
        weight_map = abs(img_em0-img_con)#/img_con
#         weight_map[(img_em0-img_con)<-2.5*img_con_std] = 0
        weight_sum = np.sum(weight_map, axis=0)
        
        img_em = np.sum(img_em0*weight_map/weight_sum, axis=0)
       
    
    if emission_type=="subtract":
        img_em = img_em - img_con
    elif emission_type=="pixel":
        img_em = img_em - np.min(img_em)
        img_con = img_con - np.min(img_con)
        img_em = img_em - img_con
    else:
        img_em = img_em
    
    if print_out:
        if centroid_type == "APER":
            print("optimal aperture:%.1f"%k_aper)

    mask_obj = obj_SE.mask_seg
#     mask_obj = obj_SE.mask_seg_obj
    if aperture_type== "separate":
        theta = obj_SE.theta
        if multi_aper==False:
            aper_size = [1]
            n_rand = 0
        x1s, y1s = np.zeros((n_rand+1,len(aper_size))),np.zeros((n_rand+1,len(aper_size)))
        x2s, y2s = np.zeros((n_rand+1,len(aper_size))),np.zeros((n_rand+1,len(aper_size)))
        aper_ems  = np.array([])
        aper_cons  = np.array([])
        if print_out:
            print("Aper (min,max): (%.2f,%.2f)"%(k_aper*aper_size[0],k_aper*aper_size[-1]))
        for k, per_apaer in enumerate(aper_size):
            a = np.max([obj_SE.a_image*per_apaer,1.])
#             a = obj_SE.a_image*per_apaer
            b = a/obj_SE.a_image * obj_SE.b_image
            da_s = np.append(stats.norm(loc=0, scale=a/3.).rvs(n_rand), 0)
            db_s = np.append(stats.norm(loc=0, scale=b/3.).rvs(n_rand), 0)
            
            for j,(da,db) in enumerate(zip(da_s, db_s)):
                try:
                    for d,img in enumerate([img_em, img_con]):
                        dx, dy = da*np.cos(theta)-db*np.sin(theta), da*np.sin(theta)+db*np.cos(theta)
                        x, y = obj_SE.center_pos[0] + dx, obj_SE.center_pos[1] + dy
                        
                        for i in range(niter_aper):
                            aper = EllipticalAperture(positions=(x,y), 
                                                      a=k_aper*a, b=k_aper*b, theta=obj_SE.theta)              
                            m0 = aper.to_mask(method='exact')[0]
                            mask_aper = m0.to_image(obj_SE.img_thumb.shape)

                            data = img * mask_aper * mask_obj
                            if np.sum(data) < 0:
                                x1s[j,k], y1s[j,k] = np.nan, np.nan
                                x2s[j,k], y2s[j,k] = np.nan, np.nan
                                aper_em, aper_con = None, None                              
                                break
                            x_new, y_new = centroid_com(data, mask=np.logical_not(mask_aper * mask_obj))
                            
                            reach_ctol = (np.sqrt((x-x_new)**2+(y-y_new)**2) < ctol_aper)   
#                             reach_aperture = (np.sqrt((x_new-obj_SE.center_pos[0])**2+(y_new-obj_SE.center_pos[1])**2) > 2*obj_SE.R_petro)
                            if  reach_ctol: 
                                # if reach n_iter, or the bound of optimal aperture, stop
                                break
                            else:
                                x, y = x_new, y_new
                        if d==0:
                            data_em = img_em * mask_aper * mask_obj
                            x1, y1 = x, y
                            x1s[j,k], y1s[j,k] = x,y
                            aper_em = EllipticalAperture(positions=(x1,y1), 
                                                      a=k_aper*a, b=k_aper*b, theta=obj_SE.theta)


                        elif d==1:
                            data_con = img_con * mask_aper * mask_obj
                            x2, y2 = x, y
                            x2s[j,k], y2s[j,k] = x, y
                            aper_con = EllipticalAperture(positions=(x2,y2), 
                                                      a=k_aper*a, b=k_aper*b, theta=obj_SE.theta)
                except (ValueError, TypeError) as error:
                    x1s[j,k], y1s[j,k] = np.nan, np.nan
                    x2s[j,k], y2s[j,k] = np.nan, np.nan
                    aper_em, aper_con = None, None
                    continue
                        
            aper_ems = np.append(aper_ems, aper_em)
            aper_cons = np.append(aper_cons, aper_con)
            
        
        use_value_1, use_value_2 = np.ones_like(aper_size, dtype=bool), np.ones_like(aper_size, dtype=bool)
        sigma_1s, sigma_2s = np.ones_like(aper_size), np.ones_like(aper_size)
            
        for k in range(len(aper_size)):
            r1_k = np.sqrt((x1s[:,k]-np.nanmean(x1s[:,k]))**2+(y1s[:,k]-np.nanmean(y1s[:,k]))**2)
            r2_k = np.sqrt((x2s[:,k]-np.nanmean(x2s[:,k]))**2+(y2s[:,k]-np.nanmean(y2s[:,k]))**2)
            sigma_1s[k], sigma_2s[k] = np.nanstd(r1_k), np.nanstd(r2_k)
#             print("\naper:",aper_size[k])
#             print((np.nanstd(r1_k)))
#             print((np.nanstd(r2_k)))
            # if centroid diverge, set the centroid to be nan
            if (np.nanstd(r1_k)>.1): use_value_1[k] = False
            if (np.nanstd(r2_k)>.1): use_value_2[k] = False
        
        if (np.sum(use_value_1)==0)|(np.sum(use_value_2)==0):
            return (0, 0 , 0)
        
        x1_eff, y1_eff = x1s[-1][use_value_1], y1s[-1][use_value_1]
        x2_eff, y2_eff = x2s[-1][use_value_2], y2s[-1][use_value_2]
        sigma_1_eff, sigma_2_eff = sigma_1s[use_value_1], sigma_2s[use_value_2]
        
        aper_ems_eff, aper_cons_eff = aper_ems[use_value_1], aper_cons[use_value_2]
#         print(use_value)

        

    elif aperture_type== "single":
        img = obj_SE.img_thumb
        x, y = obj_SE.center_pos
        for i in range(niter_aper):
            aper = EllipticalAperture(positions=(x,y), 
                                      a=k_aper*obj_SE.a_image, b=k_aper*obj_SE.b_image, theta=obj_SE.theta)
            m0 = aper.to_mask(method='exact')[0]
            mask_aper = m0.to_image(obj_SE.img_thumb.shape)           
            data = img * mask_aper * mask_obj
            x_new, y_new = centroid_com(data, mask=np.logical_not(mask_aper * mask_obj))
            reach_ctol = (np.sqrt((x-x_new)**2+(y-y_new)**2) < ctol_aper)
#             reach_aperture = (np.sqrt((x_new-obj_SE.center_pos[0])**2+(y_new-obj_SE.center_pos[1])**2) > 2*obj_SE.R_petro)
            if  reach_ctol: 
                # if reach n_iter, or the bound of optimal aperture, stop
                break
            else:
                x, y = x_new, y_new
        data_em = img_em * mask_aper * mask_obj
        x1, y1 = centroid_com(data_em, mask=np.logical_not(mask_aper * mask_obj))
        aper_em = aper
        data_con = img_con * mask_aper * mask_obj
        x2, y2 = centroid_com(data_con, mask=np.logical_not(mask_aper * mask_obj))
        aper_con = aper
        
    elif centroid_type == 'ISO1':
        sigma = 3.0 * gaussian_fwhm_to_sigma    # FWHM = 3.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        kernel.normalize()
        
        threshold1 = detect_threshold(img_em, snr=1.2)
        segm1 = detect_sources(img_em, threshold1, npixels=5, filter_kernel=kernel)
        segm_deblend_em = deblend_sources(img_em, segm1, npixels=5, filter_kernel=kernel, nlevels=32,contrast=0.001)
        
        threshold2 = detect_threshold(img_con, snr=1.2)
        segm2 = detect_sources(img_con, threshold2, npixels=5, filter_kernel=kernel)
        segm_deblend_con = deblend_sources(img_con, segm2, npixels=5, filter_kernel=kernel, nlevels=32,contrast=0.001)
        
        tbl_em = source_properties(img_em, segm_deblend_em).to_table()
        tbl_con = source_properties(img_con, segm_deblend_con).to_table()
        label_em = np.sqrt((tbl_em["xcentroid"].value-obj_SE.img_thumb.shape[1]/2.)**2+(tbl_em["ycentroid"].value-obj_SE.img_thumb.shape[0]/2.)**2).argmin()
        label_con = np.sqrt((tbl_con["xcentroid"].value-obj_SE.img_thumb.shape[1]/2.)**2+(tbl_con["ycentroid"].value-obj_SE.img_thumb.shape[0]/2.)**2).argmin()
        x1, y1 = tbl_em["xcentroid"][label_em].value, tbl_em["ycentroid"][label_em].value
        x2, y2 = tbl_con["xcentroid"][label_con].value, tbl_con["ycentroid"][label_con].value
        
        data_em  = img_em*(segm_deblend_em.data==(1+label_em))
        data_con  = img_con*(segm_deblend_con.data==(1+label_con))
        
    elif centroid_type == 'ISO2':
        sigma = 3.0 * gaussian_fwhm_to_sigma    # FWHM = 3.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        kernel.normalize()
        
        threshold = detect_threshold(obj_SE.img_thumb, snr=iso_snr)
        segm = detect_sources(obj_SE.img_thumb, threshold, npixels=5, filter_kernel=kernel)
        segm_deblend = deblend_sources(obj_SE.img_thumb, segm, npixels=5, filter_kernel=kernel, nlevels=32,contrast=0.001)
        
        tbl = source_properties(obj_SE.img_thumb, segm_deblend).to_table()
        label = np.sqrt((tbl["xcentroid"].value-obj_SE.img_thumb.shape[1]/2.)**2+(tbl["ycentroid"].value-obj_SE.img_thumb.shape[0]/2.)**2).argmin()
        
        data_em = img_em * (segm_deblend.data==(1+label))
        data_con = img_con * (segm_deblend.data==(1+label))
        x1, y1 = centroid_com(data_em)
        x2, y2 = centroid_com(data_con)
    

    
    #------------#        
    if coord_type=="euclid":
        centroid_offset = np.sqrt((x1-x2)**2+(y1-y2)**2)
        if np.ndim(pos_BCG)==1:
            (x_BCG, y_BCG) = pos_BCG

        else:
            lab = affil_map[obj_SE.x_c.astype("int64"), obj_SE.y_c.astype("int64")]
            (x_BCG, y_BCG) = pos_BCG[0] if lab==1 else pos_BCG[1]     

        dist_clus_cen = np.sqrt((x_BCG-obj_SE.X_IMAGE-1)**2 + (y_BCG-obj_SE.Y_IMAGE-1)**2)  

        pa = np.arctan2(x2-x1,y2-y1)
        clus_cen_angle = np.arctan2((x_BCG-obj_SE.X_IMAGE-1),(y_BCG-obj_SE.Y_IMAGE-1))
        if print_out:
            print("PA: %.3f,  Cluster-centric angle: %.3f"%(pa*180/np.pi, clus_cen_angle*180/np.pi))
        diff_angle = np.abs(pa*180/np.pi - clus_cen_angle*180/np.pi)
        if diff_angle>180:
            diff_angle = 360 - diff_angle
            
            
    elif coord_type=="angular":
        
        if np.ndim(coord_BCG)==0:
#             (ra_BCG, dec_BCG) = coord_BCG.ra, coord_BCG.dec
            c_BCG = coord_BCG

        else:
            lab = affil_map[obj_SE.x_c.astype("int64"), obj_SE.y_c.astype("int64")]
#             (ra_BCG, dec_BCG) = (coord_BCG[0].ra,coord_BCG[0].dec) if lab==1 else (coord_BCG[1].ra,coord_BCG[1].dec) 
            c_BCG = coord_BCG[0] if lab==1 else coord_BCG[1]

        ra0, dec0 = wcs.all_pix2world(obj_SE.X_IMAGE, obj_SE.Y_IMAGE, 1)
        c0 = SkyCoord(ra0, dec0, frame='icrs', unit="deg")

        dist_clus_cen = c0.separation(c_BCG).to(u.arcsec).value / 0.322
        clus_cen_angle = c0.position_angle(c_BCG).to(u.deg).value
            
    
        if aperture_type== "separate":
            
            #----#
#             pa_s = np.array([]) #
#             centroid_offsets = np.array([]) #
#             dist_clus_cens = np.array([]) #
#             for i,(x1,y1,x2,y2) in enumerate(zip(x1_eff, y1_eff, x2_eff, y2_eff)):

#                 ##Correct for distortion 
#                 pa, centroid_offset = measure_pa_offset((x1,y1), (x2,y2), wcs, obj_SE)
                
#                 diff_angle = np.abs(pa - clus_cen_angle)
#                 if diff_angle>180:
#                     diff_angle = 360 - diff_angle

#                 pa_s = np.append(pa_s, pa)
#                 centroid_offsets = np.append(centroid_offsets, centroid_offset)
                
#             centroid_offset = np.median(centroid_offsets)
#             pa = np.median(pa_s)
#             diff_angle = np.abs(pa - clus_cen_angle)
#             if diff_angle>180:
#                     diff_angle = 360 - diff_angle

#             if (len(x1_eff)>len(aper_size)//2)&(len(x2_eff)>len(aper_size)//2): # if half of used aperture is valied
#                 x1, y1 = np.median(x1_eff), np.median(y1_eff)
#                 x2, y2 = np.median(x2_eff), np.median(y2_eff)
#             else:
#                 x1, y1 = obj_SE.center_pos
#                 x2, y2 = obj_SE.center_pos
#                 centroid_offset = 0
            
            #----#
            if (len(x1_eff)>0)&(len(x2_eff)>0):
#                 x1 = np.sum(x1_eff/(sigma_1_eff)**2)/np.sum(1./(sigma_1_eff)**2)
                i_1, i_2 = np.argmin(sigma_1_eff), np.argmin(sigma_2_eff)
                x1, y1 = x1_eff[i_1], y1_eff[i_1]
                x2, y2 = x2_eff[i_2], y2_eff[i_2]
            
                aper_em = aper_ems_eff[i_1]
                aper_con = aper_cons_eff[i_2]
            
                if multi_aper:
                    Pos_1 = np.vstack([x1_eff,y1_eff]).T
                    Pos_2 = np.vstack([x2_eff,y2_eff]).T
                    max_dev = np.max([np.max(pdist(Pos_1)), np.max(pdist(Pos_2))]) 
                else:
                    max_dev = 0
            else:
                x1, y1 = obj_SE.center_pos
                x2, y2 = obj_SE.center_pos
                centroid_offset = 0
            
#             print("x1",x1_eff)
#             print("x2",x2_eff)
#             print("y1",y1_eff)
#             print("y2",y2_eff)
#             print("Pos em:",x1,y1)
#             print("Pos con:",x2,y2)
            
            pa, centroid_offset = measure_pa_offset((x1,y1), (x2,y2), wcs, obj_SE)
            diff_angle = np.abs(pa - clus_cen_angle)
            if diff_angle>180:
                diff_angle = 360 - diff_angle
            
#             k_aper1 = np.max([aper_size[np.argmin(sigma_1_eff)],1./a])
#             k_aper2 = np.max([aper_size[np.argmin(sigma_2_eff)],1./a])
#             aper_em = EllipticalAperture(positions=(x1,y1), 
#                               a=k_aper1*a, b=k_aper1*b, theta=obj_SE.theta)
#             aper_con = EllipticalAperture(positions=(x2,y2), 
#                               a=k_aper2*a, b=k_aper2*b, theta=obj_SE.theta)

#             print("sigma1/a",np.min(sigma_1_eff)/k_aper1*a)
#             print("sigma2/a",np.min(sigma_2_eff)/k_aper1*a)
            
        else:
            ra1, dec1 = wcs.all_pix2world(x1+obj_SE.y_min,obj_SE.img_thumb.shape[1]-obj_SE.x_max+y1, 1)
            ra2, dec2 = wcs.all_pix2world(x2+obj_SE.y_min,obj_SE.img_thumb.shape[1]-obj_SE.x_max+y2, 1)
            c1 = SkyCoord(ra1, dec1, frame='icrs', unit="deg")
            c2 = SkyCoord(ra2, dec2, frame='icrs', unit="deg")

            pa = np.mod(270 + np.arctan2(y2-y1,x2-x1)*180/np.pi, 360)
            centroid_offset = np.sqrt((x1-x2)**2+(y1-y2)**2)
            
            diff_angle = np.abs(pa - clus_cen_angle)
            if diff_angle>180:
                diff_angle = 360 - diff_angle

    
    #------------#
    
    if plot:
        plt.figure(figsize=(14,12))
        ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
        ax0.plot(wavl, spec, "whitesmoke",alpha=0.7,lw=2)
        ax0.plot(wavl[con & (wavl<6563.*(1+z_cc))], spec[con & (wavl<6563.*(1+z_cc))], "firebrick",alpha=0.9,lw=2)
        ax0.plot(wavl[con & (wavl>6563.*(1+z_cc))], spec[con & (wavl>6563.*(1+z_cc))], "firebrick",alpha=0.9,lw=2)
        ax0.plot(wavl[em],spec[em],"steelblue",alpha=1,lw=2.5)
        ax0.axvline(np.max([wavl[0], em_range[0]]), color="steelblue", lw=1.5, ls="-.", alpha=0.7)
        ax0.axvline(np.min([wavl[-1], em_range[1]]), color="steelblue", lw=1.5, ls="-.", alpha=0.7)
        ax0.set_xlabel('Wavelength ($\AA$)',fontsize=16)
        ax0.set_ylabel('Flux (10$^{-17}$erg/s/cm$^2$)',fontsize=16)
        ax0.text(0.9,0.9,"z = %.3f"%z_cc,color="k",fontsize=15,transform=ax0.transAxes)

        ax1 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1)
        s = ax1.imshow(img_em, origin="lower",cmap="viridis",vmin=0.,vmax=2*vmax_5sig(img_em))
        ax1.plot(x1, y1, "violet", marker="+", ms=10, mew=3,alpha=0.95)
       
        loc = plticker.MultipleLocator(base=10) # this locator puts ticks at regular intervals
        ax1.xaxis.set_major_locator(loc)
        ax1.yaxis.set_major_locator(loc)
        ax1.set_xticklabels(ax1.get_xticks().astype("int")+obj_SE.y_min,fontsize=10)
        ax1.set_yticklabels(ax1.get_yticks().astype("int")+obj_SE.x_min,fontsize=10)
        ax1.set_title("Emisson",color="k",fontsize=18)
        plt.colorbar(s)
        if centroid_type != "APER":
            ax1s = plt.subplot2grid((3, 3), (1, 0), colspan=1, rowspan=1)
            ax1s.imshow(data_em, origin="lower",cmap="viridis",vmin=0,vmax=vmax_5sig(data_em))
            ax1s.set_xticklabels(ax1s.get_xticks().astype("int")+obj_SE.y_min,fontsize=10)
            ax1s.set_yticklabels(ax1s.get_yticks().astype("int")+obj_SE.x_min,fontsize=10)
            ax1s.set_title("Em. seg.",color="k",fontsize=15)
            
            
        ax2 = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1)
        s = ax2.imshow(img_con*np.sum(con), origin="lower",cmap="hot",vmin=0.,vmax=2*vmax_5sig(img_con*np.sum(con)))
        ax2.plot(x2, y2, "lightgreen", marker="+", ms=10, mew=3,alpha=0.95)
        ax2.xaxis.set_major_locator(loc)
        ax2.yaxis.set_major_locator(loc)
        ax2.set_xticklabels(ax2.get_xticks().astype("int")+obj_SE.y_min,fontsize=10)
        ax2.set_yticklabels(ax2.get_yticks().astype("int")+obj_SE.x_min,fontsize=10)
        ax2.set_title("Continuum",color="k",fontsize=18)
        plt.colorbar(s)
        if centroid_type != "APER":
            ax2s = plt.subplot2grid((4, 3), (3, 2), colspan=1, rowspan=1)
            ax2s.imshow(data_con, origin="lower",cmap="hot",vmin=0.,vmax=vmax_5sig(data_con))
            ax2s.set_xticklabels(ax2s.get_xticks().astype("int")+obj_SE.y_min,fontsize=10)
            ax2s.set_yticklabels(ax2s.get_yticks().astype("int")+obj_SE.x_min,fontsize=10)
            ax2s.set_title("Cont. seg.",color="k",fontsize=15)
            
            
        
#         if centroid_type == "APER":
            
#             ax3 = plt.subplot2grid((4, 3), (1, 2), colspan=1, rowspan=1)
#             ax3.imshow(mask_aper * obj_SE.mask_seg, origin="lower", cmap="gray")
#             ax3.set_title("segmentation",color="k",fontsize=12)
        
        ax = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
        
        if deep_img is None:
#             img_g = (img_con+img_em)/2.
#             img_rgb = make_lupton_rgb(img_con, img_g, img_em, stretch=0.05, Q=10)
#             ax.imshow(img_rgb, origin="lower", vmin=0., vmax=vmax_5sig(img_rgb), norm=norm)
            ax.imshow(obj_SE.img_thumb, origin="lower", cmap="gray", vmin=0., vmax=2*vmax_5sig(obj_SE.img_thumb), norm=norm)
        else:
            ax.imshow(deep_img, origin="lower", cmap="gray", vmin=0., vmax=2*vmax_5sig(deep_img), norm=norm)
        
        xlim,ylim = ax.get_xlim(),ax.get_ylim()
        if centroid_type == "APER":
            for k, (aper1, aper2) in enumerate(zip(aper_ems,aper_cons)):
                try:
                    ls1 = '-' if (use_value_1[k]==True) else '--'
                    ls2 = '-' if (use_value_2[k]==True) else '--'
                    aper1.plot(color='violet',lw=1,ls=ls1,ax=ax1,alpha=0.9)
                    aper2.plot(color='darkseagreen',lw=1,ls=ls2,ax=ax2,alpha=0.9) 
                except AttributeError:
                    pass                    
            aper_em.plot(color='violet',lw=3,ax=ax,alpha=0.95)
            aper_con.plot(color='darkseagreen',lw=3,ax=ax,alpha=0.95)
        ax.contour(data_em,colors="lightblue",alpha=0.95, linewidths=2.5,
                  levels = [0.25*data_em.max(),0.5*data_em.max(),0.75*data_em.max(),data_em.max()])
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("X (pix)",fontsize=15)
        ax.set_ylabel("Y (pix)",fontsize=15)
        
#         for k in range(len(aper_size)):
#             for j in range(n_rand):
#                 ax1.plot(x1s[j,k], y1s[j,k], color="plum", marker=".", ms=3,mew=1,alpha=0.8, zorder=3)
#                 ax2.plot(x2s[j,k], y2s[j,k], "palegreen", marker=".", ms=3,mew=1,alpha=0.8, zorder=3)
#         ax1.scatter(obj_SE.center_pos[0]+da_s*np.cos(theta)-db_s*np.sin(theta), obj_SE.center_pos[1]+da_s*np.sin(theta)+db_s*np.cos(theta), color="orange", alpha=0.3,s=1)
#         ax2.scatter(obj_SE.center_pos[0]+da_s*np.cos(theta)-db_s*np.sin(theta), obj_SE.center_pos[1]+da_s*np.sin(theta)+db_s*np.cos(theta), color="orange", alpha=0.3,s=1)
#         ax.scatter(obj_SE.center_pos[0]+da_s*np.cos(theta)-db_s*np.sin(theta), obj_SE.center_pos[1]+da_s*np.sin(theta)+db_s*np.cos(theta), color="orange", alpha=0.3,s=5)
        
        ax.plot(x1, y1, color="violet", marker="+", ms=20,mew=4,alpha=0.95, zorder=4)
        ax.plot(x2, y2, color="lightgreen", marker="+", ms=20,mew=4,alpha=0.95, zorder=3)
        ax.set_xticklabels(ax.get_xticks().astype("int")+obj_SE.y_min)
        ax.set_yticklabels(ax.get_yticks().astype("int")+obj_SE.x_min)
        if coord_type=="euclid":
            ax.arrow(0.9,0.1,np.sin(pa)/15,np.cos(pa)/15,color="skyblue",
                     head_width=0.02, head_length=0.02,transform=ax.transAxes)
            ax.arrow(0.9,0.1,np.sin(clus_cen_angle)/15,np.cos(clus_cen_angle)/15,color="gold",
                     head_width=0.02, head_length=0.02,transform=ax.transAxes)
        elif coord_type=="angular":
            ax.arrow(0.9,0.1,-np.sin(pa*np.pi/180)/15,np.cos(pa*np.pi/180)/15,color="skyblue",
                     head_width=0.02, head_length=0.02,transform=ax.transAxes,alpha=0.95)
            ax.arrow(0.9,0.1,-np.sin(clus_cen_angle*np.pi/180)/15,np.cos(clus_cen_angle*np.pi/180)/15,color="gold",
                     head_width=0.02, head_length=0.02,transform=ax.transAxes,alpha=0.95)
        
        ax.text(0.05,0.05,r"$\bf \theta:{\ }%.1f$"%diff_angle,color="lavender",fontsize=16,transform=ax.transAxes)
        ax.text(0.05,0.1,r"$\bf \Delta\,d:{\ }%.1f$"%centroid_offset,color="lavender",fontsize=16,transform=ax.transAxes)
        mag_auto = -2.5*np.log10(obj_SE.flux_auto) + mag0
        #ax.text(0.05,0.9,"mag: %.1f"%mag_auto,color="lavender",fontsize=15,transform=ax.transAxes)
        if np.ndim(coord_BCG)>0:
            text, color = (r'$\bf NE$', 'lightcoral') if lab==1 else (r'$\bf SW$', 'lightblue')
            ax.text(0.9, 0.9, text, color=color, fontsize=18, transform=ax.transAxes)
        plt.subplots_adjust(left=0.05,right=0.95,top=0.95, bottom=0.05, wspace=0.2, hspace=0.25)
    
    return diff_angle, centroid_offset, dist_clus_cen, pa, clus_cen_angle, max_dev




def draw_centroid_offset(diff_centroids_v, diff_centroids_c, crit=[0.5, 0.8]):
    import seaborn as sns
    sns.distplot(diff_centroids_c[diff_centroids_c<np.percentile(diff_centroids_c,99)],
                 color="gray",label="All")
    sns.distplot(diff_centroids_v,color="seagreen",label="Candidate")
    p = crit[0]#np.percentile(diff_centroids_c, crit[0]*100)
    q = crit[1]#np.percentile(diff_centroids_c, crit[1]*100)
    plt.axvline(p, color="k",label="$\Delta\,d$ = %.1f pix"%p)
    plt.axvline(q, color="k",ls='--',label="$\Delta\,d$ = %.1f pix"%q)
    plt.xlabel("centroid offset $\Delta\,d$",fontsize=14)
    plt.legend(loc='best',fontsize=12)
    
def draw_angle_control(diff_angles_c, diff_centroids_c, crit=[0.5, 0.8], b=7):
    import seaborn as sns
    p = crit[0]#np.percentile(diff_centroids_c, crit[0]*100)
    q = crit[1]#np.percentile(diff_centroids_c, crit[1]*100)

    plt.hist(diff_angles_c, bins=np.linspace(0,180,b), color="k",
             normed=True, histtype="step", linestyle="-",linewidth=3,alpha=0.9, label="centroid offset all")    
    plt.hist(diff_angles_c[diff_centroids_c>p], bins=np.linspace(0,180,b), color="k",
             normed=True, histtype="step", linestyle="--",linewidth=3,alpha=0.9, label="centroid offset > %.1f pix"%p)    
    plt.hist(diff_angles_c[diff_centroids_c>q], bins=np.linspace(0,180,b), color="k",
             normed=True, histtype="step", linestyle=":", linewidth=3, alpha=0.9, label="centroid offset > %.1f pix"%q)  
    plt.xlabel("$\\theta$",fontsize=14)
    plt.legend(loc=8,fontsize=12)
    
def draw_angle_candidate(diff_angles_v, diff_centroids_v, crit=[0.5, 0.8], b=7):    
    import seaborn as sns
    p, q = crit
    print("# of offset > %d pix :%d"%(100*crit[0],np.sum(diff_centroids_v>p)))
    print("# of offset > %d pix :%d"%(100*crit[1],np.sum(diff_centroids_v>q)))

    plt.hist(diff_angles_v[diff_centroids_v>p]+2, bins=np.linspace(0,180,b)+2,
             normed=False, histtype="step", linewidth=4, alpha=0.5, label="offset > %.1f pix"%p)
    plt.hist(diff_angles_v[diff_centroids_v>q]-2, bins=np.linspace(0,180,b)-2,
             normed=False, histtype="step", linewidth=4, alpha=0.5, label="offset > %.1f pix"%q)
#     plt.hist(diff_angles_v, bins=np.linspace(0,180,b),
#              normed=False, histtype="step", linewidth=4, alpha=0.5, label="candidate all")
    plt.xlabel("$\\theta$",fontsize=14)
    plt.legend(loc=9,fontsize=12)
    
def draw_angle_radial(diff_angles_v, diff_centroids_v, distance, r0 = .8, q=1, b=7):    
    import seaborn as sns
    inner = (diff_centroids_v>q) & (distance<=r0)
    outer = (diff_centroids_v>q) & (distance>r0)
    
    print("# of offset > %d pix all:%d"%(q,np.sum(diff_centroids_v>q)))
    print("# of offset > %d pix inner:%d"%(q,np.sum(inner)))
    print("# of offset > %d pix outer:%d"%(q,np.sum(outer)))
    
    
    plt.hist(diff_angles_v[outer]+2, bins=np.linspace(0,180,b)+2, color="steelblue",
             normed=False, histtype="step", linewidth=4, alpha=0.7, label="outer d > %.1f pix"%q)
    plt.hist(diff_angles_v[inner]-2, bins=np.linspace(0,180,b)-2, color="firebrick",
             normed=False, histtype="step", linewidth=4, alpha=0.7, label="inner d > %.1f pix"%q)
    plt.hist(diff_angles_v[diff_centroids_v>q], bins=np.linspace(0,180,b), color="k",
             normed=False, histtype="step", linewidth=4, alpha=0.7, label="all d > %.1f pix"%q)
    plt.xlabel("$\\theta$",fontsize=14)
    plt.legend(loc=9,fontsize=12)
    
def draw_angle_compare(diff_angles_v, diff_centroids_v, diff_angles_c, diff_centroids_c, crit=[0.5,0.75], b=9):
    p = crit[0]#np.percentile(diff_centroids_c, crit[0]*100)
    q = crit[1]#np.percentile(diff_centroids_c, crit[1]*100)
#     plt.hist(diff_angles_v,bins=np.linspace(0,180,b), color='orange',
#         normed=True, histtype="step", linestyle=":", linewidth=4, alpha=0.5, label="candidate all")
    plt.hist(diff_angles_c, bins=np.linspace(0,180,b), 
             color="gray",facecolor="lightgray",fill=True,
             normed=True, histtype="step", linestyle="-", linewidth=4, alpha=0.8, 
             label="Non-Candidate: %d"%len(diff_angles_c))  
    plt.hist(diff_angles_v[diff_centroids_v>p]+2, bins=np.linspace(0,180,b)+2,
             color='steelblue',hatch='-',
             normed=True, histtype="step", linewidth=4, alpha=0.8, 
             label="Candidate ($\Delta\,d$>%.1f): %d"%(p, len(diff_angles_v[diff_centroids_v>p]))) 
    plt.hist(diff_angles_v[diff_centroids_v>q]-2, bins=np.linspace(0,180,b)-2,
             color='orange',hatch='/',
             normed=True, histtype="step", linewidth=4, alpha=0.8, 
             label="Candidate ($\Delta\,d$>%.1f): %d"%(q, len(diff_angles_v[diff_centroids_v>q]-2))) 
#     plt.hist(diff_angles_c[diff_centroids_c>q], bins=np.linspace(0,180,b), color="k",
#          normed=True, histtype="step", linewidth=3, alpha=0.9, label="control > %.1f pix"%q)

    plt.xlabel("$\\theta$",fontsize=14)
    plt.legend(loc=9,fontsize=11)
    
    
def generate_catalog(save_name, Datacube, Num_v, wcs, n_rand=99):
    print("Measuring Centroids for %d candidates..."%len(Num_v))
    Datacube.centroid_analysis_all(Num_v=Num_v, centroid_type='APER',coord_type="angular", n_rand=n_rand, print_out=False)
    
    mag0 = 25.2
    mag_all = -2.5*np.log10(Datacube.Tab_SE["FLUX_AUTO"]) + mag0
    mag_cut = (18.5, 22.5)

    Datacube.construct_control(Num_v=Num_v, mag_cut=(mag_cut[0],mag_cut[1]), dist_cut=50, bootstraped=False)
    

    coords = wcs.pixel_to_world(Datacube.Tab_SE["X_IMAGE"], Datacube.Tab_SE["Y_IMAGE"])
    
    ind_v = Num_v-1
    Datacube.diff_angles_v = Datacube.diff_angles[ind_v]
    Datacube.diff_centroids_v = Datacube.diff_centroids[ind_v]
    Datacube.dist_clus_cens_v = Datacube.dist_clus_cens[ind_v]
    Datacube.PAs_v = Datacube.PAs[ind_v]
    Datacube.clus_cen_angles_v = Datacube.clus_cen_angles[ind_v]

    ind_c = Datacube.Num_c-1

    inds_best_Ha = np.argmax(Datacube.CC_Rs_Temps['Ha-NII_gauss'], axis=1)
    SNR_best_Ha = np.array([Datacube.CC_SNRs_Temps['Ha-NII_gauss'][i,j] for i,j in enumerate(inds_best_Ha)])
    R_best_Ha = np.array([Datacube.CC_Rs_Temps['Ha-NII_gauss'][i,j] for i,j in enumerate(inds_best_Ha)])
    inds_best_Hb = np.argmax(Datacube.CC_Rs_Temps['Hb-OIII_gauss'], axis=1)
    SNR_best_Hb = np.array([Datacube.CC_SNRs_Temps['Hb-OIII_gauss'][i,j] for i,j in enumerate(inds_best_Hb)])
    inds_best_OII = np.argmax(Datacube.CC_Rs_Temps['OII_gauss'], axis=1)
    SNR_best_OII = np.array([Datacube.CC_SNRs_Temps['OII_gauss'][i,j] for i,j in enumerate(inds_best_OII)])

    print("Saving Catalog...")
    df = pd.DataFrame({"ID":np.concatenate([ind_v + 1, Datacube.Num_c]),
                       "ra":np.concatenate([coords.ra[ind_v], coords.ra[ind_c]]),
                       "dec":np.concatenate([coords.dec[ind_v], coords.dec[ind_c]]),
                       "redshift":np.concatenate([Datacube.z_best[ind_v], Datacube.z_best[ind_c]]),
                       "SNR_Ha": np.concatenate([SNR_best_Ha[ind_v], SNR_best_Ha[ind_c]]),
                       "R_Ha": np.concatenate([R_best_Ha[ind_v], R_best_Ha[ind_c]]),
                       "SNR_OIII": np.concatenate([SNR_best_Hb[ind_v], SNR_best_Hb[ind_c]]),
                       "SNR_OII": np.concatenate([SNR_best_OII[ind_v], SNR_best_OII[ind_c]]),
                       "mag_auto": np.concatenate([mag_all[ind_v], mag_all[ind_c]]),
                       "R_petro": np.concatenate([Datacube.Tab_SE['PETRO_RADIUS'][ind_v], Datacube.Tab_SE['PETRO_RADIUS'][ind_c]]),
                       "dist_edge": np.concatenate([Datacube.dist_edge[ind_v], Datacube.dist_edge[ind_c]]),
                       "dist_cen": np.concatenate([Datacube.dist_clus_cens[ind_v], Datacube.dist_clus_cens[ind_c]]),
                       "diff_centroid":np.concatenate([Datacube.diff_centroids_v, Datacube.diff_centroids_c]),
                       "diff_angle":np.concatenate([Datacube.diff_angles_v, Datacube.diff_angles_c]),
                       "PA_emission":np.concatenate([Datacube.PAs_v, Datacube.PAs_c]),
                       "cluster_center_angle":np.concatenate([Datacube.clus_cen_angles_v, Datacube.clus_cen_angles_c]),
                       "type":np.concatenate([['v']*len(Datacube.diff_centroids_v), ['c']*len(Datacube.diff_centroids_c)])})

    df.to_csv(save_name,sep=',',index=None)