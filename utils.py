import os
import re

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
plt.rcParams['image.origin'] = 'lower'

from scipy import interpolate, signal, integrate, stats

import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic

import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip, sigma_clip, mad_std, gaussian_fwhm_to_sigma
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
from astropy.modeling import models, fitting
from astropy.table import Table

from astropy.visualization import LogStretch, SqrtStretch, AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
norm1, norm2, norm3 = [ImageNormalize(stretch=LogStretch()) for i in range(3)]
norm0 = ImageNormalize(stretch=AsinhStretch())

from photutils import Background2D, SExtractorBackground
from photutils import EllipticalAnnulus, EllipticalAperture, aperture_photometry
from photutils import centroid_com, centroid_2dg
from photutils import detect_sources, deblend_sources, source_properties
from photutils.utils import make_random_cmap
rand_cmap = make_random_cmap(3000, random_state=12345)
rand_cmap.set_under(color='black')

############################################
# Basic
############################################

def sigmoid(x,x0=0):
    return 1. / (1 + np.exp(-(x-x0)))

def gaussian_func(x, a, sigma):
            # Define a gaussian function with offset
            return a * np.exp(-x**2/(2*sigma**2))

def colorbar(mappable, pad=0.2, size="5%", loc="right", **args):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    if loc=="bottom":
        orent = "horizontal"
        pad = 1.5*pad
        rot = 75
    else:
        orent = "vertical"
        rot = 0
    cax = divider.append_axes(loc, size=size, pad=pad)
    cb = fig.colorbar(mappable, cax=cax, orientation=orent, **args)
    cb.ax.set_xticklabels(cb.ax.get_xticklabels(),rotation=rot)
    return cb

def vmax_3sig(img):
    # upper limit of visual imshow defined by 3 sigma above median
    return np.median(img)+3*mad_std(img)

def vmax_5sig(img):
    # upper limit of visual imshow defined by 5 sigma above median
    return np.median(img)+5*mad_std(img)

def vmin_5sig(img):
    # lower limit of visual imshow defined by 5 sigma below median
    return np.median(img)-5*mad_std(img)

def vmin_3sig(img):
    # lower limit of visual imshow defined by 5 sigma below median
    return np.median(img)-3*mad_std(img)

def printsomething():
    print("Print")

def coord_Im2Array(X_IMAGE, Y_IMAGE, origin=1):
    """ Convert image coordniate to numpy array coordinate """
    x_arr, y_arr = int(max(round(Y_IMAGE)-origin, 0)), int(max(round(X_IMAGE)-origin, 0))
    return x_arr, y_arr

def coord_Array2Im(x_arr, y_arr, origin=1):
    """ Convert image coordniate to numpy array coordinate """
    X_IMAGE, Y_IMAGE = y_arr+origin, x_arr+origin
    return X_IMAGE, Y_IMAGE

def check_save_path(dir_name):
    if not os.path.exists(dir_name):
        print("%s does not exist. Make a new directory."%dir_name)
        os.makedirs(dir_name)

        

def timer(func): 
    def execute_func(*args, **kwargs): 
  
        begin = time.time() 
        func(*args, **kwargs) 
        end = time.time() 
        print("Total time taken in : ", func.__name__, end - begin) 
  
    return execute_func



############################################
# Crosmatch & Preprocess
############################################

def query_vizier(catalog_name, RA, DEC, radius, columns, column_filters, unit=(u.hourangle, u.deg)):
    """ Crossmatch with star catalog using Vizier. """
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier
    viz_filt = Vizier(columns=columns, column_filters=column_filters)
    viz_filt.ROW_LIMIT = -1

    field_coords = SkyCoord(RA + " " + DEC , unit=unit)

    result = viz_filt.query_region(field_coords, 
                                   radius=radius, 
                                   catalog=[catalog_name])
    return result

def crossmatch_gaia2(RA, DEC, radius=6*u.arcmin, band='RPmag', mag_max=18):
    result = query_vizier(catalog_name="I/345/gaia2", 
                          RA=RA, DEC=DEC, radius=radius,
                          columns=['RAJ2000', 'DEJ2000', band],
                          column_filters={band:'{0} .. {1}'.format(5, mag_max)})    
    return result[0]
    
def crossmatch_sdss12(RA, DEC, radius=6*u.arcmin, band='rmag', mag_max=18):
    result = query_vizier(catalog_name="V/147/sdss12", 
                          RA=RA, DEC=DEC, radius=radius,
                          columns=['SDSS-ID', 'RA_ICRS', 'DE_ICRS', 'class', band],
                          column_filters={'class':'=6', band:'{0} .. {1}'.format(5, mag_max)})
    return result[0]
    
def mask_streak(file_path, threshold=5, shape_cut=0.15, area_cut=500, save_plot=True):
    """ Mask stellar streaks using astride"""
    
    from astride import Streak
    from scipy import ndimage as ndi

    streak = Streak(file_path, contour_threshold=threshold, shape_cut=shape_cut, area_cut=area_cut)
    streak.detect()
    if save_plot:
        streak.plot_figures()

    mask_streak =  np.zeros_like(streak.image)

    for s in streak.streaks:
        for (X_e,Y_e) in zip(s['x'], s['y']):
            x_e, y_e = coord_Im2Array(X_e, Y_e)
            mask_streak[x_e, y_e] = 1
            
    mask_streak = ndi.binary_fill_holes(mask_streak)        
    
    return mask_streak

def make_mask_map(image, sn_thre=3, b_size=50, npix=5):
    """ Make mask map with S/N > sn_thre """
    from photutils import detect_sources, deblend_sources
    
    # detect source
    back, back_rms = background_sub_SE(image, b_size=b_size)
    threshold = back + (sn_thre * back_rms)
    segm0 = detect_sources(image, threshold, npixels=npix)
    
    segmap = segm0.data.copy()    
    segmap[(segmap!=0)&(segm0.data==0)] = segmap.max()+1
    mask_deep = (segmap!=0)
    
    return mask_deep, segmap

def calculate_seeing(tab_star, image, seg_map, R_pix=15, sigma_guess=1, min_num=5, plot=True):
    
    """ Calculate seeing FWHM using the SE star table, image, and segmentation map """
    from scipy.optimize import curve_fit
    
    if len(tab_star)<min_num:
        print("No enough target stars. Decrease flux limit of stars.")
    else:
        FWHMs = np.array([])
        for star in tab_star:
            num = star["NUMBER"]
            X_c, Y_c = star["X_IMAGE"], star["Y_IMAGE"]
            x_c, y_c = coord_Im2Array(X_c, Y_c)
            x_min, x_max = x_c - R_pix, x_c + R_pix
            y_min, y_max = y_c - R_pix, y_c + R_pix
            X_min, Y_min = coord_Array2Im(x_min, y_min)
            cen = (X_c - X_min, Y_c - Y_min)
            
            # skip detection at edges
            if (np.min([x_min, y_min])<=0) | (np.max([x_max, y_max])>= image.shape[1]): # ignore edge
                continue
            
            # Crop image to thumb, mask nearby sources
            img_thumb = image[x_min:x_max, y_min:y_max].copy()
            seg_thumb = seg_map[x_min:x_max, y_min:y_max].copy()
            mask = (seg_thumb!=num) & (seg_thumb!=0)

            # build 2d map and turn into 1d
            yy, xx = np.indices((img_thumb.shape))
            rr = np.sqrt((xx - cen[0])**2 + (yy - cen[1])**2)
            x0, y0 = rr[~mask].ravel(), img_thumb[~mask].ravel()
            x, y = x0/rr.max(), y0/y0.max()

            # Fit the profile with gaussian and return seeing FWHM in arcsec
            initial_guess = [1, sigma_guess/rr.max()]
            popt, pcov = curve_fit(gaussian_func, x, y, p0=initial_guess)
            sig_pix = abs(popt[1])*rr.max()  # sigma of seeing in pixel
            FWHM = 2.355*sig_pix*0.322
            FWHMs = np.append(FWHMs, FWHM)  
            
            if plot:
                xplot = np.linspace(0,1,1000)
                plt.figure(figsize=(5,4))
                plt.scatter(x, y, s=5, alpha=0.5)
                plt.plot(xplot, gaussian_func(xplot,*popt),"k")
                plt.title("FWHM: %.3f"%FWHM)  # seeing FWHM in arcsec
                plt.show() 
                plt.close()
                
        return FWHMs
    
def moving_average(x, w=5):
    """ 1D moving average"""
    return np.convolve(x, np.ones(w), 'valid') / w

def moving_average_cube(cube, box=(3,3,3), mask=None):
    """ 3D moving average on cube"""
    out = convolve(cube, np.ones(box), mask=mask,
                   nan_treatment='fill', normalize_kernel=True)
    return out

def moving_average_by_col(icol, cube, w=5):
    rows = cube[:,:,icol].T
    y = np.array([moving_average(row, w=w) for row in rows])
    return y 

def background_sub_SE(field, mask=None, b_size=128, f_size=3, n_iter=10):
    """ Subtract background using SE estimator with mask """ 
    from photutils import Background2D, SExtractorBackground
    try:
        Bkg = Background2D(field, mask=mask, bkg_estimator=SExtractorBackground(),
                           box_size=(b_size, b_size), filter_size=(f_size, f_size),
                           sigma_clip=SigmaClip(sigma=3., maxiters=n_iter))
        back = Bkg.background
        back_rms = Bkg.background_rms
    except ValueError:
        img = field.copy()
        if mask is not None:
            img[mask] = np.nan
        back, back_rms = np.nanmedian(field) * np.ones_like(field), np.nanstd(field) * np.ones_like(field)
    if mask is not None:
        back *= ~mask
        back_rms *= ~mask
    return back, back_rms

def display_background_sub(field, back, vmax=1e3):
    # Display and save background subtraction result with comparison 
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(16,5))
    im1 = ax1.imshow(field, origin="lower",cmap="gray", norm=norm1, vmin=vmin_5sig(back), vmax=vmax)
    colorbar(im1)   
    im2 = ax2.imshow(back, origin='lower', cmap='gray',vmin=vmin_5sig(back), vmax=vmax_5sig(back))
    colorbar(im2)
    im3 = ax3.imshow(field - back, origin='lower', cmap='gray', norm=norm2, vmin=0., vmax=vmax)
    colorbar(im3)   
    plt.tight_layout()
    return fig


############################################
# Measurement
############################################

def get_centroid(num, table, Xname="xcentroid", Yname="ycentroid"):
    """ Get centorids of the cutout for a detection """
    id = np.where(table["NUMBER"]==num)[0][0]
    X_cen, Y_cen = table[Xname][id], table[Yname][id]
    return (X_cen, Y_cen)
    
def get_bounds(num, table, shape, cen_pos=None,
               Rname='equivalent_radius', origin=0):
    """ Get bounds of the cutout for a detection """
    id = np.where(table["NUMBER"]==num)[0][0]

    if cen_pos is None:
        cen_pos = get_centroid(num, table)

    x_cen, y_cen = coord_Im2Array(cen_pos[0], cen_pos[1], origin=origin)

    r = int(max(25, 4*table[Rname][id]))

    x_min, y_min = max(x_cen-r,0), max(y_cen-r,0)
    x_max, y_max = min(x_cen+r, shape[0]-1), min(y_cen+r, shape[1]-1)

    return (x_min, y_min, x_max, y_max)

def get_cutout(num, table, image,
               bounds=None, cen_pos=None,
               Rname='equivalent_radius',
               origin=0, **kwargs):
    """ Get cutout of the input image for a detection """
    if cen_pos is None:
        cen_pos = get_centroid(num, table)

    if bounds is None:
         bounds = get_bounds(num, table, image.shape, **kwargs)

    (x_min, y_min, x_max, y_max) = bounds

    cutout = image[x_min:x_max+1, y_min:y_max+1]

    return cutout

class Obj_detection:
    def __init__(self, tab, from_SE=False, cube=None,
                 seg_map=None, deep_frame=None, mask_edge=None):
        num = tab["NUMBER"]
        if from_SE==True:
            self.X_c, self.Y_c = tab["X_IMAGE"], tab["Y_IMAGE"]
            self.x_c, self.y_c = coord_ImtoArray(self.X_c, self.Y_c)
            self.width = np.max([tab["A_IMAGE"],3])
            self.a_image, self.b_image = tab["A_IMAGE"], tab["B_IMAGE"]
            self.theta = tab["THETA_IMAGE"]

            self.R_petro = tab["PETRO_RADIUS"]
            self.flux_auto = tab["FLUX_AUTO"]

            x_min, x_max = (np.max([int(self.x_c - 6*self.width), 0]),
                            np.min([int(self.x_c + 6*self.width), cube.shape[1]]))
            y_min, y_max = (np.max([int(self.y_c - 6*self.width), 0]),
                            np.min([int(self.y_c + 6*self.width), cube.shape[2]]))
            self.origin = 1
            
        else:
            self.X_c, self.Y_c = tab["xcentroid"], tab["ycentroid"]
#             self.x_c, self.y_c = coord_ImtoArray(self.X_c, self.Y_c, 0)
            self.a_image = tab['equivalent_radius']
            self.b_image = self.a_image * (1-tab['ellipticity'])
            self.theta = tab['orientation']
    
            (x_min, y_min, x_max, y_max) = get_bounds(num, Table(tab), cube.shape[1:],
                                                      cen_pos=(self.X_c, self.Y_c))
            self.origin = 0
        
        self.bounds = (x_min, y_min, x_max, y_max)
        
        self.X_min, self.Y_min = coord_Array2Im(x_min, y_min, origin=self.origin)
        self.center_pos=(self.X_c-self.X_min, self.Y_c-self.Y_min)

        self.mask_thumb = mask_edge[x_min:(x_max+1), y_min:(y_max+1)]

        self.seg_thumb = seg_map[x_min:(x_max+1), y_min:(y_max+1)]
        self.mask_seg = (self.seg_thumb==num) | (self.seg_thumb==0)
        self.mask_seg_obj = (self.seg_thumb==num)

        self.cube_thumb = cube[:,x_min:(x_max+1), y_min:(y_max+1)]
        self.img_thumb = self.cube_thumb.sum(axis=0)
        
        if deep_frame is not None:
            self.deep_thumb = deep_frame[x_min:(x_max+1), y_min:(y_max+1)]
        else:
            self.deep_thumb = None
                
                
    def img_display(self, vmin=0., vmax=None, norm=norm0):
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
     
    def aper_photometry(self, k, k1, k2, ext_type="opt", verbose=False):
        apertures = EllipticalAperture(positions=self.center_pos, 
                                       a=k*self.a_image, b=k*self.b_image, theta=self.theta)
        
        annulus_apertures = EllipticalAnnulus(positions=self.center_pos, 
                                              a_in=k1*self.a_image, a_out=k2*self.a_image, 
                                              b_out=k2*self.b_image, theta=self.theta)

        apers = [apertures, annulus_apertures]
        m0 = apers[0].to_mask(method='exact')[0]
        area_aper = m0.to_image(self.img_thumb.shape)
        area_aper[self.mask_thumb] = 0
        m1 = apers[1].to_mask(method='exact')[0]
        area_annu = m1.to_image(self.img_thumb.shape)
        area_annu[self.mask_thumb] = 0

        bkg = self.img_thumb[np.logical_and(area_annu>=0.5,self.mask_seg)]
        signal = self.img_thumb[np.logical_and(area_aper>=0.5,self.mask_seg)]
        n_pix = len(signal)
        
        if (len(signal)==0)|(len(bkg)==0):
            if verbose: print("No signal/background!")
            return np.nan, np.nan, None
        if n_pix<=5: 
            if verbose: print("Too few pixels!")
            return np.nan, np.nan, None
        
#         S =  np.max([np.sum(signal) - n_pix * np.median(bkg), 0])
        S =  np.max([np.sum(signal) - n_pix * mad_std(bkg), 0])
        N_sky = n_pix/(n_pix-1) * np.sqrt(n_pix*mad_std(bkg)) 
        N_tot = np.sqrt(S + N_sky**2)
        
        if ext_type == "sky": N = N_sky
        else: N = N_tot
            
        if verbose: print ("Aperture: %.1f Rp; Signal: %.4f, Noise: %.4f, SNR = %.4f"%(k, S, N, S/N))
        return S, N, apers 
                   
    def compute_aper_opt(self, ks = np.arange(1.0,4.5,0.2), k1 = 5., k2 = 8., ext_type="opt",
                         verbose=True, plot=False):
        if self.R_petro==0:
            if verbose: print("Error in measuring R_petro, dubious source")
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
        
        if verbose: print ("Optimal Aperture: %.1f Rp, SNR = %.4f"%(k_opt, snr_opt))
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
        return spec  
    

def fit_continuum(spec, wavl, model='GP', 
                  kernel_scale=100, kenel_noise=1e-3, edge_ratio=0.15,
                  verbose=True, plot=True, fig=None, ax=None):
    """
    Fit continuum of the spectrum with Trapezoid1D model or Gaussian Process

    Parameters
    ----------
    spec : extracted spectrum
    model : continuum model used to subtract from spectrum
            "GP": flexible non-parametric continuum fitting
                  edge_ratio : edge threshold of the filter band (above which the
                               edge is replaced with a median to avoid egde issue).
                  kernel_scale : kernel scale of the RBF kernel
                  kenel_noise : noise level of the white kernel
            "Trapz1d": 1D Trapezoid model with constant continuum and edge fitted
                       by a constant slope

    Returns
    ----------
    res : residual spectra
    wavl_rebin : wavelength rebinned in log linear
    cont_fit : fitted continuum
        
    """
    
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
        if verbose:
            print(model_fit)
            
    elif model=='GP':
        # Fit continuum with Gaussian Process
        if np.sum((y_intp - np.median(y_intp) <= -2.5*mad_std(y_intp)))> edge_ratio*n_wavl:  
            # For some special source with sharp edge
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
        
    res = (y_intp - cont_fit)
    
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
                      temp_type="Ha-NII", temp_model="gauss",
                      temp_params={'box_width':4, 'line_ratio':[1,1,1],
                                   'sigma':3, 'a_sinc':5},
                      plot=True, alpha=0.7):
    """
    Generate a template for with line ratio [NII6548, Ha, NII6584]/[OIII4959, OIII5007]
    # For Gaussian models sigma (i.e. stddev) = sigma
    # For delta function width = box_wid
    """
    wavl_range = wavl[-1] - wavl[0]
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
        
    wavl_temp = np.e**np.linspace(np.log(lam_0*(1+z)-wavl_range/2.),
                                  np.log(lam_0*(1+z)+wavl_range/2.), n_temp)
    
    line_ratio = temp_params["line_ratio"]

    if temp_model == "box":
        box_wid = temp_params["box_width"]
        s = np.sum([models.Box1D(amplitude=lr, x_0=lp, width=box_wid) for (lr, lp) in zip(line_ratio, line_pos)])
        temp = s(wavl_temp)
        
    elif (temp_model == "gauss")|(temp_model == "sincgauss"):
        sigma = temp_params["sigma"]
        s = np.sum([models.Gaussian1D(amplitude=lr, mean=lp, stddev=sigma) for (lr, lp) in zip(line_ratio, line_pos)])
        temp = s(wavl_temp)
        
        if temp_model == "sincgauss":
            a_sinc = temp_params["a_sinc"]
            ILS = np.sinc((wavl_temp-lam_0*(1+z))/a_sinc)
            temp = np.convolve(temp, ILS, mode="same")
            temp[(wavl_temp<lam_0*(1+z)-25)|(wavl_temp>lam_0*(1+z)+25)]=0
            
    else: print('Model type: "box" or "gauss" or "sincgauss"')
        
    temp /= np.max([1, temp.max()])
    
    if plot:
        plt.plot(wavl_temp, temp,"g", alpha=alpha)
        for lp in line_pos:
            plt.axvline(lp, color="k", ls="-.", alpha=alpha) 
        
    return temp, wavl_temp  

def use_broad_window(sigma, line_ratio, SNR,
                     sigma_thre=5, sn_thre=20,
                     temp_type="Ha-NII",
                     temp_model="gauss"):
    
    strong_NII = (temp_type=="Ha-NII")&(line_ratio.max()/3 < 3)
    broad_line = (sigma>sigma_thre)|(temp_model=="box")
    high_SN = (SNR>=sn_thre)
    
    if strong_NII & broad_line & high_SN:
        return True
    else:
        return False
    
def xcor_SNR(res, wavl_rebin, 
             temps, wavl_temp, 
             d_wavl, z_sys=0.228, rv=None, 
             edge=20, h_contrast=0.1,
             n_intp=1, kind_intp="linear",
             temp_type="Ha-NII", temp_model="gauss",
             temps_params={'stddev':3,
                           'line_ratio':[1,1,1]},
             const_window=False, plot=False,
             fig=None, axes=None):
    """
    Cross-correlate with template and return rative velocity to z0 and correlation function
    
    res : normalized input spectra
    wavl_rebin : rebinned wavelength
    rv : relative redshift and velocity to z=z0, None if not known (the first object in batch)
    """ 
    
    temps_stddev = temps_params['stddev']
    temps_ratio = temps_params['line_ratio']
    
    # Interpolate data before cross-correlation to match template with the same sampling
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
    else:
        print('Available template types are: "Ha-NII", "Hb-OIII" and "OII"')
        return None
      
    z0 = 6563.*(1+z_sys)/lam_0 - 1  # reference redshift, the matched line by default is Ha
    
    # Compute relative redshift and velocity to z=z0, if rv not given 
    if rv is None:
        cc = signal.correlate(y, temps[0], mode="full", method="direct")    
        
        rz = np.linspace((x[0] - d_wavl/n_intp * temps.shape[1]/2.) / lam_0 - 1, 
                         (x[-1] + d_wavl/n_intp * temps.shape[1]/2.) / lam_0 - 1, len(cc)) - z0    

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
    
    # min/max rv based on the wavelength range
    rv_edge = ((np.max([x[x!=0][0], x[0]+edge])/lam_0-1-z0) * 3e5,   # x=0 is artificially set during the continuum fitting
               (np.min([x[x!=0][-1], x[-1]-edge])/lam_0-1-z0) * 3e5) 
    
    # possible range of rv given possible range of redshift 
    rv_zrange = (rv>(zmin-z0)*3e5) & (rv<(zmax-z0)*3e5)
    find_range = rv_zrange & (rv > rv_edge[0]) & (rv < rv_edge[1])
    
    for i, (temp, sigma, ratio) in enumerate(zip(temps, temps_stddev, temps_ratio)):
        # vicinity around lines to be excluded when measuring noise RMS
#         if temp_model=="sincgauss":
#             sigma *= 3
        if (const_window)|(temp_model=="box"):
            w_l, w_l2 = 15, 25
        else:
            w_l, w_l2 = sigma * 3, sigma * 5
        
        #For strong AGN/composite use a broader window
        broad = use_broad_window(sigma, ratio, SNRs.max(),
                                 sigma_thre=5, sn_thre=25,
                                 temp_type=temp_type,
                                 temp_model=temp_model)
        if broad:
            w_l, w_l2 = w_l*2, w_l2*2
    
        cc = signal.correlate(y, temp, mode="full", method="direct")
        cc = cc/cc.max()
        ccs[i] = cc
        
        # Peak redshift (use cubic interpolation around the peak)
        # note: difference in z with a gaussian fitting to the peak is very small
        
        rv_p = rv[find_range][np.argmax(cc[find_range])]  # rv peak
        Interp_Peak = interpolate.interp1d(rv, cc, kind='cubic')
       
        # rv_intp = np.linspace(np.max([rv_edge[0],rv_p - 3*w_l/lam_0*3e5]), np.min([rv_edge[1],rv_p + 3*w_l/lam_0*3e5]), 300)
        rv_intp = np.linspace(np.max([rv_edge[0],rv_p - 1600]), 
                              np.min([rv_edge[1],rv_p + 1600]), 200)
        cc_intp = Interp_Peak(rv_intp)
        rv_match = rv_intp[np.argmax(cc_intp)]
        rz_match = rv_intp[np.argmax(cc_intp)] / 3e5
        
        z_cc = z0 + rz_match
        z_ccs[i] = z_cc
        
        S_p = cc_intp.max()  # peak of cross-correlation function 
        
        peak_range = (abs(rv-rv_match)/3e5 < w_l/lam_0)
        
        # note: when compute noise of CC function, edge not in use 
        if (temp_type=="Ha-NII") | (temp_type=="Hb-OIII"):
            
            noise_peak_range = (~peak_range) & (rv > rv_edge[0]) & (rv < rv_edge[1])
            
            if temp_type=="Ha-NII":
                signal_range = ((rv > (rv_match - (w_l2+lam_0-6548.)/6548.*3e5)) &\
                                (rv < (rv_match + (w_l2+6584.-lam_0)/6584.*3e5)))
                
            elif temp_type=="Hb-OIII":
                signal_range = (abs(rz-rz_match) < w_l/lam_0) | (abs(rz-4959.*(1+z_cc)/5007.-z0-1)<w_l/lam_0)
                
            noise_range = (~signal_range) & (rv > rv_edge[0]) & (rv < rv_edge[1])
                
        elif temp_type=="OII":
            signal_range = peak_range
            noise_range = (~signal_range) & (rv > rv_edge[0]) & (rv < rv_edge[1])
            noise_peak_range = noise_range

        # compute noise and S/N for the lines and the peak
#         N = np.std(sigma_clip(cc[rv_zrange & noise_range], sigma=5, maxiters=10))
#         N = mad_std(cc[rv_zrange & noise_range])
        N = np.std(cc[rv_zrange & noise_range])           # simply use std?
        
        # detection S/N
        SNRs[i] = S_p/N  
        
#         N_p = np.std(sigma_clip(cc[rv_zrange & noise_peak_range], sigma=5, maxiters=10))
#         N_p = mad_std(cc[rv_zrange & noise_peak_range])
        N_p = np.std(cc[rv_zrange & noise_peak_range])    # simply use std?
        SNR_ps[i] = S_p/N_p
        
    
        ###-----------------------------------###
        # Find peaks and compute peak significance
        if temp_type=="Ha-NII":
            try:
                # find peak higher than h_contrast
                ind_peak, _ = signal.find_peaks(cc_intp, height=h_contrast, distance=50)
                
                # peak value from higher to lower
                peaks = np.sort(cc_intp[ind_peak])[::-1]

                if len(peaks)>=2:
                    Contrasts[i] = peaks[0]/peaks[1]
                
                else:     #single peak
                    Contrasts[i] = 1.0 #peaks[0]*(ratio.max()/3.)  # upper bond
                    
            except ValueError:
                Contrasts[i] = -0.1
        
        elif temp_type=="Hb-OIII":
            # For double line template, the second peak is in another rv range
            rv_intp = np.linspace(np.max([rv_edge[0],rv_p - 1000]),
                                  np.min([rv_edge[1],rv_p + 1000]), 100)
            
            rv_intp_2 = np.linspace(np.max([rv_edge[0],(4959.*(1+z_cc)/5007.-z0-1)*3e5 - 1000]), 
                                    np.min([rv_edge[1],(4959.*(1+z_cc)/5007.-z0-1)*3e5 + 1000]), 100)
            rv_intp_all = np.concatenate([rv_intp_2,rv_intp])
            
            cc_intp_all = Interp_Peak(rv_intp_all)
            ind_peak, _ = signal.find_peaks(cc_intp_all, height=h_contrast, distance=25)
            peaks = np.sort(cc_intp_all[ind_peak])[::-1]
            
            if len(peaks)>=2:
                Contrasts[i] = peaks[0]/peaks[1]
            else:
                Contrasts[i] = 1.0
            
        elif temp_type=="OII":
            Contrasts[i] = 1.0
            
    Rs = Contrasts*SNRs/np.max(SNRs)     #Significance defined as S/N weight peak ratio

    # The best matched template
#     if temp_model == "box":
#         best = np.argmax(Contrasts)
#     else:
    best = np.argmax(Rs)
    
    ###-----------------------------------##
    z_best = z_ccs[best]
    snr_best = SNRs[best]
    sig_best = temps_stddev[best]
    ratio_best = temps_ratio[best]
        
    #  if the line is within 25A to the edge, raise a flag caution
    line_at_edge = ((1+z_best)*lam_0<(x[0]+25)) | ((1+z_best)*lam_0>(x[-1]-25))   
    if line_at_edge: 
        flag_edge = 1
    else: 
        flag_edge = 0

    if plot:
        if axes is None:
            fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
        else:
            ax1, ax2 = axes
            
        ax1.step(wavl_rebin, res, c="royalblue", lw=2, where="mid", label="Residual", alpha=0.9, zorder=4)
#         ax1.plot(x, y, c="steelblue", linestyle='--', alpha=0.9, zorder=4)

        #Plot ideal model template
        sig_best, ratio_best = temps_stddev[best], temps_ratio[best]
        line_pos = line_pos*(1+z_best)

        if temp_model == "box":
            # if use box template, sigma are the same (box_wid/2.355)
            s = np.sum([models.Box1D(amplitude=lr, x_0=lp, width=sig_best*2.355)
                        for (lr, lp) in zip(ratio_best, line_pos)])
        else:
            s = np.sum([models.Gaussian1D(amplitude=lr, mean=lp, stddev=sig_best)
                        for (lr, lp) in zip(ratio_best, line_pos)])

        wavl_new = np.linspace(wavl_temp[0], wavl_temp[-1], 400)/(1+z0)*(1+z_best)
        temp_best = s(wavl_new)

        if (temp_model == "sincgauss"):
            ILS = np.sinc((wavl_new-lam_0*(1+z_best))/5.)
            temp_best = np.convolve(temp_best, ILS, mode="same")
            temp_best[(wavl_new<lam_0*(1+z_best)-50)|(wavl_new>lam_0*(1+z_best)+50)]=0  # let higher-order lobe to be 0

#             ax1.step(wavl_temp/(1+z0)*(1+z_best), temps[best], color="g", where="mid",
#                      alpha=0.5, label="Template: z=%.3g"%z_best, zorder=3)  
        ax1.step(wavl_new, temp_best/temp_best.max(), color="mediumseagreen",
                 where='mid', lw=2, linestyle='--', alpha=0.7, zorder=5)    

        if (const_window)|(temp_model=="box"):
            w_l, w_l2 = 15, 25
        else:    
            w_l, w_l2 = sig_best * 3., sig_best * 5.

        broad = use_broad_window(sig_best, ratio_best, snr_best,
                                 sigma_thre=5, sn_thre=30,
                                 temp_type=temp_type,temp_model=temp_model)    
        if broad:
            w_l, w_l2 = w_l*2, w_l2*2
                           
        for k, (lp, ln) in enumerate(zip(line_pos, line_name)):
            if ln!=" ":
                if (lp<x[0]+edge-5) | (lp>x[-1]-edge+5):
                    continue
                y_lab = res.max() if (k==1) else 0.875 * res.max()
                ax1.vlines(x=lp, ymin=y_lab+0.1, ymax=y_lab+0.18, color='k', lw=1.5)
                ax1.text(x=lp, y=y_lab+0.3, s=ln, color='k', ha="center", va="center", fontsize=14)
                ax1.set_ylim(res.min()-0.05, res.max()+0.45)
            else:
                ax1.set_ylim(res.min()-0.05, res.max()+0.25)
                
        ax1.set_xlim(x[0]-10, x[-1]+10)
        
        x_text = 0.75 if z_best < 0.24 else 0.12
        ax1.text(x_text, 0.8, "z = %.3g"%(z_best),
                 color='k',fontsize=14, transform=ax1.transAxes,alpha=0.95, zorder=4)
        ax1.set_xlabel('Wavelength ($\AA$)',fontsize=14)
        ax1.set_ylabel('Normed Flux',fontsize=14)
        
        for i, temp in enumerate(temps):    
            ax2.plot(rv, ccs[i], "firebrick", alpha=np.min([0.9, 0.9/len(temps)*3]))
        ax2.plot(rv, ccs[best], "red", lw=1, alpha=0.7, zorder=4)
        
        rv_left, rv_right = ((z_ccs[best]-z0 - w_l/lam_0)*3e5, 
                             (z_ccs[best]-z0 + w_l/lam_0)*3e5)  
        
        if temp_type=="Ha-NII":
            rv_left2, rv_right2 = (np.max([rv_edge[0],(z_ccs[best]-z0 - (w_l2+lam_0-6548.)/6548.) * 3e5]), 
                                   np.min([rv_edge[1],(z_ccs[best]-z0 + (w_l2+6584.-lam_0)/6584.) * 3e5]))
            
        elif temp_type=="Hb-OIII":
            rv_left2, rv_right2 = rv_left, rv_right
            rv_left2b, rv_right2b = (np.max([rv_edge[0],((4959.*(1+z_ccs[best])/5007.-z0-1) - w_l/lam_0) * 3e5]),
                                     np.min([rv_edge[1],((4959.*(1+z_ccs[best])/5007.-z0-1) + w_l/lam_0) * 3e5]))
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
                rv_intp = np.linspace(rv_left2, rv_right2, 200)
                d_min = 25
            elif temp_type=="Hb-OIII":
                rv_intp = np.concatenate([np.linspace(rv_left2b, rv_right2b, 100),
                                          np.linspace(rv_left2, rv_right2, 200)]) 
                d_min = 25
            elif temp_type=="OII":
                rv_intp = np.linspace(rv_left2, rv_right2, 100)
                d_min = 50
                
            Interp_Peak = interpolate.interp1d(rv, ccs[best], kind='cubic')
            cc_intp = Interp_Peak(rv_intp)

            ind_peak, _ = signal.find_peaks(cc_intp, height=h_contrast, distance = d_min)
            ax2.plot(rv_intp[ind_peak], cc_intp[ind_peak], "x", ms=5, color='orange',zorder=4) 
            
        except ValueError:
            pass
        
        ##----##
        
        ax2.axhline(0, color="k", lw=1.5, ls="--", alpha=0.7)
        ax2.axvline((z_ccs[best]-z0)*3e5, color="orangered", lw=1.5, ls="--", alpha=0.8)
                           
        ax2.text(x_text, 0.82, r"$\rm S/N: %.1f$"%(SNRs[best]),
                 color='k',fontsize=13, transform=ax2.transAxes, alpha=0.95, zorder=4)
        if temp_type!="OII":
            ax2.text(x_text, 0.7, r"$\rm R: %.1f$"%(Rs[best]),
                     color='k',fontsize=13, transform=ax2.transAxes, alpha=0.95, zorder=4)
        ax2.set_xlim((zmin-z0)*3e5, (zmax-z0)*3e5)
        ax2.set_xlabel('Relative Velocity (km/s) to z=%.3g'%z0,fontsize=14)
        ax2.set_ylabel('CCF',fontsize=14)
        axb = ax2.twiny()
        axb.plot(rv/3e5 + z0, np.ones_like(rv)) # Dummy plot
        axb.cla()
        axb.set_xlim(zmin,zmax)
        axb.set_xlabel("Redshift",fontsize=14)      
    
    result_cc = {}
    for (prop, vals) in zip(["ccf", "rv", "z_best", "sigma_best", 
                            "R", "contrast", "SNR", "SNR_p", "flag_e"],
                           [ccs[best], rv, z_best, sig_best, Rs[best], 
                            Contrasts[best], SNRs[best], SNR_ps[best], flag_edge]):
#                            [ccs, rv, z_ccs, Rs, Contrasts, SNRs, SNR_ps, flag_edge]):
        result_cc[prop] = vals
        
    return result_cc


def measure_dist_to_edge(table, mask_edge, pad=200,
                         Xname="xcentroid", Yname="ycentroid"):
    """ Measure distance of each detection to the edges of the field.
        This is to reduce some spurious detection around edges."""
    
    shape = mask_edge.shape
    # Note photutils is 0-based while DS9 and SExtractor is 1-based
    yy, xx = np.mgrid[0:shape[0],0:shape[1]]
    
    dist_edge = np.empty(len(table))
    for i, obj in enumerate(table):
        x_obj, y_obj = (obj[Xname], obj[Yname])
        if (x_obj<pad) | (x_obj>(shape[1]-pad)) | (y_obj<pad) | (y_obj>(shape[0]-pad)):
            dist_edge[i] = math.sqrt(np.min((xx[mask_edge]-x_obj)**2 + (yy[mask_edge]-y_obj)**2))
        else:
            dist_edge[i] = pad
            
    return dist_edge
    
def estimate_EW(spec, wavl, z, 
                lam_0=6563., sigma=5, edge=20,
                MC_err=False, n_MC=250, spec_err=0.15, 
                cont=None, ax=None, plot=True):
    """ Estimate Equivalent Width of the line(s) """
    
    # window range to estimate EW. Filter edges excluded.
    not_at_edge = (wavl>wavl.min()+edge) & (wavl<wavl.max()-edge)
    line_range = (wavl > lam_0*(1+z)-8*sigma) & (wavl < lam_0*(1+z)+8*sigma) & not_at_edge
    
    # estimate conitinum if not given
    if cont is None:
        cont = np.median(spec[(~line_range) & not_at_edge])
    
    # EW = intg{F_l / F_0 - 1} (EW>0: emission)
    x = wavl[line_range]
    y = spec[line_range]
    EW = np.trapz(y=y/cont-1, x=x)
    
    if MC_err is True:
        if np.ndim(spec_err)==0:
            y_err = spec_err*(abs(y-cont))
        else:
            y_err = spec_err[line_range]
        EW_err = [np.trapz(y=(y+np.random.normal(0, scale=y_err))/cont-1, x=x) for i in range(n_MC)]
        EW_std = np.std(EW_err)
    else:
        EW_std = np.nan
    
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,4))
        ax.step(wavl, spec, where="mid", color="k",alpha=0.9)
        ax.step(wavl[line_range], spec[line_range], where="mid")
        ax.axhline(cont,color="seagreen",alpha=0.8)
        if MC_err is True:
            ax.errorbar(x, y, y_err, fmt=".", lw=1, color="gray", capsize=2, alpha=0.5)
        x_text = 0.7 if z < 0.24 else 0.1
        ax.text(x_text, 0.8, r"$\rm EW=%.2f\pm%.2f\AA$"%(EW, EW_std),
                     color='k', fontsize=13, transform=ax.transAxes, alpha=0.95)
        ax.set_xlabel('Wavelength ($\AA$)',fontsize=14)
        ax.set_ylabel(r'Flux ($\rm 10^{-17}\,erg/cm^2/s/\AA$)',fontsize=13) 
        
    return EW, EW_std

    
def save_ds9_region(name, obj_pos, size, color="green", origin=1):

    import pyregion
    
    if size is None: size=10
    
    ds9_header = "# Region file format: DS9 version 4.1\nglobal color=%s dashlist=8 3 width=1\nimage\n"%color
    reg = "\n".join(["circle(%.4f,%.4f,%.3f)\n"\
                            %(pos[0]+origin, pos[1]+origin, size) for pos in obj_pos])
    region_str = ds9_header + reg
    ds9_regs = pyregion.parse(region_str)
    ds9_regs.write(os.path.join(save_path,'%s.reg'%name))    
    
def measure_offset_aper(obj, img_em, img_con, mask_obj,
                        k_aper=5, multi_aper=False,
                        aperture_type="separate",
                        aper_size=[0.7,0.8,0.9,1.1,1.2,1.],
                        n_rand_start=199, niter_aper=15, ctol_aper=0.01,
                        verbose=True):
    if verbose:
            print("optimal aperture:%.1f"%k_aper)
            
    if aperture_type== "separate":
        theta = obj.theta * np.pi/180
        
        if multi_aper is False:
            aper_size = [1]
            n_rand_start = 0
            
        x1s = np.zeros((n_rand_start+1, len(aper_size)))
        y1s, x2s, y2s = np.zeros_like(x1s), np.zeros_like(x1s), np.zeros_like(x1s)
        
        aper_ems, aper_cons  = np.array([]), np.array([])
        
        if verbose:
            print("Aper (min,max): (%.2f,%.2f)"%(k_aper*aper_size[0],k_aper*aper_size[-1]))
        
        for k, percent in enumerate(aper_size):
            a = np.max([obj.a_image*percent,1.])
            b = a/obj.a_image * obj.b_image
            da_s = np.append(stats.norm(loc=0, scale=a/3.).rvs(n_rand), 0)
            db_s = np.append(stats.norm(loc=0, scale=b/3.).rvs(n_rand), 0)

            for j,(da,db) in enumerate(zip(da_s, db_s)):
                try:
                    for d,img in enumerate([img_em, img_con]):
                        dx, dy = da*np.cos(theta)-db*np.sin(theta), da*np.sin(theta)+db*np.cos(theta)
                        x, y = obj.center_pos[0] + dx, obj.center_pos[1] + dy

                        for i in range(niter_aper):
                            aper = EllipticalAperture(positions=(x,y), 
                                                      a=k_aper*a, b=k_aper*b, theta=obj.theta)              
                            m0 = aper.to_mask(method='exact')[0]
                            mask_aper = m0.to_image(mask_obj.shape)

                            data = img * mask_aper * mask_obj
                            if np.sum(data) < 0:
                                x1s[j,k], y1s[j,k] = np.nan, np.nan
                                x2s[j,k], y2s[j,k] = np.nan, np.nan
                                aper_em, aper_con = None, None                              
                                break
                            x_new, y_new = centroid_com(data, mask=np.logical_not(mask_aper * mask_obj))

                            reach_ctol = (np.sqrt((x-x_new)**2+(y-y_new)**2) < ctol_aper)   
                            if  reach_ctol:  
                                break
                            else:
                                x, y = x_new, y_new
                                
                        if d==0:
                            data_em = img_em * mask_aper * mask_obj
                            x1, y1 = x, y
                            x1s[j,k], y1s[j,k] = x,y
                            aper_em = EllipticalAperture(positions=(x1,y1), 
                                                      a=k_aper*a, b=k_aper*b, theta=obj.theta)

                        elif d==1:
                            data_con = img_con * mask_aper * mask_obj
                            x2, y2 = x, y
                            x2s[j,k], y2s[j,k] = x, y
                            aper_con = EllipticalAperture(positions=(x2,y2), 
                                                      a=k_aper*a, b=k_aper*b, theta=obj.theta)
                            
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

            # if centroid diverge, set the centroid to be nan
            if (np.nanstd(r1_k)>.1): use_value_1[k] = False
            if (np.nanstd(r2_k)>.1): use_value_2[k] = False

        if (np.sum(use_value_1)==0)|(np.sum(use_value_2)==0):
            return (0, 0 , 0)

        x1_eff, y1_eff = x1s[-1][use_value_1], y1s[-1][use_value_1]
        x2_eff, y2_eff = x2s[-1][use_value_2], y2s[-1][use_value_2]
        sigma_1_eff, sigma_2_eff = sigma_1s[use_value_1], sigma_2s[use_value_2]

        aper_ems_eff, aper_cons_eff = aper_ems[use_value_1], aper_cons[use_value_2]
        
        result_cen = {"x1_eff":x1_eff, "y1_eff":y1_eff, "x2_eff":x2_eff, "y2_eff":y2_eff,
                      "aper_ems_eff":aper_ems_eff, "aper_cons_eff":aper_cons_eff}
        
    elif aperture_type== "single": #single aperture
        img = obj.img_thumb
        x, y = obj.center_pos
        for i in range(niter_aper):
            aper = EllipticalAperture(positions=(x,y), 
                                      a=k_aper*obj.a_image, b=k_aper*obj.b_image, theta=obj.theta)
            m0 = aper.to_mask(method='exact')[0]
            mask_aper = m0.to_image(img.shape)           
            data = img * mask_aper * mask_obj
            x_new, y_new = centroid_com(data, mask=np.logical_not(mask_aper * mask_obj))
            reach_ctol = (np.sqrt((x-x_new)**2+(y-y_new)**2) < ctol_aper)
            if  reach_ctol: 
                # if reach n_iter, or the bound of optimal aperture, stop
                break
            else:
                x, y = x_new, y_new
                
        data_em = img_em * mask_aper * mask_obj
        x1_eff, y1_eff = centroid_com(data_em, mask=np.logical_not(mask_aper * mask_obj))
        data_con = img_con * mask_aper * mask_obj
        x2_eff, y2_eff = centroid_com(data_con, mask=np.logical_not(mask_aper * mask_obj))
        
        aper_em, aper_con = aper, aper
        result_cen = {"x1":x1, "y1":y1, "x2":x2, "y2":y2, "aper_em":aper_em, "aper_con":aper_con}
        
    return result_cen
    
def compute_centroid_offset(obj, spec, wavl, z_cc, wcs,
                            coord_BCG=None,
                            iso_snr=1.2, 
                            edge=20, n_rand=199,
                            centroid_type="APER",
                            subtract=True, sum_type="weighted", 
                            k_aper=5, multi_aper=False,
                            aper_size=[0.7,0.8,0.9,1.1,1.2,1.],
                            aperture_type="separate",
                            niter_aper=15, ctol_aper=0.01,
                            line_stddev=3, line_ratio=None, mag0=25.2,
                            affil_map=None, deep_img=None,
                            plot=True, verbose=True,
                            return_for_plot=False, **kwargs):    
    """ Compute the centroid of emission and continua, and plot """
    
    # Window width of lines
    w_l, w_l2 = line_stddev * 3, line_stddev * 5
    
#     if (line_ratio<3) & (line_stddev>5):
    em_range = ((6548.-w_l)*(1+z_cc), (6584.+w_l)*(1+z_cc))
#     else: 
#         em_range = ((6563.-w_l)*(1+z_cc), (6563.+w_l)*(1+z_cc))
        
    # Emission channels
    em = (wavl > max(em_range[0], wavl[0]+edge)) & (wavl < min(em_range[1], wavl[-1]-edge))
    
    # Continuum channels
    con = (~em) & (wavl > wavl[0]+edge) & (wavl < wavl[-1]-edge) \
                & ((wavl > (6584.+w_l2)*(1+z_cc)) | (wavl < (6548.-w_l2)*(1+z_cc)))
    
    # Make emission image and continuum image from datacube
    if sum_type=="mean":
        img_em = np.mean(obj.cube_thumb[em,:,:],axis=0)
        img_con = np.mean(obj.cube_thumb[con,:,:], axis=0)
        
    elif sum_type=="median":    
        img_con = np.median(obj.cube_thumb[con,:,:], axis=0)
        img_em = np.mean(obj.cube_thumb[em,:,:],axis=0)
        
    elif sum_type=="weighted":    
        # Emission image is the sum of emission channels weigthed by flux density.
        img_con = np.median(obj.cube_thumb[con,:,:], axis=0)
        img_em0 = obj.cube_thumb[em,:,:]
        
        weight_map = abs(img_em0-img_con)
        weight_sum = np.sum(weight_map, axis=0)
        
        img_em = np.sum(img_em0*weight_map/weight_sum, axis=0)
    
    if subtract is True:
        img_em = img_em - img_con
    
    mask_obj = obj.mask_seg
    
    if centroid_type == "APER":
        res_cen = measure_offset_aper(obj, img_em, img_con, mask_obj, **kwargs)
        
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
        label_em = np.sqrt((tbl_em["xcentroid"].value-obj.img_thumb.shape[1]/2.)**2+(tbl_em["ycentroid"].value-obj.img_thumb.shape[0]/2.)**2).argmin()
        label_con = np.sqrt((tbl_con["xcentroid"].value-obj.img_thumb.shape[1]/2.)**2+(tbl_con["ycentroid"].value-obj.img_thumb.shape[0]/2.)**2).argmin()
        x1, y1 = tbl_em["xcentroid"][label_em].value, tbl_em["ycentroid"][label_em].value
        x2, y2 = tbl_con["xcentroid"][label_con].value, tbl_con["ycentroid"][label_con].value
        
        data_em  = img_em*(segm_deblend_em.data==(1+label_em))
        data_con  = img_con*(segm_deblend_con.data==(1+label_con))
        
    elif centroid_type == 'ISO2':
        sigma = 3.0 * gaussian_fwhm_to_sigma    # FWHM = 3.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        kernel.normalize()
        
        threshold = detect_threshold(obj.img_thumb, snr=iso_snr)
        segm = detect_sources(obj.img_thumb, threshold, npixels=5, filter_kernel=kernel)
        segm_deblend = deblend_sources(obj.img_thumb, segm, npixels=5, filter_kernel=kernel, nlevels=32,contrast=0.001)
        
        tbl = source_properties(obj.img_thumb, segm_deblend).to_table()
        label = np.sqrt((tbl["xcentroid"].value-obj.img_thumb.shape[1]/2.)**2+(tbl["ycentroid"].value-obj.img_thumb.shape[0]/2.)**2).argmin()
        
        data_em = img_em * (segm_deblend.data==(1+label))
        data_con = img_con * (segm_deblend.data==(1+label))
        x1, y1 = centroid_com(data_em)
        x2, y2 = centroid_com(data_con)

    ###-----------------------------------###
            
    if np.ndim(coord_BCG)==0:
#             (ra_BCG, dec_BCG) = coord_BCG.ra, coord_BCG.dec
        c_BCG = coord_BCG

    else:
        lab = affil_map[obj.x_c.astype("int64"), obj.y_c.astype("int64")]
#             (ra_BCG, dec_BCG) = (coord_BCG[0].ra,coord_BCG[0].dec) if lab==1 else (coord_BCG[1].ra,coord_BCG[1].dec) 
        c_BCG = coord_BCG[0] if lab==1 else coord_BCG[1]

    ra0, dec0 = wcs.all_pix2world(obj.X_c, obj.Y_c, obj.origin)
    c0 = SkyCoord(ra0, dec0, frame='icrs', unit="deg")

    dist_clus_cen = c0.separation(c_BCG).to(u.arcsec).value / 0.322
    clus_cen_angle = c0.position_angle(c_BCG).to(u.deg).value


    if aperture_type== "separate":
        
        x1_eff, y1_eff, x2_eff, y2_eff = [res_cen[key] for key in ["x1_eff", "y1_eff", "x2_eff", "y2_eff"]]
        aper_ems_eff, aper_cons_eff = res_cen["aper_ems_eff"], res_cen["aper_cons_eff"]

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
            x1, y1 = obj.center_pos
            x2, y2 = obj.center_pos
            centroid_offset = 0

        pa, centroid_offset = measure_pa_offset((x1,y1), (x2,y2), wcs, obj)
        diff_angle = np.abs(pa - clus_cen_angle)
        if diff_angle>180:
            diff_angle = 360 - diff_angle

    else:
        x1, y1, x2, y2 = [res_cen[key] for key in ["x1", "y1", "x2", "y2"]] 
        aper_em, aper_con = res_cen["aper_em"], res_cen["aper_con"]
        
        ra1, dec1 = wcs.all_pix2world(x1+obj.X_min,obj.img_thumb.shape[1]-obj.Y_max+y1, obj.origin)
        ra2, dec2 = wcs.all_pix2world(x2+obj.X_min,obj.img_thumb.shape[1]-obj.Y_max+y2, obj.origin)
        c1 = SkyCoord(ra1, dec1, frame='icrs', unit="deg")
        c2 = SkyCoord(ra2, dec2, frame='icrs', unit="deg")

        pa = np.mod(270 + np.arctan2(y2-y1,x2-x1)*180/np.pi, 360)
        centroid_offset = np.sqrt((x1-x2)**2+(y1-y2)**2)

        diff_angle = np.abs(pa - clus_cen_angle)
        if diff_angle>180:
            diff_angle = 360 - diff_angle

    
    ###-----------------------------------###
    
    if plot:
        plt.figure(figsize=(13,12))
        ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
        ax0.plot(wavl, spec, "k",alpha=0.7,lw=3)
        ax0.plot(wavl[con & (wavl<6563.*(1+z_cc))], spec[con & (wavl<6563.*(1+z_cc))], "firebrick",alpha=0.9,lw=3)
        ax0.plot(wavl[con & (wavl>6563.*(1+z_cc))], spec[con & (wavl>6563.*(1+z_cc))], "firebrick",alpha=0.9,lw=3)
        ax0.plot(wavl[em],spec[em],"steelblue",alpha=1,lw=2.5)
        ax0.axvline(np.max([wavl[0], em_range[0]]), color="steelblue", lw=1.5, ls="-.", alpha=0.7)
        ax0.axvline(np.min([wavl[-1], em_range[1]]), color="steelblue", lw=1.5, ls="-.", alpha=0.7)
        ax0.set_xlabel('Wavelength ($\AA$)',fontsize=26)
        ax0.set_ylabel('Flux (10$^{-17}$erg/s/cm$^2$)',fontsize=22)
        ax0.text(0.85,0.85,"z = %.3f"%z_cc,color="k",fontsize=22,transform=ax0.transAxes)

        ax1 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1)
        s = ax1.imshow(img_em, origin="lower",cmap="viridis",vmin=0.,vmax=vmax_5sig(img_em))
        ax1.plot(x1, y1, "violet", marker="+", ms=15, mew=3,alpha=0.95)
       
        loc = plticker.MultipleLocator(base=10) # this locator puts ticks at regular intervals
        ax1.xaxis.set_major_locator(loc)
        ax1.yaxis.set_major_locator(loc)
        ax1.set_xticklabels(ax1.get_xticks().astype("int")+obj.y_min,fontsize=15)
        ax1.set_yticklabels(ax1.get_yticks().astype("int")+obj.x_min,fontsize=15)
#         ax1.set_title("Emisson",color="k",fontsize=18)
#         plt.colorbar(s)
        if centroid_type != "APER":
            ax1s = plt.subplot2grid((3, 3), (1, 0), colspan=1, rowspan=1)
            ax1s.imshow(data_em, origin="lower",cmap="viridis",vmin=0,vmax=vmax_5sig(data_em))
            ax1s.set_xticklabels(ax1s.get_xticks().astype("int")+obj.y_min,fontsize=10)
            ax1s.set_yticklabels(ax1s.get_yticks().astype("int")+obj.x_min,fontsize=10)
            ax1s.set_title("Em. seg.",color="k",fontsize=15)
            
            
        ax2 = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1)
        s = ax2.imshow(img_con*np.sum(con), origin="lower",cmap="hot",vmin=0.,vmax=vmax_5sig(img_con*np.sum(con)))
        ax2.plot(x2, y2, "lightgreen", marker="+", ms=15, mew=3,alpha=0.95)
        ax2.xaxis.set_major_locator(loc)
        ax2.yaxis.set_major_locator(loc)
        ax2.set_xticklabels(ax2.get_xticks().astype("int")+obj.y_min,fontsize=15)
        ax2.set_yticklabels(ax2.get_yticks().astype("int")+obj.x_min,fontsize=15)
#         ax2.set_title("Continuum",color="k",fontsize=18)
#         plt.colorbar(s)
        if centroid_type != "APER":
            ax2s = plt.subplot2grid((4, 3), (3, 2), colspan=1, rowspan=1)
            ax2s.imshow(data_con, origin="lower",cmap="hot",vmin=0.,vmax=vmax_5sig(data_con))
            ax2s.set_xticklabels(ax2s.get_xticks().astype("int")+obj.y_min,fontsize=10)
            ax2s.set_yticklabels(ax2s.get_yticks().astype("int")+obj.x_min,fontsize=10)
            ax2s.set_title("Cont. seg.",color="k",fontsize=15)
        
        ax = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
        
        if hasattr(obj, deep_thumb):
            ax.imshow(deep_img, origin="lower", cmap="gray", vmin=0., vmax=vmax_5sig(deep_img), norm=norm)
        else:
            img = img_con*np.sum(con)
            ax.imshow(img, origin="lower", cmap="gray", vmin=0., vmax=vmax_5sig(img))
        
        xlim,ylim = ax.get_xlim(),ax.get_ylim()
        if centroid_type == "APER":
#             for k, (aper1, aper2) in enumerate(zip(aper_ems,aper_cons)):
#                 try:
#                     ls1 = '-' if (use_value_1[k]==True) else '--'
#                     ls2 = '-' if (use_value_2[k]==True) else '--'
#                     aper1.plot(color='violet',lw=1,ls=ls1,ax=ax1,alpha=0.9)
#                     aper2.plot(color='darkseagreen',lw=1,ls=ls2,ax=ax2,alpha=0.9) 
#                 except AttributeError:
#                     pass                    
            aper_em.plot(color='violet',lw=3,ax=ax1,alpha=0.95)
            aper_con.plot(color='darkseagreen',lw=3,ax=ax2,alpha=0.95)
        ax.contour(data_em,colors="skyblue",alpha=0.9, linewidths=3.5,
                  levels = [0.1*data_em.max(),0.45*data_em.max(),0.8*data_em.max(),data_em.max()])
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("X (pix)",fontsize=28)
        ax.set_ylabel("Y (pix)",fontsize=28)
        
#         for k in range(len(aper_size)):
#             for j in range(n_rand):
#                 ax1.plot(x1s[j,k], y1s[j,k], color="plum", marker=".", ms=3,mew=1,alpha=0.8, zorder=3)
#                 ax2.plot(x2s[j,k], y2s[j,k], "palegreen", marker=".", ms=3,mew=1,alpha=0.8, zorder=3)
#         ax1.scatter(obj.center_pos[0]+da_s*np.cos(theta)-db_s*np.sin(theta), obj.center_pos[1]+da_s*np.sin(theta)+db_s*np.cos(theta), color="orange", alpha=0.3,s=1)
#         ax2.scatter(obj.center_pos[0]+da_s*np.cos(theta)-db_s*np.sin(theta), obj.center_pos[1]+da_s*np.sin(theta)+db_s*np.cos(theta), color="orange", alpha=0.3,s=1)
#         ax.scatter(obj.center_pos[0]+da_s*np.cos(theta)-db_s*np.sin(theta), obj.center_pos[1]+da_s*np.sin(theta)+db_s*np.cos(theta), color="orange", alpha=0.3,s=5)
        
        ax.plot(x1, y1, color="violet", marker="+", ms=25,mew=5,alpha=0.95, zorder=4)
        ax.plot(x2, y2, color="lightgreen", marker="+", ms=25,mew=5,alpha=0.95, zorder=3)
        ax.set_xticklabels(ax.get_xticks().astype("int")+obj.y_min)
        ax.set_yticklabels(ax.get_yticks().astype("int")+obj.x_min)
        
        ax.arrow(0.85,0.15,-np.sin(pa*np.pi/180)/10,np.cos(pa*np.pi/180)/10,color="lightblue",
                 head_width=0.02, head_length=0.02,lw=3,transform=ax.transAxes,alpha=0.95)
        ax.arrow(0.85,0.15,-np.sin(clus_cen_angle*np.pi/180)/10,np.cos(clus_cen_angle*np.pi/180)/10,color="orange",
                 head_width=0.02, head_length=0.02,lw=3,transform=ax.transAxes,alpha=0.95)
        
        ax.text(0.05,0.05,r"$\bf \theta:{\ }%.1f$"%diff_angle,color="lavender",fontsize=25,transform=ax.transAxes)
        ax.text(0.05,0.12,r"$\bf \Delta\,d:{\ }%.1f$"%centroid_offset,color="lavender",fontsize=25,transform=ax.transAxes)
        #mag_auto = -2.5*np.log10(obj.flux_auto) + mag0
        #ax.text(0.05,0.9,"mag: %.1f"%mag_auto,color="lavender",fontsize=15,transform=ax.transAxes)
        if np.ndim(coord_BCG)>0:
            text, color = (r'$\bf NE$', 'lightcoral') if lab==1 else (r'$\bf SW$', 'thistle')
            ax.text(0.85, 0.9, text, color=color, fontsize=25, transform=ax.transAxes)
        plt.subplots_adjust(left=0.05,right=0.95,top=0.95, bottom=0.05, wspace=0.2, hspace=0.25)
    
    if return_for_plot:
        return (em, con), (img_em, img_con), data_em, (aper_em, aper_con), ((x1,y1),(x2,y2))
    else:
        return diff_angle, centroid_offset, dist_clus_cen, pa, clus_cen_angle

    
def measure_pa_offset(pos1, pos2, wcs, obj):
    (x1, y1) = pos1 
    (x2, y2) = pos2
    ra1, dec1 = wcs.all_pix2world(x1 + obj.y_min, obj.img_thumb.shape[1] - obj.x_max + y1, 1)
    ra2, dec2 = wcs.all_pix2world(x2 + obj.y_min, obj.img_thumb.shape[1] - obj.x_max + y2, 1)
    c1 = SkyCoord(ra1, dec1, frame='icrs', unit="deg")
    c2 = SkyCoord(ra2, dec2, frame='icrs', unit="deg")

    pa = np.mod(270 + np.arctan2(y2-y1,x2-x1)*180/np.pi, 360)
    offset = np.sqrt((x1-x2)**2+(y1-y2)**2)

    return pa, offset


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
    Datacube.centroid_analysis_all(Num_v=Num_v, centroid_type='APER',coord_type="angular", n_rand=n_rand, verbose=False)
    
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