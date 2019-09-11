import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.stats import mad_std
from astroquery.vizier import Vizier

import re
import glob
import pprint as pp

from utils import *


def read_SITELLE_DeepFrame(file_path, name, SE_catalog=None, seg_map=None, mask_edge=None):
    """ 
    Read raw SITELLE datacube.
    
    Parameters
    ----------
    file_path : deepframe path.
    name : name of object for saving.
    
    SE_catalog : SExtractor table (None if reading raw).
    seg_map : SExtractor segmentation map (None if reading raw).
    mask_edge : edge mask (None if reading raw).

    Returns
    -------
    deepframe: deepframe class.
    
    """
    deepframe = DeepFrame(file_path, name, SE_catalog, seg_map, mask_edge)
    return deepframe


class DeepFrame: 
    """ A class for SITTLE DeepFrame """
    
    def __init__(self, file_path, name, SE_catalog=None, seg_map=None, mask_edge=None):
        self.hdu = fits.open(file_path)
        self.header = self.hdu[0].header
        self.image = self.hdu[0].data
        self.name = name
        
        # WCS
        self.RA, self.DEC = self.header["TARGETR"], self.header["TARGETD"]
        self.wcs = WCS(self.header)

        try:
            # Read SE measurement
            self.table = Table.read(SE_catalog, format="ascii.sextractor")
            self.seg_map = fits.open(seg_map)[0].data
            self.mask_edge = fits.open(mask_edge)[0].data.astype('bool') 
        
        except TypeError:
            # none for raw frame
            self.table = None
            self.seg_map = None
            self.mask_edge = self.image < 5*mad_std(self.image)
        
    def crossmatch_sdss(self, radius=6*u.arcmin, mag_max=18):
        """ Match bright stars with SDSS (optional) """
        tab_sdss = crossmatch_sdss12(self, self.RA, self.DEC, radius=radius, band='rmag', mag_max=mag_max)
        return tab_sdss
    
    def make_mask_edge(self, save_path = './'):
        """ Mask edges """
        check_save_path(save_path)
        hdu_edge = fits.PrimaryHDU(data=self.mask_edge.astype("float"))
        hdu_edge.writeto(save_path + '%s_DF_edge.fits'%self.name,overwrite=True)
    
    def make_mask_streak(self, file_path, threshold=5, shape_cut=0.15, area_cut=500,
                         display=True, save_plot=True):
        """ 
        Mask bright star streaks (optional)
        
        Parameters
        ----------
        file_path : fits path
        threshold : contour threshold
        shape_cut : cut for shape factor
        area_cut : cut for area inside each border
        save_plot : save streak detection plot in file_path

        """
        self.mask_streak = mask_streak(file_path, threshold=threshold, 
                                       shape_cut=shape_cut, area_cut=area_cut, save_plot=True)
        if display:
            plt.figure(figsize=(6, 6))   
            plt.imshow(self.mask_streak, origin="lower")

    def make_weight_map(self, thre=0.55, wt_min=1e-3, 
                        display=True, save_path = './'):
        """ Make weight map for SE, de-weighting edges / corners (/streaks) 
            to reduce spurious detection """
        image = self.image
          
        yy, xx = np.indices(image.shape)
        cen = ((image.shape[1]-1)/2., (image.shape[0]-1)/2.)

        rr = np.sqrt(((xx-cen[0])/image.shape[1])**2+((yy-cen[1])/image.shape[0])**2)

        weight = (1-sigmoid(rr*100, thre*100)) * (1-wt_min) + wt_min
        weight[self.mask_edge] = wt_min
        
        try:
            weight[self.mask_streak] = wt_min
        except AttributeError:
            pass
        
        if display:
            fig, (ax) = plt.subplots(1,1,figsize=(6,6))
            im = ax.imshow(np.log10(weight), vmin=-5, vmax=0, origin="lower", cmap="rainbow")
            colorbar(im)
            
        check_save_path(save_path)
        hdu_weight_map = fits.PrimaryHDU(data=weight)
        hdu_weight_map.writeto(save_path + 'weight_map_%s.fits'%self.name, overwrite=True)
    
    def subtract_background(self, b_size=128,
                            display=True, plot=False, save_path = './', suffix=""):
        """ Subtract background with a moving box. Edges masked. Sources masked if has SE table. """
        field = self.image
        
        # Clean the spurious detection using the first measurement
        if self.seg_map is None:
            mask = self.mask_edge
        else:
            table = self.table
            segmap2 = self.seg_map.copy()
            num_defect = table["NUMBER"][(table["FLUX_AUTO"]<0)|(table["ELONGATION"]>10)]
            for num in num_defect:
                segmap2[segmap2==num] = 0               
            mask = (segmap2!=0) | (self.mask_edge)
        
        # Subtract background, mask SE detections if has a first run
        back, back_rms = background_sub_SE(field, mask=mask, b_size=b_size, f_size=3, n_iter=20)
        field_sub = field - back
        
        if display:
            display_background_sub(field, back, vmax=1e3)
        if plot:
            check_save_path(save_path)
            hdu_new = fits.PrimaryHDU(data=field_sub, header=self.header)
            hdu_new.writeto(save_path + '/' + '%s_DF%s.fits'%(self.name, suffix), overwrite=True)
            print("Saved background subtracted image as %s_DF%s.fits"%(self.name, suffix))
        
        self.field_sub = field_sub    
        
    def calculate_seeing(self, R_pix=15, cut=[95,99.5], sigma_guess=1., plot=False):
        """ 
        Use the brigh star-like objects (class_star>0.7) to calculate 
        median seeing fwhm by fitting gaussian profiles.
            
        Parameters
        ----------
        R_pix : max range of profiles to be fitted (in pixel)
        cut : percentage of brightness cut
        sigma_guess : initial guess of sigma of gaussian profile (in pixel) 
        plot : plot radial profile for each object in use
        
        """
        
        if self.table is None:
            print("No SE measurement. Pass.")
            return None
        else:
            table = self.table

        # Cut in CLASS_STAR, roundness and brightness
        star_cond = (table["CLASS_STAR"]>0.7) & (table["PETRO_RADIUS"]>0) \
                    & (table["B_IMAGE"]/table["A_IMAGE"]>0.8) & (table["FLAGS"] <4)
        tab_star = table[star_cond]
        F_limit = np.percentile(tab_star["FLUX_AUTO"], cut)
        tab_star = tab_star[(tab_star["FLUX_AUTO"]>F_limit[0])&(tab_star["FLUX_AUTO"]<F_limit[1])]
        
        FWHM = calculate_seeing(tab_star, self.image, self.seg_map, 
                                R_pix=R_pix, sigma_guess=sigma_guess, min_num=5, plot=True)
        self.seeing_fwhm = np.median(FWHM)
        print("Median seeing FWHM in arcsec: %.3f"%self.seeing_fwhm)
        
        

def read_raw_SITELLE_datacube(file_path, name, wavn_range=[12100, 12550]):
    """ 
    Read raw SITELLE datacube.
    
    Parameters
    ----------
    file_path: datacube path.
    name: name of object for saving.
    wavn_range: range of wave number (in cm^-1) to be used.

    Returns
    -------
    raw_datacube: raw SITTLE datacube class.
    
    """
    raw_datacube = Raw_Datacube(file_path, name, wavn_range=[12100, 12550])
    return raw_datacube

class Raw_Datacube: 
    """ A class for raw SITTLE datacube """
    
    def __init__(self, file_path, name, wavn_range):
        self.hdu = fits.open(file_path)
        self.header = self.hdu[0].header
        self.name = name
        
        # Pixel scale, Step of wavenumber, Start of wavenumber, Number of steps
        self.pix_scale = self.hdu[0].header["PIXSCAL1"]   #arcsec/pixel
        self.d_wavn = self.hdu[0].header["CDELT3"]        #cm^-1 / pixel
        self.wavn_start = self.hdu[0].header["CRVAL3"]    #cm^-1
        self.nstep = self.hdu[0].header["STEPNB"] 
        
        # Wavenumber axis
        self.wavn = np.linspace(self.wavn_start, self.wavn_start+(self.nstep-1)*self.d_wavn, self.nstep)  #cm^-1
        
        # Effective wave number range
        wavn_eff_range = (self.wavn>wavn_range[0]) & (self.wavn<wavn_range[1])
        self.wavl = 1./self.wavn[wavn_eff_range][::-1]*1e8
        
        # Raw effective datacube and stack field
        self.raw_datacube = self.hdu[0].data[wavn_eff_range][::-1]
        self.raw_stack_field = self.raw_datacube.sum(axis=0)
        
        # Field edge/saturations to be masked
        self.mask_edge = self.raw_stack_field < 5 * mad_std(self.raw_stack_field)
        
        # Modify the fits for saving the new cube
        self.hdu_header_new = self.hdu[0].header.copy()
        self.hdu_header_new["CRVAL3"] = self.wavn[wavn_eff_range][0]  # New minimum wavenumber in cm-1
        self.hdu_header_new["STEPNB"] = np.sum(wavn_eff_range)   # New step
        
    def display(self, img, vmax=None, a=0.1):
        # Display a image
        plt.figure(figsize=(12,12))
#         if vmax==None: vmax=vmax_2sig(img)
        norm = ImageNormalize(stretch=AsinhStretch(a=a))
        plt.imshow(img, vmin=0.,vmax=vmax, norm=norm, origin="lower",cmap="gray")
        
    def save_mask_edge(self, save_path = './'):
        # Save the edge mask to be read later
        check_save_path(save_path)
        hdu_edge = fits.PrimaryHDU(data=self.mask_edge.astype("float")*10)
        hdu_edge.writeto(save_path + 'Raw_stack_%s_mask.fits'%self.name,overwrite=True)
        
    def save_weight_map(self, region_path, weight=0.001, save_path = './'):
        # Mask regions (eg. star spikes) by making a weight map for SE input
        import pyregion
        reg = pyregion.open(region_path)
        self.mymask = reg.get_mask(hdu=self.hdu[0], shape=self.hdu[0].shape[1:])

        weight_map =(~self.mymask).astype("float") + weight
        
        check_save_path(save_path)
        hdu_weight_map = fits.PrimaryHDU(data=weight_map)
        hdu_weight_map.writeto(save_path + 'weight_map_stack_%s.fits'%self.name, overwrite=True)
    
    def remove_background(self, box_size=128, filter_size=3, n_iter=10, save_path=None, plot=False):
        """ Remove background (low-frequency component) using SEXtractor estimator (filtering + mode) """
        if plot:
            check_save_path(save_path)
        
        self.datacube_bkg_sub = np.empty_like(self.raw_datacube)
        for i, field in enumerate(self.raw_datacube):
            if np.mod(i+1, 10)==0: print("Removing background... Channel: %d"%(i+1))
            
            back, back_rms = background_sub_SE(field, mask=self.mask_edge,
                                               b_size=box_size, f_size=filter_size, n_iter=n_iter)
            field_sub = field - back

            self.datacube_bkg_sub[i] = field_sub
            
            # Save background subtraction
            if plot:
                fig = display_background_sub(field, back, vmax=1)
                plt.savefig(save_path + "bkg_sub_channel%d.png"%(i+1),dpi=100)
                plt.close(fig)  
                
        self.stack_field = self.datacube_bkg_sub.sum(axis=0)
        
    def remove_fringe(self, channels=None, sn_source=2,
                      box_size=8, filter_size=1, n_iter=20, save_path=None, plot=False):
        """ Remove fringe and artifacts for specified channels """
        if channels is None: return None
        
        check_save_path(save_path)
        
        mask_source, segmap = make_mask_map(self.stack_field, sn_thre=sn_source, b_size=128)
        mask = (self.mask_edge) | (mask_source)
        
        self.datacube_bkg_fg_sub = self.datacube_bkg_sub.copy()
        for k in channels:
            field = self.datacube_bkg_sub[k-1]
            print("Removing fringe... Channel: %d"%k)
                
            back, back_rms = background_sub_SE(field, mask=mask,
                                               b_size=box_size, f_size=filter_size, n_iter=n_iter)
            self.datacube_bkg_fg_sub[k-1] = field - back

            # Save fringe subtraction
            if plot:
                fig = display_background_sub(field, back, vmax=1)
                plt.savefig(save_path + "fg_sub_channel%d.png"%k,dpi=100)
                plt.close(fig)  
                    
        self.stack_field = self.datacube_bkg_fg_sub.sum(axis=0)
        self.subtract_fringe = True
                    
    def save_fits(self, save_path = './', suffix=""):
        """Write Stacked Image and Datacube after background & fringe subtraction"""
        print("Saving background & fringe subtracted datacube and stacked field...")
        
        datacube_sub = self.datacube_bkg_fg_sub if self.subtract_fringe else self.datacube_bkg_sub
        
        check_save_path(save_path)
        hdu_cube = fits.PrimaryHDU(data = datacube_sub, header=self.hdu_header_new)
        hdu_cube.writeto(save_path + '%s_cube%s.fits'%(self.name, suffix),overwrite=True)
        
        for keyname in ["CTYPE3","CRVAL3","CUNIT3","CRPIX3","CDELT3","CROTA3"]:
            try:
                self.hdu_header_new.remove(keyname)
            except KeyError:
                pass
        hdu_stack = fits.PrimaryHDU(data = self.stack_field, header=self.hdu_header_new)
        hdu_stack.writeto(save_path + '%s_stack%s.fits'%(self.name, suffix),overwrite=True)
        
        
class Read_Datacube:
    
    """ 
    Read SITELLE datacube.
    
    Parameters
    ----------
    cube_path: datacube path.
    name: name of object for saving.
    mode: spectra extraction method 'ISO' or 'APER' (requring SExtractor output)
    
    table: table of source properties from Photutils / SExtractor (None before source extraction).
    seg_map: SExtractor segmentation map (None before source extraction).
    mask_edge: edge mask (None if reading raw).

    """
    
    def __init__(self, cube_path, name, mode="MMA",
                 table=None, seg_map=None, mask_edge=None,
                 deep_frame=None, z0=None):
        
        self.hdu = fits.open(cube_path)
        self.header = self.hdu[0].header
        self.cube = self.hdu[0].data
        self.stack_field = self.cube.sum(axis=0)
        self.shape = self.cube.shape
        
        self.name = name
        self.z0 = z0
        self.mode = mode
        
        if (mode=="MMA")|(mode=="m2"):            
            self.mask_edge = None
            if table is not None:
                self.table = Table.read(table, format='ascii')
        elif mode=="APER":
            # Read SE measurement. 
            self.table = Table.read(table, format="ascii.sextractor")
            self.mask_edge = fits.open(mask_edge)[0].data.astype('bool')             
        else:
            raise ValueError("Choose an extraction mode: 'MMA' or 'm2' or 'APER' ('APER' requires SExtractor output)")
        
        try:
            self.seg_map = fits.open(seg_map)[0].data
        except ValueError:
            self.seg_map = None
        
        if deep_frame is not None:
            self.deep_frame = fits.open(deep_frame)[0].data
        else:
            self.deep_frame = None
        
        # Pixel scale, Step of wavenumber, Start of wavenumber, Number of steps
        self.d_wavn = self.hdu[0].header["CDELT3"] #cm^-1 / pixel
        self.wavn_start = self.hdu[0].header["CRVAL3"] #cm^-1
        self.nstep = self.hdu[0].header["STEPNB"] 
        
        # Total wavenumber
        self.wavn = np.linspace(self.wavn_start, self.wavn_start+(self.nstep-1)*self.d_wavn, self.nstep)  #cm^-1
        
        # Effective data range from 12100 cm^-1 to 12550 cm^-1 
        self.wavl = 1./self.wavn[::-1]*1e8
        self.d_wavl = np.diff(self.wavl).mean()
        
        # For generating different templates 
        self.Temp_Lib = {}
        self.wavl_temps = {}
        self.Stddev_Temps = {}
        self.Line_Ratios_Temps = {}
        
        # Fot CC using different templates
        self.CC_zccs_Temps = {}
        self.CC_Rs_Temps = {}
        self.CC_SNRs_Temps = {}
        self.CC_SNR_ps_Temps = {}
        self.CC_flag_edge_Temps = {}
        self.CCs_Temps  = {}
       
    
    def ISO_source_detection(self, sn_thre=2, npixels=8,
                             nlevels=64, contrast=0.01, closing=True,
                             columns=['id', 'xcentroid', 'ycentroid', 'ellipticity', 'equivalent_radius', 'area'],
                             n_win=3, parallel=False, save=True, save_path = './', suffix=""):
        """ Source Extraction based on Isophotal of S/N """
        from skimage import morphology
        from photutils import source_properties
        
        # Pixel-wise manipulation in wavelength
        
        if self.mode == "MMA":
            print("Use the map of maximum of moving average (MMA) along wavelength to detect source.")
            
            if parallel:
                print("Run moving average in parallel. This could be slower using few cores.")
                
                from usid_processing import parallel_compute
                from functools import partial
                p_moving_average_cube = partial(moving_average_cube, cube=self.cube.copy())
                results = parallel_compute(np.arange(self.shape[2]), p_moving_average_cube,
                                           lengthy_computation=False, verbose=True)
                src_map = np.max(results,axis=2).T
                
            else:
                src_map = np.empty_like(self.stack_field)
                for i in range(self.shape[1]):
                    for j in range(self.shape[2]):
                        src_map[i,j] = max(moving_average(self.cube[:,i,j], n_win))
                    
        elif self.mode == "m2":
            print("Use the map of second maximum (m2) along wavelength to detect source.")
            src_map = np.partition(self.cube, kth=-2, axis=0)[-2]
        
        # Detect + deblend source
        print("Detecting and deblending source...")
        back, back_rms = background_sub_SE(src_map, b_size=128)
        threshold = back + (sn_thre * back_rms)
        segm0 = detect_sources(src_map, threshold, npixels=npixels)
        segm = deblend_sources(src_map, segm0, npixels=npixels, nlevels=nlevels, contrast=contrast)
        if closing is True:
            seg_map = morphology.closing(segm.data)
        else:
            seg_map = segm.data
        
        # Save 2nd max map and segmentation map
        if save:
            check_save_path(save_path)
            hdu_m2 = fits.PrimaryHDU(data=src_map, header=self.header)
            hdu_m2.writeto(save_path + '%s_%s%s.fits'%(self.name, self.mode, suffix), overwrite=True)
            hdu_seg = fits.PrimaryHDU(data=seg_map, header=None)
            hdu_seg.writeto(save_path + '%s_segm_%s%s.fits'%(self.name, self.mode, suffix), overwrite=True)
        
        # Meausure properties of detections
        cat = source_properties(src_map, segm)
        tab = cat.to_table(columns=columns)   
        for name in tab.colnames:
            if name!="id":
                try:
                    tab[name].info.format = '.2f'
                except ValueError:
                    pass
            
        # Save as table
        if save:
            check_save_path(save_path)
            tab.rename_column('id', 'NUMBER')
            tab.write(save_path + '%s_%s%s.dat'%(self.name, self.mode, suffix), format='ascii', overwrite=True)
            
        self.table = tab
        self.src_map = src_map
 
        return src_map, segm, seg_map
    
    def ISO_spec_extraction_all(self, seg_map):
        labels = np.unique(seg_map)[1:]
        
        for k, lab in enumerate(labels):
            if np.mod(k+1, 400)==0: print("Extract spectra... %d/%d"%(k+1, len(labels)))
            spec_opt = self.cube[:, seg_map==lab].sum(axis=1)
            self.obj_specs_opt = np.vstack([self.obj_specs_opt, spec_opt]) if k>0 else [spec_opt]   
        
        self.obj_nums = labels
        
    def calculate_seeing(self, R_pix = 15, cut=[95,99.5], sigma_guess=1., plot=False):
        """ 
        Use the brigh star-like objects (class_star>0.7) to calculate 
        median seeing fwhm by fitting gaussian profiles.
            
        Parameters
        ----------
        R_pix : max range of profiles to be fitted (in pixel)
        cut : percentage of cut in isophotal magnitude
        sigma_guess : initial guess of sigma of gaussian profile (in pixel) 
        plot : plot radial profile for each object in use
        
        """
        
        # Cut in CLASS_STAR, measurable, roundness and mag threshold
        star_cond = (self.table["CLASS_STAR"]>0.7) & (self.table["PETRO_RADIUS"]>0) \
                    & (self.table["B_IMAGE"]/self.table["A_IMAGE"]>0.7) & (table["FLAGS"] <4)
        tab_star = self.table[star_cond]
        F_limit = np.percentile(tab_star["FLUX_AUTO"], cut)
        tab_star = tab_star[(tab_star["FLUX_AUTO"]>F_limit[0])&(tab_star["FLUX_AUTO"]<F_limit[1])]
        
        FWHM = calculate_seeing(tab_star, self.stack_field, self.seg_map, 
                                R_pix=R_pix, sigma_guess=sigma_guess, min_num=5, plot=True)
        self.seeing_fwhm = np.median(FWHM)
        print("Median seeing FWHM in arcsec: %.3f"%self.seeing_fwhm)
        
    def spec_extraction(self, num, ext_type='opt', 
                        ks = np.arange(1.,4.5,0.2), k1=5., k2=8.,
                        print_out=False, plot=False, display=False):
        """
        Extract spectal using an optimal aperture, local background evaluated from (k1, k2) annulus
        
        Parameters
        ----------
        num : SE object number
        ks : a set of aperture ring (in SE petrosian radius) to determined the optimized aperture
        k1, k2 : inner/outer radius (in SE petrosian radius) of annulus for evaluate background RMS
        print_out : print out the optimized k and S/N
        plot : whether to plot the S/N vs k curve
        display: whether to display the image thumbnail with apertures
        
        """
        
        tab = self.table[self.table["NUMBER"]==num]
        obj_SE = Object_SE(tab, cube=self.cube, deep_frame=self.deep_frame,
                           img_seg=self.seg_map, mask_field=self.mask_edge)
        
        if obj_SE.R_petro==0:
            if print_out: print("Error in measuring R_petro, dubious source")
            return (np.zeros_like(self.wavl), 1., 0, None)
        
        k, snr = obj_SE.compute_aper_opt(ks=ks, k1=k1, k2=k2,
                                                 ext_type=ext_type, print_out=print_out, plot=plot)
        spec = obj_SE.extract_spec(k, ext_type=ext_type, k1=k1, k2=k2, wavl=self.wavl, plot=plot)
        if display&(snr>0):
            obj_SE.img_display()
        return spec, k, snr, obj_SE.apers
    
    def spec_extraction_all(self, ks = np.arange(1.,3.6,0.2), k1=5., k2=8., display=True, save_path=None):
        """Extract spectra all SE detections with two optimized aperture"""
        self.obj_nums = np.array([])
        self.obj_aper_cen = np.array([])
        self.obj_aper_opt = np.array([])
        for num in self.table["NUMBER"]:
            spec_cen, k_cen, snr_cen, apers = self.spec_extraction(num=num, ext_type='sky', 
                                                       ks=ks, k1=k1, k2=k2,
                                                       print_out=False, plot=False, display=False)
            self.obj_specs_cen = np.vstack((self.obj_specs_cen, spec_cen)) if len(self.obj_nums)>0 else [spec_cen]
            self.obj_aper_cen = np.append(self.obj_aper_cen, k_cen)
            
            spec_opt, k_opt, snr_opt, apers_opt = self.spec_extraction(num=num, ext_type='opt', 
                                                                       ks=ks, k1=k1, k2=k2,
                                                                       print_out=False, plot=False, display=display)
            self.obj_specs_opt = np.vstack((self.obj_specs_opt, spec_opt)) if len(self.obj_nums)>0 else [spec_opt]
            self.obj_aper_opt = np.append(self.obj_aper_opt, k_opt)
                
            # Display the image marking the two apertures and annulus
            if display&(snr_opt>0):
                apers[0].plot(color='limegreen', lw=1.5, ls='--')
                plt.savefig(save_path+"SE#%d.png"%(num),dpi=150)
                plt.close()
            
            self.obj_nums = np.append(self.obj_nums, num)
            print ("#%d spectra extracted"%(num))
            
    
    def save_spec_plot(self, save_path=None):
        """Save normalized extracted spectra by the two apertures"""
        check_save_path(save_path)
        for (n, spec_opt) in zip(self.obj_nums, self.obj_specs_opt):
            plt.plot(self.wavl, spec_opt/spec_opt.max(), color='k', label="opt", alpha=0.9)
            plt.xlabel("wavelength",fontsize=12)
            plt.ylabel("normed flux",fontsize=12)
            plt.legend(fontsize=12)
            plt.savefig(save_path+"#%d.png"%n,dpi=100)
            plt.close()
            
    def fit_continuum_all(self, model='GP', save_path=None, plot=False, verbose=True,
                          GP_kwds={'edge_ratio':None, 'kernel_scale':100, 'kenel_noise':1e-3}):   
        """
        Fit continuum of all the extracted spectrum

        Parameters
        ----------
        model : continuum model used to subtract from spectrum
                "GP": flexible non-parametric continuum fitting
                      edge_ratio : edge threshold of the filter band (above which the
                                   edge is replaced with a median to avoid egde issue).
                                   
                                   If None, use the meadian edge ratio estimated from
                                   the stack of all the spectra.
                                   
                      kernel_scale : kernel scale of the RBF kernel
                      kenel_noise : noise level of the white kernel
                
                "Trapz1d": 1D Trapezoid model with constant continuum and edge fitted by a constant slope
        
        """
        
        if plot:
            check_save_path(save_path)
            
        if (model=='GP') & (GP_kwds['edge_ratio'] is None):
            # Stacked spectra to estimate edge ratio
            self.spec_stack = (self.obj_specs_opt/self.obj_specs_opt.max(axis=1).reshape(-1,1)).sum(axis=0)
            GP_kwds['edge_ratio'] = 0.5*np.sum((self.spec_stack - np.median(self.spec_stack) \
                                            <= -2.5*mad_std(self.spec_stack))) / len(self.wavl)
            if verbose:
                print('Fit continuum with GP. No edge_ratio is given. Use estimate = %.2f'%GP_kwds['edge_ratio'])
        
        
        for n in self.table["NUMBER"]:
            
            # Skip supurious detections (edges, spikes, etc.)
            iloc = (self.table["NUMBER"]==n)
            if self.mode!="APER":
                spurious_detection = self.table[iloc]['ellipticity']>= 0.9
            else:
                spurious_detection = (self.table[iloc]["PETRO_RADIUS"] <=0) | (self.table[iloc]["FLUX_AUTO"] <=0)
            
            blank_array = np.zeros(len(self.wavl)+1)
                
            if spurious_detection:
                if verbose:
                    print("Spurious detection #%d ... Skip"%n)
                res, cont_fit = (blank_array, blank_array)
            else:    
                if verbose:
                    if np.mod(n, 200)==0: 
                        print("Fit spectra continuum ... %d/%d"%(n, len(self.table["NUMBER"])))
                try:   
                    # Fit continuum
                    spec = self.obj_specs_opt[self.obj_nums==n][0]
                    res, wavl_rebin, cont_fit = fit_continuum(spec, self.wavl, model=model, 
                                                              edge_ratio=GP_kwds['edge_ratio'],
                                                              kernel_scale=GP_kwds['kernel_scale'], 
                                                              kenel_noise=GP_kwds['kenel_noise'],
                                                              verbose=False, plot=plot)
                    if plot:
                        plt.savefig(save_path+"#%d.png"%(n),dpi=100)
                        plt.close()

                except Exception as e:
                    res, cont_fit = (blank_array, blank_array)
                    if verbose:
                        print("Spectrum #%d continuum fit failed ... Skip"%n)

            self.obj_res_opt = np.vstack((self.obj_res_opt, res)) if n>1 else [res]
            self.obj_cont_fit = np.vstack((self.obj_cont_fit, cont_fit)) if n>1 else [cont_fit]
        
        print("Continuum Fitting Finished!")
        self.wavl_rebin = wavl_rebin
       
            
    def save_spec_fits(self, save_path='./', suffix="_all"):  
        """Save extracted spectra and aperture (in the unit of SE pertrosian radius) as fits"""
        check_save_path(save_path)
        hdu_num = fits.PrimaryHDU(data=self.obj_nums)
        
        hdu_spec_opt = fits.ImageHDU(data=self.obj_specs_opt)
        size = self.table['equivalent_radius'].value if self.mode!="APER" else self.obj_aper_opt
        hdu_size = fits.ImageHDU(data=size)
#         hdu_size = fits.BinTableHDU.from_columns([fits.Column(name="size",array=self.obj_aper_opt,format="E")])
        
        hdu_res_opt = fits.ImageHDU(data=self.obj_res_opt)
        hdu_wavl_rebin = fits.ImageHDU(data=self.wavl_rebin)
        
        hdul = fits.HDUList(hdus=[hdu_num, hdu_spec_opt, hdu_size, hdu_res_opt, hdu_wavl_rebin])
        hdul.writeto(save_path+'%s-spec-%s%s.fits'%(self.name, self.mode, suffix), overwrite=True)
        
        
    def read_spec(self, file_path):
        """Read generated extracted spectra and aperture fits"""
        hdu_spec = fits.open(file_path)
        
        self.obj_nums = hdu_spec[0].data
        
#         self.obj_specs_cen = hdu_spec[1].data
#         self.k_apers_cen = hdu_spec[2].data['k_aper_cen']
        
        self.obj_specs_opt = hdu_spec[1].data
        self.size_opt = hdu_spec[2].data
        
        self.obj_res_opt = hdu_spec[3].data
        self.wavl_rebin = hdu_spec[4].data
        
        
    def generate_template(self, n_ratio=50, n_stddev=10, n_intp=2,
                          temp_type="Ha-NII", 
                          temp_model="gauss", 
                          ratio_prior="log-uniform", 
                          z_type="cluster",
                          ratio_range=[1., 8], 
                          temp_params={'box_width':4,
                                       'sigma_max':200,
                                       'a_sinc':5},
                          plot=False):
        
        """
        Generate template(s) for cross-correlation.
        
        Parameters
        ----------
        temp_type : template type ("Ha-NII" or "Hb-OIII" or "OII")
        temp_model : template line profile model ("gauss", "sincgauss" or "delta")
        n_stddev/n_ratio : number of template=n_stddev*n_ratio, if >1 generate a template library 
        n_intp : time of interpolation (controling the smoothness of CC function), =1 is no interpolation
        
        ratio_prior : template line ratios prior type 
                    "sdss": sdss line ratio from MPA-JHU catalog
                    "log-uniform": uniform grid in log  
                    
        ratio_range : range of line ratio (lower, upper)
                        only used for the template library with "uniform" prior)
                        
        temp_params : 
                    'box_width' : width of box model (should be small to approximate delta,
                                  but remind to check if sampled by the wavelength axis)
                                  
                    'sigma_max' : upper limit of broadening in km/s for Gaussian models
                                  fwhm (2.355*sigma) between spectral resolution to galaxy broadening v=300km/s.
                    # sigma_v = v/sqrt(3), sigma_gal = sigma_v/3e5 * 6563*(1+z_cluster) 
                    
                    'a_sinc' : parameter a in the sinc function (y=sin(ax)/(ax))
        """
        
        
        
        if temp_model == "box":
            box_wid = np.max([temp_params['box_width'], self.d_wavl])
            self.Stddevs = np.ones(n_ratio) * box_wid/2.355 # Minimum width for cross-correlation
            n_stddev = 1
            
        elif (temp_model == "gauss")|(temp_model == "sincgauss"):
        
            if temp_type=="OII":
                n_ratio = 1
                
                
            sigma_max = temp_params['sigma_max']
            #sig_gal_m = sigma_max/np.sqrt(3)/3e5 * 6563*(1+self.z0)
            sig_gal_m = sigma_max/3e5 * 6563*(1+self.z0)
            sig_min = self.d_wavl/2.355
            sig_max = np.sqrt(sig_min**2 + sig_gal_m**2)
            
            stddevs = np.linspace(sig_min, sig_max, n_stddev)
            self.Stddevs = np.repeat(stddevs, n_ratio).reshape(n_ratio, n_stddev).T.ravel() 
                
        else: print('Model type: "box" or "gauss" or "sincgauss"')
        
        num_temp = n_ratio * n_stddev
        print("Template: %s_%s  Total Number: %d"%(temp_type, temp_model, num_temp))
        
        if ratio_prior=="log-uniform":    
            # ratio of 1st peak vs 2nd peak
            peak_ratio = np.logspace(np.log10(ratio_range[0]), 
                                     np.log10(ratio_range[1]), n_ratio)[::-1]  # line ratio from high to low
            peak_ratios = np.repeat(peak_ratio, n_stddev).ravel()

        elif ratio_prior=="sdss":
            dist_line_ratio = np.loadtxt("./%s/SDSS-DR7_line_ratio.txt"%self.name)
            peak_ratios = np.random.choice(dist_line_ratio, size=num_temp)   # note: Ha_NII6584 ratios

        if z_type=="mid":
            # line located at the middle of the filter
            z_ref = (self.wavl[0]+self.wavl[-1])/2./6563.-1  
        
        elif z_type=="cluster":
            z_ref = self.z0

        if temp_type=="Ha-NII":
            self.Line_ratios = np.vstack([[1, 3*ratio, 3] for ratio in peak_ratios])
            z_temp = 6563*(1+z_ref)/6563.-1

        elif temp_type=="Hb-OIII":        
            self.Line_ratios = np.vstack([[1, ratio] for ratio in peak_ratios])
            z_temp = 6563*(1+z_ref)/5007.-1

        elif temp_type=="OII":
            self.Line_ratios = np.vstack([[1] for k in range(num_temp)])
            z_temp = 6563*(1+z_ref)/3727.-1

        self.n_intp = n_intp
        temp_step = int(len(self.wavl)) * n_intp +1
        temps = np.empty((num_temp, temp_step))

        for i, (lr, std) in enumerate(zip(self.Line_ratios, self.Stddevs)):
            temp_par={'line_ratio':lr, 'sigma':std}
            
            if temp_model == "box":
                temp_par['box_width'] = box_wid
            if temp_model == "sincgauss":
                temp_par['a_sinc'] = temp_params['a_sinc']
                
            temps[i], self.wavl_temp = generate_template(self.wavl, z=z_temp, 
                                                         temp_model=temp_model, 
                                                         temp_type=temp_type,
                                                         temp_params=temp_par,                                                      
                                                         n_intp=n_intp,
                                                         alpha=0.1, plot=plot)
        self.temps = temps
          
        typ_mod = temp_type + "_" + temp_model
        
        self.Temp_Lib[typ_mod] = self.temps
        self.wavl_temps[typ_mod] = self.wavl_temp
        self.Stddev_Temps[typ_mod] = self.Stddevs
        self.Line_Ratios_Temps[typ_mod] = self.Line_ratios
        
    def Save_Template(self, temp_type="Ha-NII", temp_model="gauss", 
                      save_path='./', suffix=""): 
        
        """Save Emission Line Template as Fits"""
        
        typ_mod = temp_type + "_" + temp_model
        
        hdu_temps  = fits.PrimaryHDU(self.Temp_Lib[typ_mod])
        hdu_wavl_temp = fits.ImageHDU(self.wavl_temp)
        hdu_stddev_temp = fits.ImageHDU(self.Stddev_Temps[typ_mod])
        hdu_line_ratio_temp = fits.ImageHDU(self.Line_Ratios_Temps[typ_mod])

        hdul = fits.HDUList(hdus=[hdu_temps, hdu_wavl_temp, hdu_stddev_temp, hdu_line_ratio_temp])
        print("Save %s templates for %s"%(self.name, typ_mod))
        hdul.writeto(save_path+'Template-%s_%s%s.fits'%(self.name, typ_mod, suffix), overwrite=True) 

    def Read_Template(self, dir_name, n_intp=2):
        
        """Read Emission Line Template from directory"""
        temp_list = glob.glob(dir_name+'/Template-%s_*.fits'%self.name)
        
        self.n_intp = n_intp
        print("Read Emission Line Template:"), pp.pprint(temp_list)
        
        for filename in temp_list:
            temp_type, temp_model, _ = re.split(r'\s|_|\.', filename)[-3:]
            typ_mod = temp_type + "_" + temp_model
            
            hdul = fits.open(filename)
            self.temps = hdul[0].data
            self.wavl_temp = hdul[1].data
            self.Stddevs = hdul[2].data
            self.Line_ratios = hdul[3].data

            self.Temp_Lib[typ_mod] = self.temps
            self.wavl_temps[typ_mod] = self.wavl_temp
            self.Stddev_Temps[typ_mod] = self.Stddevs
            self.Line_Ratios_Temps[typ_mod] = self.Line_ratios
            
    def cross_correlation(self, num, 
                          temp_type="Ha-NII", 
                          temp_model="gauss", 
                          h_contrast=0.1, 
                          edge=15,
                          kind_intp="linear", 
                          rv=None, plot=True):
        
        """Cross-correlation for one SE detection.
        
        Parameters
        ----------
        num : number of object in the SE catalog
        temp_type : template type ("Ha-NII" or "Hb-OIII" or "OII")
        temp_model : template line profile model ("gauss", "sincgauss" or "delta")
        kind_intp : scipy interp1d kind, default is "linear"
        h_contrast : threshold of CC peak finding, relative to the highest peak
        rv : relative velocity to z=z0, None if not computed yet (e.g. the first trial in the library, or single template)
        
        Returns
        ----------
        ccs : cross-correlation functions for each template
        rv : relative velocity to z_sys
        z_ccs : matched redshift for each template
        Rs : peak significance for each template
        Contrasts : peak contrast (peak_1/peak_2) for each template
        SNRs : detection S/N (masking signal) for each template
        SNR_ps : peak S/N (masking peak) for each template
        
        """ 
            
        if plot:    
            fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(9,7))
        else:
            fig, axes = None, (None, None)
        
        res = self.obj_res_opt[self.obj_nums==num][0]  # Read residual spectra
        
        # Read the template information, and pass to cross-correlation
        temps = self.Temp_Lib[temp_type + "_" + temp_model]
        wavl_temp = self.wavl_temps[temp_type + "_" + temp_model]
        stddevs = self.Stddev_Temps[temp_type + "_" + temp_model]
        line_ratios = self.Line_Ratios_Temps[temp_type + "_" + temp_model]
        
        # result_cc: ccs, rv, z_ccs, Rs, Contrasts, SNRs, SNR_ps, flag_edge
        result_cc = xcor_SNR(res=res, 
                             wavl_rebin=self.wavl_rebin, 
                             temps=temps, 
                             wavl_temp=wavl_temp, 
                             d_wavl=self.d_wavl, 
                             z_sys=self.z0, 
                             n_intp=self.n_intp, 
                             kind_intp=kind_intp,                                                                 
                             temp_type=temp_type, 
                             temp_model=temp_model,
                             temps_params={'stddev':stddevs, 
                                           'line_ratio':line_ratios}
                             h_contrast=h_contrast, 
                             rv=rv, edge=edge,
                             plot=plot, fig=fig, axes=axes)
        
        ccs, rv, z_ccs, Rs, Contrasts, SNRs, SNR_ps, flag_edge = result_cc
        
        if plot: plt.tight_layout()
            
        r_max = np.argmax(Rs)    
        z_best = z_ccs[r_max]
        print ("Detection #%d  z: %.3f  Peak R: %.3f  Detction S/N: %.3f Peak S/N: %.3f"\
                               %(num, z_best, Rs[r_max], SNRs[r_max], SNR_ps[r_max]))
        
            
        return ccs, rv, z_ccs, Rs, Contrasts, SNRs, SNR_ps, flag_edge

    def cross_correlation_all(self, 
                              temp_type="Ha-NII", 
                              temp_model="gauss",
                              kind_intp="linear",
                              edge=30,
                              h_contrast=0.1):
        """Cross-correlation for all SE detections in the catalog.
        
        Parameters
        ----------
        temp_type : template type ("Ha-NII" or "Hb-OIII" or "OII")
        temp_model : template line profile model ("gauss", "sincgauss" or "delta")
        kind_intp : scipy interp1d kind, default is "linear"
        h_contrast : threshold of CC peak finding, relative to the highest peak
        
        """ 
        
        self.temp_type = temp_type
        self.temp_model = temp_model
        typ_mod = temp_type + "_" + temp_model
        self.typ_mod = typ_mod
        
        self.temps = self.Temp_Lib[typ_mod]
        
        self.cc_nums = np.array([], dtype="int64")
        self.CC_zccs = np.zeros((len(self.obj_nums), len(self.temps)))
        self.CC_Rs = np.zeros((len(self.obj_nums), len(self.temps)))
        self.CC_SNRs = np.zeros((len(self.obj_nums), len(self.temps)))
        self.CC_SNR_ps = np.zeros((len(self.obj_nums), len(self.temps)))
        self.CC_flag_edge = np.zeros(len(self.obj_nums))
        
      
        for j, num in enumerate(self.obj_nums):
            if j==0:
                ccs, rv0, z_ccs, Rs, Contrasts, SNRs, SNR_ps, flag_edge = self.cross_correlation(num, h_contrast=h_contrast, 
                                                                              temp_type=temp_type, temp_model=temp_model, 
                                                                              kind_intp=kind_intp, edge=edge, rv=None, plot=False)
                self.rv = rv0
            else:
                ccs, rv, z_ccs, Rs, Contrasts, SNRs, SNR_ps, flag_edge = self.cross_correlation(num, h_contrast=h_contrast, 
                                                                             temp_type=temp_type, temp_model=temp_model, 
                                                                             kind_intp=kind_intp, edge=edge, rv=self.rv, plot=False)
               
            self.CC_zccs[j] = z_ccs
            self.CC_Rs[j] = Rs
            self.CC_SNRs[j] = SNRs
            self.CC_SNR_ps[j] = SNR_ps
            self.CC_flag_edge[j] = flag_edge
            self.CCs = np.concatenate([self.CCs, [ccs]]) if len(self.cc_nums)>0 else [ccs]
            
            self.cc_nums = np.append(self.cc_nums, num)
        
        self.CC_zccs_Temps[typ_mod] = self.CC_zccs
        self.CC_Rs_Temps[typ_mod] = self.CC_Rs
        self.CC_SNRs_Temps[typ_mod] = self.CC_SNRs
        self.CC_SNR_ps_Temps[typ_mod] = self.CC_SNR_ps
        self.CC_flag_edge_Temps[typ_mod] = self.CC_flag_edge
        self.CCs_Temps[typ_mod] = self.CCs
        
            
    def save_cc_fits(self, save_path='./', suffix=""): 
        """Save cross-correlation results as fits"""
        
        hdu_cc_nums = fits.PrimaryHDU(np.int64(self.cc_nums))
        hdu_CC_Rs = fits.ImageHDU(data=self.CC_Rs)
        hdu_CC_SNRs = fits.ImageHDU(data=self.CC_SNRs)
        hdu_CC_SNR_ps = fits.ImageHDU(data=self.CC_SNR_ps)
        hdu_CC_zccs = fits.ImageHDU(data=self.CC_zccs)
        hdu_CC_flag_edge = fits.ImageHDU(data=self.CC_flag_edge)
#         hdu_CCs = fits.ImageHDU(data=self.CCs)

        hdul = fits.HDUList(hdus=[hdu_cc_nums, hdu_CC_Rs, hdu_CC_SNRs, hdu_CC_SNR_ps, hdu_CC_zccs, hdu_CC_flag_edge])
        print("Save cross-correlation results for %s using %s templates"%(self.name, self.typ_mod))
        hdul.writeto(save_path+'%s-cc_%s_%s.fits'%(self.name, self.typ_mod, suffix), overwrite=True) 
        
    def read_cc(self, filename):
        """Read cross-correlation results""" 
        
        _, temp_type, temp_model, _ = re.split(r'\s|_', filename)
        typ_mod = temp_type + "_" + temp_model
        self.typ_mod = typ_mod
        print("Read cross-correlation result... Template:%s"%(typ_mod))
        
        hdu_cc = fits.open(filename)
        self.cc_nums = hdu_cc[0].data
        
        self.CC_Rs = hdu_cc[1].data
        inds_best = np.argmax(self.CC_Rs, axis=1)
        
        self.CC_SNRs = hdu_cc[2].data
        self.SNR_best = np.array([self.CC_SNRs[i, inds_best[i]] for i in range(len(self.cc_nums))])
        
        self.CC_SNR_ps = hdu_cc[3].data
        self.SNRp_best = np.array([self.CC_SNR_ps[i, inds_best[i]] for i in range(len(self.cc_nums))])
        
        self.CC_zccs = hdu_cc[4].data
        self.z_best = np.array([self.CC_zccs[i, inds_best[i]] for i in range(len(self.cc_nums))])
        
        self.line_stddev_best = self.Stddev_Temps[self.typ_mod][inds_best]
        
        if np.ndim(self.Line_Ratios_Temps[self.typ_mod])==2:
            self.line_ratio_best = (self.Line_Ratios_Temps[self.typ_mod].max(axis=1)/3.)[inds_best]
        else:
            self.line_ratio_best = np.ones_like(self.cc_nums)*10
        
#         CCs = hdu_cc[5].data
#         self.CC_best = np.array([CCs[i, inds_best[i]] for i in range(len(self.cc_nums))])
        
        self.CC_zccs_Temps[typ_mod] = self.CC_zccs
        self.CC_Rs_Temps[typ_mod] = self.CC_Rs
        self.CC_SNRs_Temps[typ_mod] = self.CC_SNRs
        self.CC_SNR_ps_Temps[typ_mod] = self.CC_SNR_ps
        try:
            self.CC_flag_edge = hdu_cc[5].data
            self.CC_flag_edge_Temps[typ_mod] = self.CC_flag_edge
        except IndexError:
            pass       
        
    
    def read_cluster_boundary(self, filepath):
        """Read Cluster boundary map and BCG position, only for double cluster"""
        self.boundary_map = fits.open(filepath)[0].data 
        
    def assign_BCG_position(self, id_BCG):
        """Assign BCG position (x, y) in pixel from id_BCG, ((x1,y1),(x2,y2)) for double cluster"""
        if id_BCG is not None:
            if np.ndim(id_BCG)==0:
                self.pos_BCG = (self.table[id_BCG]["X_IMAGE"], self.table[id_BCG]["Y_IMAGE"])
            elif np.ndim(id_BCG)==1:
                pos_BCG1 = (self.table[id_BCG[0]]["X_IMAGE"], self.table[id_BCG[0]]["Y_IMAGE"])
                pos_BCG2 = (self.table[id_BCG[1]]["X_IMAGE"], self.table[id_BCG[1]]["Y_IMAGE"])
                self.pos_BCG = (pos_BCG1,pos_BCG2)
            return self.pos_BCG
        else:
            print("BCG is not in the field. Assign BCG position mannually: Datacube.pos_BCG=(x,y) (in pix)")
            
    def assign_BCG_coordinate(self, coord_BCG):
        """Assign BCG position (ra, dec) in deg from id_BCG, ((ra1,dec1),(ra2,dec2)) for double cluster""" 
        # WCS
        self.wcs = WCS(self.hdu[0].header, naxis=2)
        if np.ndim(coord_BCG)==1:
            self.coord_BCG = SkyCoord(coord_BCG[0], coord_BCG[1], frame='icrs', unit="deg")
            print("BCG coordinate: ", self.coord_BCG.to_string('hmsdms'))
        elif np.ndim(coord_BCG)==2:
            coord_BCG1 = SkyCoord(coord_BCG[0][0], coord_BCG[0][1], frame='icrs', unit="deg")
            coord_BCG2 = SkyCoord(coord_BCG[1][0], coord_BCG[1][1], frame='icrs', unit="deg")
            self.coord_BCG = (coord_BCG1, coord_BCG2)
            print("BCG1 coordinate: ", coord_BCG1.to_string('hmsdms'))
            print("BCG2 coordinate: ", coord_BCG2.to_string('hmsdms'))
        
        
    def centroid_analysis(self, num, z=None, k_wid=6, 
                          centroid_type="APER", coord_type="angular", sum_type="weight",
                          emission_type="subtract", aperture_type="separate",
                          fix_window=False, multi_aper=True, 
                          n_rand=199, aper_size=[0.7,0.8,0.9,1.1,1.2,1.],
                          plot=True, print_out=True, return_for_plot=False):
        """Centroid analysis for one candidate.
        
        Parameters
        ----------
        num : number of object in the SE catalog
        z : candidate redshift
        k_wid : half-width of the thumbnail (note: not to be too small)
        centroid_type : method of computing centroid position for emission and continua
            "APER" : centroid computing within a fixed aperture, aperture from iterative photometry on the combined image
            "ISO2" : centroid computing within isophotes (connected pixel map), isophotes from the combined image
        coord_type : method of computing cluster-centric vector 
            "angular" : use wcs coordinate (TAN-SIP corrected)
            "euclid" : use pixel position
        fix_window : whether to fix the emission window. True for non-candidate.
        n_rand : number of random start 
        aper_size : apertures used in measuring centroid (in a_image)
        
        Returns
        ----------
        d_angle : angle difference between emission-to-continuum vector and cluster-centric vector
        offset : centroid offset of emission and contiuum distribution (in pixel)
        dist_clus : distance to cluster center (in pixel)
        """ 
        
        ind = (self.obj_nums==num)
        
        obj_SE = Object_SE(self.table[ind], k_wid=k_wid, cube=self.datacube,
                           img_seg=self.img_seg, deep_frame=self.deep_frame, mask_field=self.mask_edge)
        self.obj_SE=obj_SE

        spec = self.obj_specs_opt[ind][0]

        if centroid_type == "APER":
            k_aper =  self.k_apers_opt[ind][0]
        else:
            k_aper = None
        
        if z is None:
            z = self.z_best[ind]
        
        try: 
            boundary_map = self.boundary_map
        except AttributeError:
            boundary_map = None
            
        try:
            snr = self.SNR_best[ind]
            if fix_window:
                lstd, lr = 3., 10.
            else:
                lstd, lr = self.line_stddev_best[ind], self.line_ratio_best[ind]

            if plot:
                deep_img = obj_SE.deep_thumb
            else:
                deep_img = None
            d_angle, offset, dist_clus, \
            pa, clus_cen_angle = compute_centroid_offset(obj_SE, spec=spec, wavl=self.wavl, 
                                                         k_aper=k_aper, z_cc=z,
                                                         pos_BCG = self.pos_BCG,
                                                         coord_BCG = self.coord_BCG, 
                                                         wcs = self.wcs, 
                                                         centroid_type=centroid_type, 
                                                         coord_type=coord_type, 
                                                         line_stddev=lstd,
                                                         line_ratio=lr,
                                                         affil_map=boundary_map, 
                                                         deep_img=deep_img, 
                                                         sum_type=sum_type,
                                                         emission_type=emission_type,
                                                         aperture_type=aperture_type,
                                                         multi_aper=multi_aper,
                                                         n_rand=n_rand, aper_size=aper_size,
                                                         plot=plot, print_out=print_out,
                                                         return_for_plot=return_for_plot)
            return (d_angle, offset, dist_clus, pa, clus_cen_angle)
        
        except (ValueError, TypeError) as error:
            if print_out:
                print("Unable to compute centroid! Error raised.")
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
    
    def centroid_analysis_all(self, Num_v, k_wid=6, 
                              centroid_type="APER", coord_type="angular", aperture_type="separate", 
                              n_rand=199, aper_size = [0.7,0.8,0.9,1.1,1.2,1.],
                              plot=False, print_out=True):
        """Compute centroid offset and angle for all SE detections
        
        Parameters
        ----------
        Num_v : number list of target candiate in the SE catalog
        k_wid : half-width of the thumbnail (note: not to be too small)
        centroid_type : method of computing centroid position for emission and continua
            "APER" : centroid computing within a fixed aperture, aperture from iterative photometry on the combined image
            "ISO2" : centroid computing within isophotes (connected pixel map), isophotes from the combined image
        coord_type : method of computing cluster-centric vector 
            "angular" : use wcs coordinate (TAN-SIP corrected)
            "euclid" : use pixel position
        """
        
        self.centroid_type = centroid_type
        self.diff_angles = np.array([]) 
        self.diff_centroids = np.array([])
        self.dist_clus_cens = np.array([])
        self.PAs = np.array([])
        self.clus_cen_angles = np.array([])
        self.max_devs = np.array([])
        if print_out:
            print("Current Model: ", self.typ_mod)
        for num in self.obj_nums:
            if print_out:
                print("#EL-%1d"%num)
            ind = (self.obj_nums==num)

            if num in Num_v:  # For emission galaxies, pick a window 70A aorund emission
                z = self.z_best[ind]
                fix_w = False
                emission_type = "subtract"
#                 sum_type = "mean"
                sum_type = "weight"
#                 sum_type = "median"
                multi_aper = True
                
            else:  # For non-emission galaxies, pick a random window excluding the edge 
                z = np.random.uniform(low=(self.wavl[0]+25)/6563.-1, 
                                      high=(self.wavl[-1]-25)/6563.-1)
                fix_w = True
                emission_type = "narrowband"
                sum_type = "median"
                multi_aper = False
                
            d_angle, offset, dist_clus, \
            pa, clus_cen_angle = self.centroid_analysis(num, z=z, k_wid=k_wid, 
                                                        centroid_type=centroid_type, 
                                                        coord_type=coord_type,
                                                        fix_window=fix_w, 
                                                        emission_type=emission_type,
                                                        aperture_type=aperture_type,
                                                        sum_type=sum_type,
                                                        multi_aper=multi_aper,
                                                        n_rand=n_rand, aper_size=aper_size,
                                                        plot=False, print_out=print_out)
            if num in Num_v:
                if print_out:
                    print("Angle: %.3f  Offset: %.3f"%(d_angle, offset))    
            self.diff_angles = np.append(self.diff_angles, d_angle)
            self.diff_centroids = np.append(self.diff_centroids, offset)
            self.dist_clus_cens = np.append(self.dist_clus_cens, dist_clus)
            self.PAs = np.append(self.PAs, pa)
            self.clus_cen_angles = np.append(self.clus_cen_angles, clus_cen_angle)
        
    def construct_control(self, Num_v, mag_cut=None, mag0=25.2, dist_cut=25, bootstraped=False, n_boot=100):
        """Construct control sample for comparison
        
        Parameters
        ----------
        Num_v : number list to be excluded in the control sample in the SE catalog
        mag_cut: magnitude range (lower, upper)
        dist_cut : minium distance (in pixel) to the field edge
        
        """
        # remove nan values that fail in centroid analysis
        Num_not_v = np.setdiff1d(self.table["NUMBER"], Num_v)
        ind_not_v = Num_not_v - 1
        Num_c_all = np.setdiff1d(self.table["NUMBER"][ind_not_v], 
                                 1 + np.where(np.isnan(self.diff_centroids))[0])
        
        # magnitude cut
        mag = -2.5*np.log10(self.table["FLUX_AUTO"])+mag0
        mag_no_nan = mag[~np.isnan(mag)]
        
        if mag_cut is None:
            mag_cut = (mag.min(),mag.max())
            
        Num_mag_ctrl = self.table["NUMBER"][(mag>mag_cut[0])&(mag<mag_cut[1])]
        Num_c = np.intersect1d(Num_c_all, Num_mag_ctrl)
        
        # area cut
        thre_iso_area = 10.
        print("Isophotal area threshold: ", thre_iso_area)
        Num_area_ctrl = self.table["NUMBER"][self.table["ISOAREA_IMAGE"]>thre_iso_area]
        
        Num_c = np.intersect1d(Num_c, Num_area_ctrl)
        
#         # SNR cut
#         thre_snr = np.percentile(self.SNRp_best,90)
#         print("peak S/N threshold: ", thre_iso_area)
#         Num_snr_ctrl = self.table["NUMBER"][(self.SNRp_best<thre_snr)]
#         Num_c = np.intersect1d(Num_c, Num_snr_ctrl)
        
        # radius cut
        Num_rp_ctrl = self.table["NUMBER"][(self.table["PETRO_RADIUS"]>0)&(self.table["PETRO_RADIUS"]<10)]        
        Num_c = np.intersect1d(Num_c, Num_rp_ctrl)      
        
        # elongation cut < 2 sigma
        thre_elong = np.percentile(self.table["ELONGATION"],97.5)
        Num_elong_ctrl = self.table["NUMBER"][self.table["ELONGATION"]<thre_elong]
        Num_c = np.intersect1d(Num_c, Num_elong_ctrl)
        
        # edge cut
        Y_pix, X_pix = np.indices(self.mask_edge.shape)
        dist_edge = np.array([np.sqrt((X_pix[~self.mask_edge]-gal["X_IMAGE"])**2 + \
                              (Y_pix[~self.mask_edge]-gal["Y_IMAGE"])**2).min() \
                              for gal in self.table])
        self.dist_edge = dist_edge
        print("Distance to field edge threshold: ", dist_cut)
        Num_edge_ctrl = self.table["NUMBER"][dist_edge>dist_cut]
        Num_c = np.intersect1d(Num_c, Num_edge_ctrl)
        
        print('Control Sample : n=%d'%len(Num_c))
        self.Num_c = Num_c
        ind_c = Num_c - 1
        
        self.diff_angles_c = self.diff_angles[ind_c]
        self.diff_centroids_c = self.diff_centroids[ind_c]
        self.dist_clus_cens_c = self.dist_clus_cens[ind_c]
        self.PAs_c = self.PAs[ind_c]
        self.clus_cen_angles_c = self.clus_cen_angles[ind_c]
        
        if bootstraped:
            # Bootstrap control sample
            from astropy.stats import bootstrap
            Num_c_bp = bootstrap(Num_c, bootnum=100, samples=None).astype("int")
            self.diff_angles_c_bp = np.zeros_like(Num_c_bp)
            self.diff_centroids_c_bp = np.zeros_like(Num_c_bp)
            self.dist_clus_cens_c_bp = np.zeros_like(Num_c_bp)
            for i in range(n_boot):
                if np.mod(i+1,n_boot//5)==0: print("Bootsrap sample: %d"%(i+1))
                ind_c_bp = Num_c_bp[i] - 1
                self.diff_angles_c_bp[i] = self.diff_angles[ind_c_bp]
                self.diff_centroids_c_bp[i] = self.diff_centroids[ind_c_bp]
                self.dist_clus_cens_c_bp[i] = self.dist_clus_cens[ind_c_bp]
                
    def measure_EW_Ha_all(self, Num_v):
        for num in self.obj_nums:
            ind = (self.obj_nums==num)

            if num in Num_v:  # For emission galaxies, pick a window 70A aorund emission
                z_best = self.z_best[ind]
                spec = self.obj_specs_opt[ind][0]
                lstd = self.line_stddev_best[ind]
                EW_Ha = measure_EW(spec, self.wavl, z=z_best, line_stddev = lstd)
            else:  
                EW_Ha = np.nan
                
            self.EW_Ha = np.append(self.EW_Ha, EW_Ha) if num>1 else [EW_Ha]
        