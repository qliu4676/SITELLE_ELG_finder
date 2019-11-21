import re
import os
import glob
import time
import pprint as pp

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table, Column, join, hstack
from astropy.stats import mad_std
from astroquery.vizier import Vizier

from utils import *


def Read_SITELLE_DeepFrame(file_path, name, SE_catalog=None, seg_map=None, mask_edge=None):
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
        tab_sdss = crossmatch_sdss12(self.RA, self.DEC, radius=radius, band='rmag', mag_max=mag_max)
        return tab_sdss
    
    def make_mask_edge(self, save_path = './'):
        """ Mask edges """
        check_save_path(save_path)
        hdu_edge = fits.PrimaryHDU(data=self.mask_edge.astype("float"))
        hdu_edge.writeto(os.path.join(save_path, '%s_DF_edge.fits'%self.name), overwrite=True)
    
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
        hdu_weight_map.writeto(os.path.join(save_path, 'weight_map_DF_%s.fits'%self.name), overwrite=True)
    
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
        back, back_rms = background_sub_SE(field, mask=mask, b_size=b_size, f_size=3, maxiters=20)
        field_sub = field - back
        
        if display:
            display_background_sub(field, back, vmax=1e3)
        if plot:
            check_save_path(save_path, clear=True)
            hdu_new = fits.PrimaryHDU(data=field_sub, header=self.header)
            hdu_new.writeto(os.path.join(save_path, '%s_DF%s.fits'%(self.name, suffix)), overwrite=True)
            print("Saved background subtracted image as %s_DF%s.fits"%(self.name, suffix))
        
        self.field_sub = field_sub    
        
    def calculate_seeing(self, tab_SDSS=None, R_pix=15, cut=[95,99.5], sep=1 * u.arcsec, sigma_guess=1., plot=False):
        """ 
        Use central SDSS stars or brigh star-like objects (class_star>0.7) to calculate 
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
            
        if tab_SDSS is None:
            # Cut in CLASS_STAR, roundness and brightness
            star_cond = (table["CLASS_STAR"]>0.7) & (table["PETRO_RADIUS"]>0) \
                        & (table["B_IMAGE"]/table["A_IMAGE"]>0.8) & (table["FLAGS"] <4)
            tab_star = table[star_cond]
            
            F_limit = np.percentile(tab_star["FLUX_AUTO"], cut)
            tab_star = tab_star[(tab_star["FLUX_AUTO"]>F_limit[0])&(tab_star["FLUX_AUTO"]<F_limit[1])]
        else:
            pos_SE = np.vstack([table["X_IMAGE"], table["Y_IMAGE"]]).T
            coord_SE = self.wcs.all_pix2world(pos_SE, 1)
            c_SE = SkyCoord(ra=coord_SE[:,0], dec=coord_SE[:,1], unit="deg")

            c_sdss = SkyCoord(ra=tab_SDSS['RA_ICRS'], dec=tab_SDSS['DE_ICRS'])
            idx, d2d, d3d = c_SE.match_to_catalog_sky(c_sdss)
            tab_star = table[d2d < sep]
            
            tab_star = tab_star[(tab_star["B_IMAGE"]/tab_star["A_IMAGE"]>0.8)]
            
            
        FWHM = calculate_seeing(tab_star, self.image, self.seg_map, 
                                R_pix=R_pix, sigma_guess=sigma_guess, min_num=5, plot=True)
        self.seeing_fwhm = np.median(FWHM)
        print("Median seeing FWHM in arcsec: %.3f"%self.seeing_fwhm)
        
        

def Read_Raw_SITELLE_datacube(file_path, name, wavn_range=[12100, 12550], wavl_mask=None):
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
    raw_datacube = Raw_Datacube(file_path, name, wavn_range=wavn_range, wavl_mask=wavl_mask)
    return raw_datacube

class Raw_Datacube: 
    """ A class for raw SITTLE datacube """
    
    def __init__(self, file_path, name, wavn_range, wavl_mask=None):
        self.hdu = fits.open(file_path)
        self.header = self.hdu[0].header
        self.name = name
        
        # Pixel scale, Step of wavenumber, Start of wavenumber, Number of steps
        if "PIXSCAL1" in self.header:
            self.pix_scale = self.header["PIXSCAL1"]   #arcsec/pixel
        self.d_wavn = self.header["CDELT3"]        #cm^-1 / pixel
        self.wavn_start = self.header["CRVAL3"]    #cm^-1
        self.nstep = self.header["STEPNB"] 
        
        # Wavenumber axis
        self.wavn = np.linspace(self.wavn_start, self.wavn_start+(self.nstep-1)*self.d_wavn, self.nstep)  #cm^-1
        
        # Effective wave number range
        wavn_eff_range = (self.wavn>wavn_range[0]) & (self.wavn<wavn_range[1])
        self.wavl = 1./self.wavn[wavn_eff_range][::-1]*1e8
        
        # Raw effective datacube and stack field
        self.raw_cube = self.hdu[0].data[wavn_eff_range][::-1]
        self.cube_process = self.raw_cube.copy()
        self.shape = self.raw_cube.shape
        self.raw_stack_field = self.raw_cube.sum(axis=0)
        
        # Field edge/saturations to be masked
        self.mask_edge = self.raw_stack_field < 5 * mad_std(self.raw_stack_field)
        
        # Masked channels making stack
        if wavl_mask is not None:
            _, mask_wl = self.get_channel(wavl_mask)
            print(wavl_mask," will be masked in making maximum map.")
        else:
            mask_wl = np.zeros(self.shape[0], dtype=bool)
        self.wavl_mask = wavl_mask
        self.mask_wl = mask_wl
        
        # Modify the fits for saving the new cube
        self.header_new = self.header.copy()
        self.header_new["CRVAL3"] = self.wavn[wavn_eff_range][0]  # New minimum wavenumber in cm-1
        self.header_new["STEPNB"] = np.sum(wavn_eff_range)   # New step
        
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
        hdu_edge.writeto(os.path.join(save_path, 'Raw_stack_%s_mask.fits'%self.name),overwrite=True)
        
    def save_weight_map(self, region_path, weight=0.001, save_path = './'):
        # Mask regions (eg. star spikes) by making a weight map for SE input
        import pyregion
        reg = pyregion.open(region_path)
        self.mymask = reg.get_mask(hdu=self.hdu[0], shape=self.hdu[0].shape[1:])

        weight_map =(~self.mymask).astype("float") + weight
        self.weight_map = weight_map
        
        check_save_path(save_path)
        hdu_weight_map = fits.PrimaryHDU(data=weight_map)
        hdu_weight_map.writeto(os.path.join(save_path, 'weight_map_stack_%s.fits'%self.name), overwrite=True)
    
    @property
    def stack_field(self):
        return self.cube_process[~self.mask_wl].sum(axis=0)
        
    def remove_background(self, box_size=128, filter_size=3, maxiters=10, save_path='./bkg/', plot=False):
        """ Remove background (low-frequency component) using SEXtractor estimator (filtering + mode) """
        if plot:
            check_save_path(save_path, clear=True)
        
        self.datacube_bkg_sub = np.empty_like(self.raw_cube)
        for i, field in enumerate(self.raw_cube):
            if np.mod(i+1, 10)==0: print("Removing background... Channel: %d"%(i+1))
            
            back, back_rms = background_sub_SE(field, mask=self.mask_edge,
                                               b_size=box_size, f_size=filter_size, maxiters=maxiters)
            field_sub = field - back

            self.datacube_bkg_sub[i] = field_sub
            
            # Save background subtraction
            if plot:
                fig = display_background_sub(field, back, vmax=1)
                plt.savefig(os.path.join(save_path, "bkg_sub_channel%d.png"%(i+1)),dpi=100)
                plt.close(fig)  
                
        self.cube_process = self.datacube_bkg_sub.copy()
    
    def get_channel(self, wavl_intp_range=None):
        if wavl_intp_range is None:
            wavl_intp_range =  [self.wavl[0], self.wavl[-1]]
            
        mask = np.zeros_like(self.wavl, dtype=bool)
        for w_range in np.atleast_2d(wavl_intp_range):
            mask[(w_range[0]<=self.wavl) & (self.wavl<=w_range[1])] = True
        channel = np.where(mask)[0] + 1
        return channel, mask
            
    def interp_bad_channel(self, wavl_intp_range=[7980,7995], interp=False):
        """ Interpolate bad channels, e.g. suffering from strong fringes, from adjacent channels """
        cube = self.datacube_bkg_sub.copy()
        bad_channel, bad = self.get_channel(wavl_intp_range=wavl_intp_range)
        if len(bad_channel)>0:
            if not interp:
                return bad_channel

            print("Interpolate bad channels:", ", ".join((bad_channel).astype(str)))
            good_channel = np.where(~bad)[0] + 1

            for i in range(self.shape[1]):
                for j in range(self.shape[2]):
                    y_good = cube[:,i,j][~bad]
                    cube[bad,i,j] = np.interp(bad_channel, good_channel, cube[:,i,j][~bad])
                                            #\+ np.random.normal(scale=mad_std(y_good),size=len(bad_channel))
            self.cube_process = cube
            return cube
        else:
            print("No bad channles.")
            self.cube_process = cube
            return None
    
        
    def remove_fringe(self, channels=None, sn_source=3, n_iter=3,
                      k_size=(12,3), box_size=16, filter_size=1, maxiters=10,
                      method='SE', save_path='./bkg/',
                      verbose=False, parallel=False, clear=True, plot=False):
        """ Clean (fringe and artifacts) for specified channels """
        if channels is None:
            return None
        
        if plot:
            check_save_path(save_path, clear=clear)
            
        print("Run iteration to clean fringes for channels: \n", channels)
        if len(channels)>20: parallel = True
        inds = np.atleast_1d(channels)-1

        kernel = Gaussian2DKernel(k_size[0],k_size[1])
        
        stack_field = self.datacube_bkg_sub.sum(axis=0) if method=="SE" else self.stack_field
        n_source = 0
        
        if (method=="LPF") & parallel:
            from parallel import parallel_compute
            from functools import partial
            if verbose:
                print("'\nRun low-pass filtering in parallel. Used when run on >20 channels.")

        for n in range(n_iter):
            # In every iteration, detect, count and mask sources
            # The cube does not change but the stack field is inherited from previous iteration
            # Final fringe (small-scale background) subtraction depends on the final stack field
            if verbose:
                print("Iteration %d : %d sources detected."%(n+1, n_source))
                
            self.cube_process = self.datacube_bkg_sub.copy()
            
            mask_source, segmap = make_mask_map(stack_field, sn_thre=sn_source, b_size=128)
            mask = (self.mask_edge) | (mask_source)
            
            n_source_new = segmap.max()
            stable_source_SE = (abs(n_source-n_source_new)/n_source_new < 0.05)
            
            # if < 5% new sources are detected, or reach max_iter, break
            break_cond = (stable_source_SE) | (n+1==n_iter)
            
            if (method=="LPF") & parallel:
                
                p_convolve2D_cube = partial(convolve2D_cube,
                                            cube=self.cube_process.copy(),
                                            kernel=kernel, mask=mask)
                backs = parallel_compute(inds, p_convolve2D_cube,
                                         lengthy_computation=True, verbose=False)
                for i, ind in enumerate(inds):
                    backs[i][self.mask_edge] = 0
                    self.cube_process[ind] -= backs[i]
                    
            else:
                backs = np.zeros_like(self.cube_process)
                for i, ind in enumerate(inds):
                    field = self.cube_process[ind].copy()
                    
                    if method=="LPF":
                        backs[i] = convolve(field, kernel, mask=mask, normalize_kernel=True)
                        backs[i][self.mask_edge] = 0

                    elif method=="SE":    
                        backs[i], _ = background_sub_SE(field, mask=mask,
                                                        b_size=box_size, f_size=filter_size, maxiters=20)

                    self.cube_process[ind] = field - backs[i]

            if break_cond :
                if verbose: print("\nIteration finished.")
                # Save fringe subtraction
                if plot:
                    for i, ind in enumerate(inds):
                        fig = display_background_sub(self.datacube_bkg_sub[ind], backs[i], vmax=1)
                        plt.savefig(os.path.join(save_path, "fg_sub_channel%d.png"%(i+1)),dpi=80)
                        plt.close(fig)
                break
            
            # new stack field and # of sources
            n_source = n_source_new
            stack_field = self.stack_field
            
        self.datacube_bkg_fg_sub = self.cube_process.copy()
                    
    def save_fits(self, save_path = './', suffix=""):
        """Write processed datacube and stacked Image subtraction"""
        print("Saving processed datacube and stacked field...")
        
        check_save_path(save_path)
        
        hdu_cube = fits.PrimaryHDU(data = self.cube_process, header=self.header_new)
        hdu_cube.writeto(os.path.join(save_path, '%s_cube%s.fits'%(self.name, suffix)),overwrite=True)
        
        self.header_new_stack = self.header_new.copy()
        for keyname in ["CTYPE3","CRVAL3","CUNIT3","CRPIX3","CDELT3","CROTA3"]:
            try:
                self.header_new_stack.remove(keyname)
            except KeyError:
                pass
        hdu_stack = fits.PrimaryHDU(data = self.stack_field, header=self.header_new_stack)
        hdu_stack.writeto(os.path.join(save_path, '%s_stack%s.fits'%(self.name, suffix)),overwrite=True)
        
        
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
                 cube_supplementary=None, wavl_mask=None,
                 table=None, seg_map=None, mask_edge=None, 
                 deep_frame=None, z0=None):
        
        self.name = name
        self.z0 = z0
        self.hdu = fits.open(cube_path)
        self.header = self.hdu[0].header
        self.RA = self.header["TARGETR"]
        self.DEC = self.header["TARGETD"]
        
        self.cube = self.hdu[0].data
        self.shape = self.cube.shape
        self.mode = mode
        
        if (mode=="MMA"):            
            if table is not None:
                self.table = Table.read(table, format='ascii')
        elif mode=="APER":
            # Read SE measurement. 
            self.table = Table.read(table, format="ascii.sextractor")
                         
        else:
            raise ValueError("Choose an extraction mode: 'MMA' or 'APER' ('APER' requires SExtractor output)")
        
        try:
            self.seg_map = fits.getdata(seg_map)
        except ValueError:
            self.seg_map = None
            
        if cube_supplementary is not None:
            self.cube_sup = fits.getdata(cube_supplementary)
        
        if mask_edge is not None:
            self.mask_edge = fits.getdata(mask_edge).astype('bool')
            
        if deep_frame is not None:
            self.deep_frame = fits.getdata(deep_frame)
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
                
        # Masked channels making stack
        if wavl_mask is not None:
            _, mask_wl = self.get_channel(wavl_mask)
            print(wavl_mask," are affected by edge / fringes.")
        else:
            mask_wl = np.zeros(self.shape[0], dtype=bool)
        self.mask_wl = mask_wl
        self.wavl_mask = wavl_mask
        
        # For generating different templates 
        self.Temp_Lib = {}
        self.wavl_temps = {}
        self.Stddev_Temps = {}
        self.Line_Ratios_Temps = {}
        
        self.CC_result_Temps  = {}
        self.result_centroid = {}
    
    def get_wcs(self, hide=True):
        """ Read WCS info from the datacube header """
        if hide:
            with HiddenPrints():
                self.wcs = WCS(self.header,naxis=2)
        else:
            self.wcs = WCS(self.header,naxis=2)
        return self.wcs
    
    @property
    def stack_field(self):
        return self.cube[~self.mask_wl].sum(axis=0)

    def get_centroid(self, num, Xname="xcentroid", Yname="ycentroid"):
        """ Get centorids of the cutout for a detection """
        return get_centroid(num, self.table, Xname=Xname, Yname=Yname)
    
    def get_bounds(self, num, cen_pos=None,
                   Rname='equivalent_radius',
                   origin=0, **kwargs):
        """ Get bounds of the cutout for a detection """
        return get_bounds(num, self.table, self.stack_field.shape, **kwargs)
    
    def get_cutout(self, num, image=None,
                   bounds=None, cen_pos=None,
                   origin=0, **kwargs):
        """ Get cutout of the input image for a detection """
        if image is None:
            image = self.stack_field
        return get_cutout(num, self.table, image, **kwargs) 
    
    def get_channel(self, wavl_intp_range=None):
        if wavl_intp_range is None:
            wavl_intp_range =  [self.wavl[0], self.wavl[-1]]
        mask = np.zeros_like(self.wavl, dtype=bool)
        for w_range in np.atleast_2d(wavl_intp_range):
            mask[(w_range[0]<=self.wavl) & (self.wavl<=w_range[1])] = True
        channel = np.where(mask)[0] + 1
        return channel, mask
    
    def ISO_source_detection(self, sn_thre=3, npixels=8, b_size=128,
                             nlevels=64, contrast=0.01, closing=False,
                             add_columns=['equivalent_radius'],
                             box=[3,3,3], seeing=1, mask_map=None,
                             parallel=False, save=True, save_path = './', suffix=""):
        """ Source Extraction based on Isophotal of S/N """
        from skimage import morphology
        from photutils import source_properties
        
        # Pixel-wise manipulation in wavelength
        
        mask_edge_cube = np.repeat(self.mask_edge[np.newaxis, :, :], self.shape[0], axis=0)

        mask_wl = self.mask_wl
        
        # datacube used to detect source, use supplementray cube if given
        cube_detect = getattr(self, 'cube_sup', self.cube.copy())
        
        if self.mode == "MMA":
            print("Use the map of maximum of moving average (MMA) to detect source.")
            print("Box shape: ",box)
            
            if parallel:
                print("Run moving average in parallel. This could be slower using few cores.")
                
                from parallel import parallel_compute
                from functools import partial
                
                
                p_moving_average_cube = partial(moving_average_by_col, cube=cube_detect)
                results = parallel_compute(np.arange(self.shape[2]), p_moving_average_cube,
                                           lengthy_computation=False, verbose=True)
                src_map = np.max(results,axis=2).T
                
            else:
                cube_MA = moving_average_cube(cube_detect, box=box, mask=mask_edge_cube)
                src_map = np.max(cube_MA[~mask_wl], axis=0)
        elif self.mode == "stack":
            src_map = self.stack_field
        
        # Detect + deblend source
        print("Detecting and deblending source...")
        mask_source, _ = make_mask_map(self.stack_field, sn_thre=sn_thre, b_size=b_size)
#         if mask_map is not None:
#             mask_weight = fits.getdata(mask_map) < 1e-3
        mask = dilation(self.mask_edge) | (mask_source)
        
        back, back_rms = background_sub_SE(src_map, mask=mask, b_size=b_size, f_size=1)
        threshold = back + (sn_thre * back_rms)

        segm0 = detect_sources(src_map, threshold, npixels=npixels)
        segm = deblend_sources(src_map, segm0, npixels=npixels, nlevels=nlevels, contrast=contrast)
        
        if closing is True:
            seg_map = morphology.closing(segm.data)
        else:
            seg_map = segm.data
        
        # Save max map and segmentation map
        if save:
            check_save_path(save_path)
            src_map_name = '%s_%s%s.fits'%(self.name, self.mode, suffix)
            seg_map_name = '%s_segm_%s%s.fits'%(self.name, self.mode, suffix)

            hdu_src = fits.PrimaryHDU(data=src_map, header=self.header)
            hdu_src.writeto(os.path.join(save_path, src_map_name), overwrite=True)
            hdu_seg = fits.PrimaryHDU(data=seg_map, header=None)
            hdu_seg.writeto(os.path.join(save_path, seg_map_name), overwrite=True)
        
        # Meausure properties of detections
        cat = source_properties(src_map, segm)
        columns = ['id', 'xcentroid', 'ycentroid', 'ellipticity', 'area', 'orientation'] + add_columns
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
            tab_name = os.path.join(save_path,'%s_%s%s.dat'%(self.name, self.mode, suffix))
            tab.write(tab_name, format='ascii', overwrite=True)
            print("Finish. Saved as %s"%tab_name)
        
        self.table = tab
        self.src_map = src_map
 
        return src_map, seg_map
    
    def ISO_spec_extraction_all(self, seg_map):
        labels = np.unique(seg_map)[1:]
        cube_detect = getattr(self, 'cube_sup', self.cube.copy())
        
        for k, lab in enumerate(labels):
            if np.mod(k+1, 400)==0: print("Extract spectra... %d/%d"%(k+1, len(labels)))
            spec_opt = self.cube[:, seg_map==lab].sum(axis=1)
            self.obj_specs_opt = np.vstack([self.obj_specs_opt, spec_opt]) if k>0 else [spec_opt]
            spec_det = cube_detect[:, seg_map==lab].sum(axis=1)
            self.obj_specs_det = np.vstack([self.obj_specs_det, spec_det]) if k>0 else [spec_det]   
        
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
                        verbose=False, plot=False, display=False):
        """
        Extract spectal using an optimal aperture, local background evaluated from (k1, k2) annulus
        
        Parameters
        ----------
        num : SE object number
        ks : a set of aperture ring (in SE petrosian radius) to determined the optimized aperture
        k1, k2 : inner/outer radius (in SE petrosian radius) of annulus for evaluate background RMS
        verbose : print out the optimized k and S/N
        plot : whether to plot the S/N vs k curve
        display: whether to display the image thumbnail with apertures
        
        """
        
        id = np.where(self.table["NUMBER"]==num)[0][0]
        obj_SE = Obj_detection(self.table[id], cube=self.cube, deep_frame=self.deep_frame,
                               seg_map=self.seg_map, mask_edge=self.mask_edge)
        
        if obj_SE.R_petro==0:
            if verbose: print("Error in measuring R_petro, dubious source")
            return (np.zeros_like(self.wavl), 1., 0, None)
        
        k, snr = obj_SE.compute_aper_opt(ks=ks, k1=k1, k2=k2,
                                                 ext_type=ext_type, verbose=verbose, plot=plot)
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
                                                       verbose=False, plot=False, display=False)
            self.obj_specs_cen = np.vstack((self.obj_specs_cen, spec_cen)) if len(self.obj_nums)>0 else [spec_cen]
            self.obj_aper_cen = np.append(self.obj_aper_cen, k_cen)
            
            spec_opt, k_opt, snr_opt, apers_opt = self.spec_extraction(num=num, ext_type='opt', 
                                                                       ks=ks, k1=k1, k2=k2,
                                                                       verbose=False, plot=False, display=display)
            self.obj_specs_opt = np.vstack((self.obj_specs_opt, spec_opt)) if len(self.obj_nums)>0 else [spec_opt]
            self.obj_aper_opt = np.append(self.obj_aper_opt, k_opt)
                
            # Display the image marking the two apertures and annulus
            if display&(snr_opt>0):
                apers[0].plot(color='limegreen', lw=1.5, ls='--')
                plt.savefig(os.path.join(save_path, "SE#%d.png"%num),dpi=150)
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
            plt.savefig(os.path.join(save_path, "#%d.png"%n),dpi=100)
            plt.close()
            
    @property
    def spec_stack(self):
        return (self.obj_specs_opt).sum(axis=0)
    
    def fit_continuum_all(self, model='GP', save_path='./fit_cont/', plot=False, verbose=True,
                          edge_ratio=None, kernel_scale=100, kenel_noise=1e-3):   
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
            check_save_path(save_path, clear=True)
            
        
        if (model=='GP') & (edge_ratio is None):
            # Stacked spectra to estimate edge ratio
            spec_stack = self.spec_stack
            edge_ratio = 0.5*np.sum((spec_stack-np.median(spec_stack) <= -2.5*mad_std(spec_stack))) / len(self.wavl)
            if verbose:
                print('Fit continuum with GP. No edge_ratio is given. Use estimate = %.2f'%edge_ratio)
        
        spurious_nums = np.array([], dtype=int)
        for n in self.table["NUMBER"]:
            
            # Skip supurious detections (edges, spikes, etc.)
            iloc = (self.table["NUMBER"]==n)
            if self.mode!="APER":
                spurious_detection = (self.table[iloc]['ellipticity']>= 0.9)
            else:
                spurious_detection = (self.table[iloc]["PETRO_RADIUS"] <=0) | (self.table[iloc]["FLUX_AUTO"] <=0)
            
            blank_array = np.zeros(len(self.wavl)+1)
            random_array = np.random.rand(len(self.wavl)+1)
                
            if spurious_detection:
                if verbose:
                    spurious_nums = np.append(spurious_nums, n)
                res, cont_fit = (random_array, blank_array)
            else:    
                if verbose:
                    if np.mod(n, 400)==0: 
                        print("Fit spectra continuum ... %d/%d"%(n, len(self.table["NUMBER"])))
                try:   
                    # Fit continuum
                    spec = self.obj_specs_det[self.obj_nums==n][0]
                    res, wavl_rebin, cont_fit = fit_continuum(spec, self.wavl, model=model, 
                                                              edge_ratio=edge_ratio,
                                                              kernel_scale=kernel_scale, 
                                                              kenel_noise=kenel_noise,
                                                              verbose=False, plot=plot)
                    if plot:
                        plt.savefig(os.path.join(save_path, "#%d.png"%n),dpi=100)
                        plt.close()

                except Exception as e:
                    res, cont_fit = (random_array, blank_array)
                    if verbose:
                        print("Spectrum #%d continuum fit failed ... Skip"%n)
            
            self.obj_res_det = np.vstack((self.obj_res_det, res)) if n>1 else [res]
            self.obj_cont_fit = np.vstack((self.obj_cont_fit, cont_fit)) if n>1 else [cont_fit]
        if verbose:
            print("Skip spurious detection: ",
                  "#"+" #".join(spurious_nums.astype(str)),
                  " ... Replaced with random noise.")
        print("Continuum Fitting Finished!")
        self.wavl_rebin = wavl_rebin
        self.num_spurious = spurious_nums
       
            
    def save_spec_fits(self, save_path='./', suffix="_all"):  
        """Save extracted spectra and aperture (in the unit of SE pertrosian radius) as fits"""
        check_save_path(save_path)
        hdu_num = fits.PrimaryHDU(data=self.obj_nums)
        
        hdu_spec_opt = fits.ImageHDU(data=self.obj_specs_opt)
        hdu_spec_det = fits.ImageHDU(data=self.obj_specs_det)
        hdu_cont_fit = fits.ImageHDU(data=self.obj_cont_fit)
        
        hdu_res_det = fits.ImageHDU(data=self.obj_res_det)
        hdu_wavl_rebin = fits.ImageHDU(data=self.wavl_rebin)
        hdu_num_spurious = fits.ImageHDU(data=self.num_spurious)
        
        hdul = fits.HDUList(hdus=[hdu_num, hdu_spec_opt, hdu_spec_det, hdu_cont_fit,
                                  hdu_res_det, hdu_wavl_rebin, hdu_num_spurious])
        filename = '%s-spec-%s%s.fits'%(self.name, self.mode, suffix)
        hdul.writeto(os.path.join(save_path, filename), overwrite=True)
        
        
    def read_spec(self, file_path):
        """Read generated extracted spectra and aperture fits"""
        hdu_spec = fits.open(file_path)
        
        self.obj_nums = hdu_spec[0].data.copy()
        
#         self.obj_specs_cen = hdu_spec[1].data
#         self.k_apers_cen = hdu_spec[2].data['k_aper_cen']
        
        self.obj_specs_opt = hdu_spec[1].data.copy()
        self.obj_specs_det = hdu_spec[2].data.copy()
        self.obj_cont_fit = hdu_spec[3].data.copy()
        
        self.obj_res_det = hdu_spec[4].data.copy()
        self.wavl_rebin = hdu_spec[5].data.copy()
        self.num_spurious = hdu_spec[6].data.copy()
        
        
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
                      save_path='./temp/', suffix=""): 
        
        """Save Emission Line Template as Fits"""
        check_save_path(save_path)
        
        typ_mod = temp_type + "_" + temp_model
        
        hdu_temps  = fits.PrimaryHDU(self.Temp_Lib[typ_mod])
        hdu_wavl_temp = fits.ImageHDU(self.wavl_temp)
        hdu_stddev_temp = fits.ImageHDU(self.Stddev_Temps[typ_mod])
        hdu_line_ratio_temp = fits.ImageHDU(self.Line_Ratios_Temps[typ_mod])

        hdul = fits.HDUList(hdus=[hdu_temps, hdu_wavl_temp, hdu_stddev_temp, hdu_line_ratio_temp])
        
        filename = 'Template-%s_%s%s.fits'%(self.name, typ_mod, suffix)
        print("Save %s templates for %s as %s"%(self.name, typ_mod, filename))
        
        hdul.writeto(os.path.join(save_path, filename), overwrite=True) 

    def Read_Template(self, dir_name, n_intp=2, name='*'):
        """Read Emission Line Template from directory"""
        
        temp_list = glob.glob(dir_name+'/Template-%s_*.fits'%name)
        
        if len(temp_list)>0:
            self.n_intp = n_intp
            print("Read Emission Line Template:"), pp.pprint(temp_list)

            for path_name in temp_list:
                path, filename_w_ext = os.path.split(path_name)
                filename, _ = filename_w_ext.rsplit(".", 1)
                _, temp_type, temp_model = re.split(r'\_', filename)[:3]
                typ_mod = temp_type + "_" + temp_model

                hdul = fits.open(path_name)
                self.temps = hdul[0].data.copy()
                self.wavl_temp = hdul[1].data.copy()
                self.Stddevs = hdul[2].data.copy()
                self.Line_ratios = hdul[3].data.copy()

                self.Temp_Lib[typ_mod] = self.temps
                self.wavl_temps[typ_mod] = self.wavl_temp
                self.Stddev_Temps[typ_mod] = self.Stddevs
                self.Line_Ratios_Temps[typ_mod] = self.Line_ratios
        else:
            print('%s does not exist or does not have templates ("Template-*").'%dir_name)
            
    def cross_correlation(self, num, rv=None,
                          temp_type="Ha-NII", 
                          temp_model="gauss",
                          kind_intp="linear",
                          h_contrast=0.1,
                          edge=20, edge_pad=15,
                          const_window=False, 
                          verbose=True, plot=True,
                          fig=None, axes=None):
        
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
            if axes is None:
                fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(9,7))
        
        # Note spurious detections are skipped
        res = self.obj_res_det[self.obj_nums==num][0]  # Read residual spectra
        
        # Read the template information, and pass to cross-correlation
        typ_mod = temp_type + "_" + temp_model
        temps = self.Temp_Lib[typ_mod]
        wavl_temp = self.wavl_temps[typ_mod]
        stddevs = self.Stddev_Temps[typ_mod]
        line_ratios = self.Line_Ratios_Temps[typ_mod]
        
        # result_cc: ccs, rv, z_ccs, Rs, Contrasts, SNRs, SNR_ps, flag_edge
        result_cc = xcor_SNR(res=res, rv=rv, 
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
                                           'line_ratio':line_ratios},
                             h_contrast=h_contrast, 
                             edge=edge, edge_pad=edge_pad,
                             const_window=const_window, 
                             plot=plot, fig=fig, axes=axes)
        
        if plot: plt.tight_layout()
            
        if verbose:
            print ("Detection #%d  z: %.3f  sigma: %.3f  Peak R: %.3f  Detction S/N: %.3f Peak S/N: %.3f" \
                   %(num, result_cc["z_best"], result_cc["sigma_best"],
                     result_cc["R"], result_cc["SNR"], result_cc["SNR_p"]))
            
        return result_cc

    
    def cross_correlation_pipe(self, 
                               temp_type="Ha-NII", 
                               temp_model="gauss",
                               kind_intp="linear",
                               edge=20, edge_pad=15,
                               h_contrast=0.1,
                               const_window=False, 
                               cores=None, verbose=True):
        
        from parallel import parallel_compute
        from functools import partial
        
        CC_result = {}
        
        # Read the template information, and pass to cross-correlation
        typ_mod = temp_type + "_" + temp_model
        temps = self.Temp_Lib[typ_mod]
        wavl_temp = self.wavl_temps[typ_mod]
        stddevs = self.Stddev_Temps[typ_mod]
        line_ratios = self.Line_Ratios_Temps[typ_mod]
        
        result_cc = self.cross_correlation(num=1, rv=None, edge=edge, edge_pad=edge_pad, 
                                           temp_type=temp_type, temp_model=temp_model,
                                           const_window=const_window, kind_intp=kind_intp,
                                           h_contrast=h_contrast, verbose=False, plot=False)
        self.rv = result_cc['rv']
        result_cc.pop("rv", None)
        CC_result["1"] = result_cc
    
        p_xcor = partial(xcor_SNR, 
                         wavl_rebin=self.wavl_rebin, 
                         temps=temps, 
                         wavl_temp=wavl_temp, 
                         d_wavl=self.d_wavl, 
                         z_sys=self.z0, 
                         temp_type=temp_type, 
                         temp_model=temp_model,
                         temps_params={'stddev':stddevs, 
                                       'line_ratio':line_ratios},
                         rv=self.rv,
                         n_intp=self.n_intp, 
                         kind_intp=kind_intp,
                         edge=edge, edge_pad=edge_pad, 
                         h_contrast=h_contrast,
                         const_window=const_window,
                         plot=False)
        
        start = time.time()
        results_cc = parallel_compute(self.obj_res_det[1:], p_xcor, cores=cores,
                                      lengthy_computation=False, verbose=verbose)
        end = time.time()
        if verbose:
            print("Run in parallel. Total Time %.2fs"%(end-start))
        
        # Retrieve cc parallelly computed result
        for k, num in enumerate(self.obj_nums[1:].astype(str)):
            result_cc.pop("rv", None)
            CC_result[num] = results_cc[k]
        
        CC_result['rv'] = self.rv
            
        return CC_result
    
    def cross_correlation_all(self, 
                              temp_type="Ha-NII", 
                              temp_model="gauss",
                              kind_intp="linear",
                              h_contrast=0.1,
                              edge=20, edge_pad=15,
                              parallel=True, cores=None, 
                              const_window=False, verbose=True):
        """Cross-correlation for all SE detections in the catalog.
        
        Parameters
        ----------
        temp_type : template type ("Ha-NII" or "Hb-OIII" or "OII")
        temp_model : template line profile model ("gauss", "sincgauss" or "delta")
        kind_intp : scipy interp1d kind, default is "linear"
        h_contrast : threshold of CC peak finding, relative to the highest peak
        
        """ 
        
        n_obj = len(self.obj_nums)
        
        typ_mod = temp_type + "_" + temp_model
        self.typ_mod = typ_mod
        temps = self.Temp_Lib[typ_mod]
        
        # Initialize CC result as null dictionary
        self.CC_result = {}
        
        # Cross-correlation with the templates
        print("Do cross-correlation using %s model templates..."%typ_mod)
        
        if parallel:
            self.CC_result = self.cross_correlation_pipe(temp_type=temp_type, temp_model=temp_model,
                                                         cores=cores, verbose=verbose,
                                                         edge=edge, edge_pad=edge_pad,
                                                         const_window=const_window,
                                                         kind_intp=kind_intp, h_contrast=h_contrast)
        else:
            
            for j, num in enumerate(self.obj_nums):
                if verbose:
                    if np.mod(j, 400)==0: 
                        print("Cross-correlating spectra with templates ... %d/%d"%(j, len(self.obj_nums)))
                        
                if j==0:
                    # Compute relative velocity axis only at the first step 
                    result_cc = self.cross_correlation(num, rv=None, edge=edge, edge_pad=edge_pad,
                                                       temp_type=temp_type, temp_model=temp_model,
                                                       verbose=verbose, plot=False,
                                                       const_window=const_window,
                                                       kind_intp=kind_intp, h_contrast=h_contrast)
                else:
                    result_cc = self.cross_correlation(num, rv=self.rv, edge=edge, edge_pad=edge_pad,
                                                       temp_type=temp_type, temp_model=temp_model,
                                                       verbose=verbose, plot=False,
                                                       const_window=const_window,
                                                       kind_intp=kind_intp, h_contrast=h_contrast)
                result_cc.pop("rv", None)
                self.CC_result[num] = result_cc
                
            self.CC_result['rv'] = self.rv
            
        # Assign results of the specific template
        self.CC_result_Temps[typ_mod] = self.CC_result
        

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
        filename = '%s-cc_%s_%s.fits'%(self.name, self.typ_mod, suffix)
        hdul.writeto(os.path.join(save_path, filename), overwrite=True) 
       
    def read_cc_fits(self, filename):
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
    
    
    def save_cc_result(self, save_path='./', suffix=""): 
        """Save cross-correlation results as pkl"""
        import pickle
        check_save_path(save_path)
        filename = os.path.join(save_path, '%s-cc-%s%s'%(self.name, self.mode, suffix))
        
        with open(filename + '.pkl', 'wb') as file:
            pickle.dump(self.CC_result_Temps, file, pickle.HIGHEST_PROTOCOL)

        print("Save cross-correlation results for %s as : %s"%(self.name, filename + '.pkl'))
        print("Template used: "+str([key for key in self.CC_result_Temps.keys()]))
        
    def read_cc_result(self, filename):
        import pickle
        print("Read cross-correlation results for %s from : %s"%(self.name, filename))
        
        if os.path.isfile(filename): 
            with open(filename, 'rb') as f:
                self.CC_result_Temps = pickle.load(f)

            print("Template used: "+str([key for key in self.CC_result_Temps.keys()]))
            
        else:
            print("%s does not exist. Check file path."%filename)
        
        
    def dict_property(func=None): 
        def get_prop(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyError:
                print("Template / Property name does not exist.")
        return get_prop

    @dict_property
    def get_CC_result_best(self, prop, typ_mod, nums=None):
        """
        Get the best matched value of a property from CC results.
        
        prop: 'sigma_best', 'ratio_best', 'z_best', 'SNR' ...
        """
        
        if nums is None:
            nums = self.obj_nums
        return np.array([self.CC_result_Temps[typ_mod][num][prop]
                         for num in np.atleast_1d(nums).astype(str)])
            
    
    def estimate_EW(self, num, z=None, sigma=None, 
                    temp_type="Ha-NII", temp_model="gauss",
                    MC_err=True, n_MC=250, edge=10,
                    use_cont_fit=False, ax=None, plot=True):
        
        id = num-1
        typ_mod = temp_type + "_" + temp_model
        
        if temp_type=="Ha-NII":
            lam_0 = 6563.
        elif temp_type=="Hb-OIII":
            lam_0 = 5007.
        elif temp_type=="OII":
            lam_0 = 3727.

        if use_cont_fit:
            cont = np.median(np.interp(self.wavl, self.wavl_rebin, self.obj_cont_fit[id]))
        else:
            cont = None
            
        if sigma is None:
            if temp_model=="box":
                sigma = self.d_wavl
            else:
                sigma = self.CC_result_Temps[typ_mod][str(num)]['sigma_best']

#         ratio_best = self.CC_result_Temps[typ_mod][str(num)]['ratio_best']
#         if ratio_best.max()/3 < 3:
#             sigma *= 2
        
        if z is None:
            z = self.CC_result_Temps[typ_mod][str(num)]['z_best']
        
        EW, EW_std = estimate_EW(self.obj_specs_opt[id], self.wavl,
                                 z=z, lam_0=lam_0, sigma=sigma, 
                                 MC_err=MC_err, n_MC=n_MC,
                                 edge=edge, cont=cont, ax=ax, plot=plot)
        
        return EW, EW_std

    def estimate_EW_all(self, temp_type="Ha-NII", temp_model="gauss", sigma=5,
                        MC_err=True, n_MC=250, use_cont_fit=False, edge=10):
            
        typ_mod = temp_type + "_" + temp_model
       
        z_bests = self.get_CC_result_best('z_best', typ_mod)
        sigma_bests = self.get_CC_result_best('sigma_best', typ_mod)
        
        EWs = np.zeros(len(self.obj_nums))
        EW_stds = np.zeros(len(self.obj_nums))
        
        if sigma is None:
            sigma = sigma_bests[k]
        for k, num in enumerate(self.obj_nums):
            if np.mod(k+1, 400)==0: print("Measure EW... %d/%d"%(k+1, len(self.obj_nums)))
            EWs[k], EW_stds[k] = self.estimate_EW(num, z=z_bests[k], sigma=sigma,
                                                  temp_type=temp_type, temp_model=temp_model,
                                                  MC_err=MC_err, n_MC=n_MC, edge=edge,
                                                  use_cont_fit=use_cont_fit, plot=False)
        self.EWs = EWs
        self.EW_stds = EW_stds
        
    def match_sdss_star(self, sep=3*u.arcsec,
                        search_radius=7*u.arcmin, band='rmag', mag_max=18):
       
        # Retrieve SDSS star catalog of the target field
        tab_sdss = crossmatch_sdss12(self.RA, self.DEC,
                                     radius=search_radius, band=band, mag_max=mag_max)
        
        # Read WCS info from the datacube header
        wcs = self.get_wcs()

        c_star = SkyCoord(ra=tab_sdss["RA_ICRS"], dec=tab_sdss["DE_ICRS"])
        
        # Use wcs.all_world2pix to perform all transformations in series (core WCS, SIP and distortions) 
        # Note pixel cooordinate of centroids in photutils is 0-based!
        star_pos = np.array(c_star.to_pixel(wcs, origin=0)).T  
        
        obj_pos = np.array([self.table['xcentroid'], self.table['ycentroid']]).T
        coords_obj = wcs.all_pix2world(obj_pos, 0)  # 0-based position
        
        # Convert positions of detections to world coordinates
        c_obj = SkyCoord(ra=coords_obj[:,0], dec=coords_obj[:,1], frame="icrs", unit="deg")
    
        # Match the detection postions with the SDSS star catalog
        idx, d2d, d3d = c_star.match_to_catalog_sky(c_obj)
        match = d2d < sep
        cat_star_match = tab_sdss[match]
        cat_obj_match = self.table[idx[match]]
        
        # Combine the detection table with the SDSS star catalog
        cat_star_match.add_column(cat_obj_match["NUMBER"], index=0, name="NUMBER")
        cat_match = join(cat_obj_match, cat_star_match, keys='NUMBER')
        
        return cat_match
    
    def plot_candidate(self, num, temp_type="Ha-NII", temp_model="gauss", sigma=5, fig=None):
    
        from matplotlib.gridspec import GridSpec
        if fig is None:
            fig = plt.figure(figsize=(13,10))
        gs = GridSpec(3, 4, figure=fig)
        ax1 = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[1, :3])
        ax3 = fig.add_subplot(gs[2, :3])
        ax4,ax5,ax6 = [fig.add_subplot(gs[i, 3]) for i in range(3)]

        self.rv = self.CC_result_Temps[temp_type+"_"+temp_model]["rv"]
        
        EW, EW_std = self.estimate_EW(num, temp_type=temp_type, temp_model=temp_model, sigma=sigma, ax=ax1)

        result = self.cross_correlation(num, const_window=False, rv=self.rv,
                                        temp_type=temp_type, temp_model=temp_model,
                                        verbose=False, axes=(ax2,ax3))

        X_cen, Y_cen = self.get_centroid(num) 
        bounds = self.get_bounds(num, cen_pos=(X_cen, Y_cen)) 
        x_min, y_min, _, _ = bounds
        X_min, Y_min = coord_Array2Im(x_min, y_min, 0)

        if hasattr(self, "src_map"):
            cutout = self.get_cutout(num, self.src_map, bounds=bounds, cen_pos=(X_cen, Y_cen))
            ax4.imshow(cutout, norm=norm1, origin="lower", vmin=np.median(self.src_map), vmax=1)
        
        cutout_stack = self.get_cutout(num, self.stack_field, bounds=bounds, cen_pos=(X_cen, Y_cen))
        ax5.imshow(cutout_stack, norm=norm2, origin="lower", vmin=np.median(self.stack_field), vmax=1e2)
        
        if self.deep_frame is not None:
            cutout_deep = self.get_cutout(num, self.deep_frame, bounds=bounds, cen_pos=(X_cen, Y_cen))
            ax6.imshow(cutout_deep, norm=norm3, origin="lower", vmin=np.median(self.deep_frame), vmax=1e3)

        for ax, title in zip([ax4,ax5,ax6],["MMA", "Stack", "Deep"]):
            ax.plot(X_cen-X_min, Y_cen-Y_min, 'r+', ms=15)
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.yaxis.set_major_locator(plt.MaxNLocator(6))
            ax.set_xticklabels(ax.get_xticks().astype('int64')+X_min)
            ax.set_yticklabels(ax.get_yticks().astype('int64')+Y_min)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("X (pix)")
            ax.set_ylabel("Y (pix)")
            
        plt.tight_layout()

    
    def read_cluster_boundary(self, filepath):
        """Read Cluster boundary map and BCG position, only for double cluster"""
        self.boundary_map = fits.open(filepath)[0].data 
        
    def assign_BCG_position(self, id_BCG, xname='xcentroid', yname='ycentroid'):
        """Assign BCG position (x, y) in pixel from id_BCG, ((x1,y1),(x2,y2)) for double cluster"""
        if id_BCG is not None:
            if np.ndim(id_BCG)==0:
                self.pos_BCG = (self.table[id_BCG][xname], self.table[id_BCG][yname])
            elif np.ndim(id_BCG)==1:
                pos_BCG1 = (self.table[id_BCG[0]][xname], self.table[id_BCG[0]][yname])
                pos_BCG2 = (self.table[id_BCG[1]][xname], self.table[id_BCG[1]][yname])
                self.pos_BCG = (pos_BCG1,pos_BCG2)
            return self.pos_BCG
        else:
            print("BCG is not in the field. Assign BCG position mannually: Datacube.pos_BCG=(x,y) (in pix)")
            
    def assign_BCG_coordinate(self, coord_BCG):
        """Assign BCG position (ra, dec) in deg from id_BCG, ((ra1,dec1),(ra2,dec2)) for double cluster""" 
        # WCS
        if np.ndim(coord_BCG)==1:
            self.coord_BCG = SkyCoord(coord_BCG[0], coord_BCG[1], frame='icrs', unit="deg")
            print("BCG coordinate: ", self.coord_BCG.to_string('hmsdms'))
        elif np.ndim(coord_BCG)==2:
            coord_BCG1 = SkyCoord(coord_BCG[0][0], coord_BCG[0][1], frame='icrs', unit="deg")
            coord_BCG2 = SkyCoord(coord_BCG[1][0], coord_BCG[1][1], frame='icrs', unit="deg")
            self.coord_BCG = (coord_BCG1, coord_BCG2)
            print("BCG1 coordinate: ", coord_BCG1.to_string('hmsdms'))
            print("BCG2 coordinate: ", coord_BCG2.to_string('hmsdms'))
        
        
    def centroid_analysis(self, num, z=None,
                          centroid_type="APER", sum_type="weighted",
                          subtract_continuum=True, from_SE=False,
                          line_width=None, multi_aper=True, **kwargs):
        """Centroid analysis for one candidate.
        
        Parameters
        ----------
        num : number of object in the SE catalog
        z : candidate redshift
        centroid_type : method of computing centroid position for emission and continua
            "APER" : centroid computing within a fixed aperture, aperture from iterative photometry on the combined image
            "ISO-D" : centroid computing within isophotes (connected pixel map), isophotes from the combined image
        line_width : width of emission window. Fixed for non-candidate.
        
        Returns
        ----------
        d_angle : angle difference between emission-to-continuum vector and cluster-centric vector
        offset : centroid offset of emission and contiuum distribution (in pixel)
        dist_clus : distance to cluster center (in pixel)
        """ 
        
        ind = np.where(self.obj_nums==num)[0][0]
        
        obj = Obj_detection(self.table[ind], cube=self.cube, seg_map=dilation(self.seg_map),
                            deep_frame=self.deep_frame, mask_edge=self.mask_edge)

        spec = self.obj_specs_opt[ind]

        if from_SE is True:
            k_aper =  self.k_apers_opt[ind][0]
        else:
            k_aper = None
        
        if z is None:
            z = self.get_CC_result_best('z_best', 'Ha-NII_gauss', nums=num)[0]
        
        boundary_map = getattr(self, 'boundary_map', None)
        
        if line_width is None:
            line_width = self.get_CC_result_best('sigma_best', 'Ha-NII_gauss', nums=num)[0]
#         try:
        affil_map = getattr(self, 'boundary_map', None)
        res_measure = compute_centroid_offset(obj, spec=spec, wavl=self.wavl,
                                              z_cc=z, wcs=self.wcs,
                                              coord_BCG=self.coord_BCG,
                                              centroid_type=centroid_type,
                                              sum_type=sum_type, affil_map=affil_map,
                                              k_aper=k_aper, multi_aper=multi_aper,
                                              line_stddev=line_width, 
                                              subtract=subtract_continuum, **kwargs)
            
        return res_measure

    
    def centroid_analysis_all(self, Num_V, nums_obj=None,
                              centroid_type="APER", sum_type="weighted", subtract=True,
                              plot=False, verbose=True, smooth=False, morph_cen=False, **kwargs):
        """Compute centroid offset and angle for all SE detections
        
        Parameters
        ----------
        Num_V : number list of target candiate in the SE catalog
        centroid_type : method of computing centroid position for emission and continua
            "APER" : centroid computing within a fixed aperture, aperture from iterative photometry on the combined image
            "ISO" : centroid computing within isophotes (connected pixel map), isophotes from the combined image
        """
        
        self.centroid_type = centroid_type
        
        self.z_best = self.get_CC_result_best('z_best', 'Ha-NII_gauss')
        self.sigma_best = self.get_CC_result_best('sigma_best', 'Ha-NII_gauss')
        
        key = centroid_type
        if morph_cen: key += 'm'
        if smooth: key += 's'
        
        self.result_centroid[key] = {}
        
        nums_obj = self.obj_nums if nums_obj is None else np.atleast_1d(nums_obj)
        
        for num in nums_obj:
            if verbose:
                print("\nCandidate: #%d"%num)
            ind = np.where(self.obj_nums==num)[0][0]

            if num in Num_V:  # Candidate
                z = self.z_best[ind]
                line_width = self.sigma_best[ind]
                subtract = subtract
                multi_aper = True
                
            else:  # For non-emission galaxies, pick a random window excluding the edge 
                z = np.random.uniform(low=(self.wavl[0]+25)/6563.-1, 
                                      high=(self.wavl[-1]-25)/6563.-1)
                use_good = False
                line_width = 3
                subtract = False
                multi_aper = False
                
            res_measure = self.centroid_analysis(num, z=z,
                                                 centroid_type=centroid_type, 
                                                 line_width=line_width,
                                                 subtract_continuum=subtract,
                                                 sum_type=sum_type,
                                                 multi_aper=multi_aper,
                                                 plot=False, verbose=verbose,
                                                 smooth=smooth, morph_cen=morph_cen, **kwargs)
            if (num in Num_V) & (len(res_measure)>0):
                if verbose:
                    print("Angle: %.2f +/- %.2f"%(res_measure["diff_angle"],res_measure["diff_angle_std"]))
                    print("Offset: %.2f +/- %.2f"%(res_measure["cen_offset"], res_measure["cen_offset_std"]))    
            
            self.result_centroid[key]["%s"%num] = res_measure
    
    @property
    def centroid_result_names(self):
        return ['diff_angle', 'cen_offset', 'diff_angle_std', 'cen_offset_std',
                'pa', 'clus_cen_angle', 'dist_clus_cen']
    
    @dict_property        
    def get_centroid_result(self, prop, centroid_type="APER", nums=None, fill_value=0):
        """
        Get the best matched value of a property from centroid analysis results.
        
        prop: 'diff_angle', 'cen_offset', 'diff_angle_std', 'cen_offset_std', ...
        """
        if prop in self.centroid_result_names:
            result = self.result_centroid[centroid_type]
            if nums is None:
                nums = result.keys()
            else:
                nums = nums.astype(str)
            return np.array([result[num].get(prop, fill_value) for num in nums])
    
    def save_centroid_measurement(self, Num_v, save_path='./', suffix="", ID_field=""):
        if hasattr(self, 'result_centroid'):
            check_save_path(save_path)
            
            ID = [str(num) + ID_field for num in Num_v]
            pos = np.array([self.get_centroid(num) for num in Num_v])
            
            coords = np.around(self.wcs.all_pix2world(pos, 0),4)
            
            z_best =  np.around(self.get_CC_result_best('z_best', 'Ha-NII_gauss', Num_v), 4)
            SN_Ha  = np.around(self.get_CC_result_best('SNR', 'Ha-NII_gauss', Num_v), 3)
            SN_OIII = np.around(self.get_CC_result_best('SNR', 'Hb-OIII_gauss', Num_v), 3)
            SN_OII = np.around(self.get_CC_result_best('SNR', 'OII_gauss', Num_v), 3)
            
            tab = Table(np.vstack([ID, coords[:,0], coords[:,1], z_best,
                                   pos[:,0], pos[:,1], SN_Ha, SN_OIII, SN_OII]).T,
                        names=["ID","ra","dec","z","X","Y",'SN_Ha', 'SN_OIII', 'SN_OII'])
            
            for k, centroid_type in enumerate(self.result_centroid.keys()):
                for prop_name in self.centroid_result_names:
                    common_prop = prop_name in ['clus_cen_angle', 'dist_clus_cen'] 
                    if common_prop & (k<len(self.result_centroid.keys())-1):
                        continue
                    else:
                        fval = np.nan #99 if 'std' in prop_name else 0
                        prop = self.get_centroid_result(prop_name, centroid_type, fill_value=fval)
                        colname = prop_name if common_prop else prop_name+'_'+centroid_type
                        tab[colname] = np.around(prop, 3)
                    
            fname = os.path.join(save_path, 'centroid_analysis_%s%s.txt'%(self.name, suffix))
            tab.write(fname, format='ascii', delimiter=' ', overwrite=True)

            print("Save centroid measurement as catalog: ", fname)
    
            
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

