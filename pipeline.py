import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.stats import mad_std
import scipy.stats as stats
import re

from utils import *

class Read_Raw_Datacube: 
    def __init__(self, file_path, name, wavn_range=[12100, 12550]):
        self.hdu = fits.open(file_path)
        self.header = self.hdu[0].header
        self.name = name
        
        # Pixel scale, Step of wavenumber, Start of wavenumber, Number of steps
#             self.pix_scale = self.hdu[0].header["PIXSCAL1"] #arcsec/pixel
        self.d_wavn = self.hdu[0].header["CDELT3"] #cm^-1 / pixel
        self.wavn_start = self.hdu[0].header["CRVAL3"] #cm^-1
        self.nstep = self.hdu[0].header["STEPNB"] 
        
        # Total wavenumber
        self.wavn = np.linspace(self.wavn_start, self.wavn_start+(self.nstep-1)*self.d_wavn, self.nstep)  #cm^-1
        
        # Effective data range from 12100 cm^-1 to 12550 cm^-1 
        wavn_eff_range = (self.wavn>wavn_range[0]) & (self.wavn<wavn_range[1])
        self.wavl = 1./self.wavn[wavn_eff_range][::-1]*1e8
        
        # Raw effective datacube and stack field in wavelength
        self.raw_datacube = self.hdu[0].data[wavn_eff_range][::-1]
        self.raw_stack_field = self.raw_datacube.sum(axis=0)
        
        # Field edge/saturations to be masked
        self.mask_edge = self.raw_stack_field < 3*mad_std(self.raw_stack_field)

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
        hdu_edge = fits.PrimaryHDU(data=self.mask_edge.astype("float")*10)
        hdu_edge.writeto(save_path + 'Raw_stack_%s_mask.fits'%self.name,overwrite=True)
        
    def save_weight_map(self, region_path, weight=0.01, save_path = './'):
        # Mask regions (eg. star spikes) by making a weight map for SE input
        import pyregion
        reg = pyregion.open(region_path)
        self.mymask = reg.get_mask(hdu=self.hdu[0], shape=self.hdu[0].shape[1:])

        weight_map =(~self.mymask).astype("float") + weight
        hdu_weight_map = fits.PrimaryHDU(data=weight_map)
        hdu_weight_map.writeto(save_path + 'Weight_map_stack_%s.fits'%self.name, overwrite=True)
        
    def remove_background(self, box_size=64, filter_size=3, n_iter=5, save_path = None, plot=False):
        # Remove background (low-frequency component) using SEXtractor estimator (filtering + mode)
        self.datacube_bkg_sub = np.empty_like(self.raw_datacube)
        for i, field in enumerate(self.raw_datacube):
            if np.mod(i+1, 10)==0: print("Removing background...Frame: %d"%(i+1))
            
            field_sub, back = background_sub_SE(field, mask=self.mask_edge, 
                                                b_size=box_size, f_size=filter_size, n_iter=n_iter)
            self.datacube_bkg_sub[i] = field_sub
            
            # Save background subtraction
            if plot:
                fig = display_background_sub(field, back)
                plt.savefig(save_path + "bkg_sub_channel%d.pdf"%(i+1),dpi=100)
                plt.close(fig)  
                
        self.stack_field = self.datacube_bkg_sub.sum(axis=0)
        
        # Possible Sources to be masked
        mask_source = self.stack_field > 3*mad_std(self.stack_field)
        self.mask_source = np.logical_or(mask_source, self.mask_edge)

    def remove_fringe(self, skip_frames=[None], box_size=8, filter_size=1, n_iter=20, save_path = None, plot=False):
        # Remove fringe and artifacts (high-frequency component)
        self.datacube_bkg_fg_sub = np.empty_like(self.datacube_bkg_sub)
        for i, field in enumerate(self.datacube_bkg_sub):
            if np.mod(i+1, 10)==0: print("Removing fringe...Frame: %d"%(i+1))
                
            if i in skip_frames:
                self.datacube_bkg_fg_sub[i] = field
            else:
                field_sub, back = background_sub_SE(field, mask=self.mask_source, 
                                                    b_size=box_size, f_size=filter_size, n_iter=n_iter)
                self.datacube_bkg_fg_sub[i] = field_sub
                
                # Save fringe subtraction
                if plot:
                    fig = display_background_sub(field, back)
                    plt.savefig(save_path + "fg_sub_channel%d.png"%(i+1),dpi=100)
                    plt.close(fig)  
                    
        self.stack_field = self.datacube_bkg_fg_sub.sum(axis=0)
                    
    def save_fits(self, save_path = './', suffix=""):
        """Write Stacked Image and Datacube after background & fringe subtraction"""
        print("Saving background & fringe subtracted datacube and stacked field...")
        
        hdu_cube = fits.PrimaryHDU(data = self.datacube_bkg_fg_sub, header=self.hdu_header_new)
        hdu_cube.writeto(save_path + '%s_cube%s.fits'%(self.name, suffix),overwrite=True)
        
        for keyname in ["CTYPE3","CRVAL3","CUNIT3","CRPIX3","CDELT3","CROTA3"]:
            try:
                self.hdu_header_new.remove(keyname)
            except KeyError:
                pass
        hdu_stack = fits.PrimaryHDU(data = self.stack_field, header=self.hdu_header_new)
        hdu_stack.writeto(save_path + '%s_stack%s.fits'%(self.name, suffix),overwrite=True)
        
class Read_Datacube:
    def __init__(self, cube_path, name, SE_catalog, deep_frame=None, z0=None):
        self.hdu = fits.open(cube_path)
        self.header = self.hdu[0].header
        self.datacube = self.hdu[0].data
        self.name = name
        self.z0 = z0
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
        wavn_eff_range = (self.wavn>12100) & (self.wavn<12550)
        self.wavl = 1./self.wavn[wavn_eff_range][::-1]*1e8
        self.d_wavl = np.diff(self.wavl).mean()
        
        # If run SE twice, read the second run SE catalog. Format should be ascii.sextractor
        self.Tab_SE = Table.read(SE_catalog, format="ascii.sextractor")
        
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
        
        
    def read_mask(self, file_path):
        # Open edge mask
        hdu_mask = fits.open(file_path)
        self.mask_edge = (hdu_mask[0].data == 0)
        
    def read_seg(self, file_path):
        # Open SE segmentation mask
        hdu_seg = fits.open(file_path)
        self.img_seg = hdu_seg[0].data
        
    def calculate_seeing(self, R_pix = 10, mag_cut=[0.2,0.8], sigma_guess=1., plot=False):
        """ Use the brigh star-like objects (class_star>0.7) to calculate 
            median seeing fwhm by fitting gaussian profiles.
            
        Parameters
        ----------
        R_pix : max range of profiles to be fitted (in pixel)
        magcut : percentage of cut in isophotal magnitude
        sigma_guess : initial guess of sigma of gaussian profile (in pixel) 
        plot : plot radial profile for each object in use
        """
        
        from scipy.optimize import curve_fit
        def gaussian_func(x, a, sigma):
            # Define a gaussian function with offset
            return a * np.exp(-x**2/(2*sigma**2))

        # Cut in CLASS_STAR, measurable, roundness and mag threshold
        mag_all = -2.5*np.log10(self.Tab_SE["FLUX_ISO"])
        mag0 = [np.percentile(mag_all, mag_cut[0]*100), np.percentile(mag_all, mag_cut[1]*100)]
        star_cond = (self.Tab_SE["CLASS_STAR"]>0.7) & (self.Tab_SE["PETRO_RADIUS"]>0) \
                    & (self.Tab_SE["B_IMAGE"]/self.Tab_SE["A_IMAGE"]>0.7) \
                    & (mag_all>mag0[0]) &(mag_all<mag0[1]) \
                    & (np.sqrt(self.Tab_SE["A_IMAGE"]**2+self.Tab_SE["B_IMAGE"]**2)<self.Tab_SE["KRON_RADIUS"])
        
        self.Tab_star = self.Tab_SE[star_cond]
        if len(self.Tab_star)<5:
            print("Mag cut too low. No enough target stars.")
        else:
            FWHMs = np.array([])
            for star in self.Tab_star:

                x_c, y_c = coord_ImtoArray(star["X_IMAGE"], star["Y_IMAGE"])
                x_min, x_max = (x_c-R_pix).astype("int"), (x_c+R_pix).astype("int")
                y_min, y_max = (y_c-R_pix).astype("int"), (y_c+R_pix).astype("int")
                if (np.min([x_min, y_min])<=0) | (np.max([x_max, y_max])>=np.min(self.datacube.shape[1:])): # ignore edge
                    continue

                cube_thumb = self.datacube[:,x_min:x_max, y_min:y_max]
                img_thumb = cube_thumb.sum(axis=0)

                x, y = np.indices((img_thumb.shape))
                rr = np.sqrt((x - (x_c-x_min))**2 + (y - (y_c-y_min))**2)
                x, y = rr.ravel()/rr.max(), img_thumb.ravel()/img_thumb.ravel().max()

                # Fit the profile with gaussian and return seeing FWHM in arcsec
                initial_guess = [1,sigma_guess/rr.max()]
                popt, pcov = curve_fit(gaussian_func, x, y, p0=initial_guess)
                sig_pix = abs(popt[1])*rr.max()  # sigma of seeing in pixel
                FWHM = 2.355*sig_pix*0.322
                FWHMs = np.append(FWHMs, FWHM)
                self.seeing_fwhm = np.median(FWHMs)    

                if plot:
                    xplot = np.linspace(0,1,1000)
                    plt.scatter(x, y)
                    plt.plot(xplot, gaussian_func(xplot,*popt),"k")
                    plt.show() 
                    plt.close()
                    print(FWHM)  # seeing FWHM in arcsec

            print("Median seeing FWHM in arcsec: %.3f"%self.seeing_fwhm)
        
    def spec_extraction(self, num, ext_type='opt', 
                        ks = np.arange(1.,4.5,0.2), k1=5., k2=8.,
                        print_out=False, plot=False, display=False):
        """Extract spectal using an optimal aperture, local background evaluated from (k1, k2) annulus
        
        Parameters
        
        num : SE object number
        ks : a set of aperture ring (in SE petrosian radius) to determined the optimized aperture
        k1, k2 : inner/outer radius (in SE petrosian radius) of annulus for evaluate background RMS
        print_out : print out the optimized k and S/N
        plot : whether to plot the S/N vs k curve
        display: whether to display the image thumbnail with apertures
        
        """
        
        tab = self.Tab_SE[self.Tab_SE["NUMBER"]==num]
        obj_SE = Object_SE(tab, cube=self.datacube, deep_frame=self.deep_frame,
                           img_seg=self.img_seg, mask_field=self.mask_edge)
        
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
        for num in self.Tab_SE["NUMBER"]:
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
        for num, spec, spec_opt in zip(self.obj_nums, self.obj_specs_cen, self.obj_specs_opt):
            plt.plot(self.wavl, spec/spec.max(), color='gray', label="Aperture cen", alpha=0.7)
            plt.plot(self.wavl, spec_opt/spec_opt.max(), color='k', label="Aperture opt", alpha=0.9)
            plt.xlabel("wavelength",fontsize=12)
            plt.ylabel("normed flux",fontsize=12)
            plt.legend(fontsize=12)
            plt.savefig(save_path+"SE#%d.png"%num,dpi=150)
            plt.close()
            
            
    def fit_continuum(self, spec, model='GP', edge_ratio=0.15, kernel_scale=100, kenel_noise=1e-3, plot=True):
        """Fit continuum for the extracted spectrum
        
        Parameters
        ----------
        spec : extracted spectrum
        model : continuum model used to subtract from spectrum
            "GP": flexible non-parametric continuum fitting
                edge_ratio : filter band edge threshold (above which the edge is replaced with a median to avoid egde issue). 
                             If None, use the meadian edge ratio determined from spectra of all detections.
                kernel_scale : kernel scale of the RBF kernel
                kenel_noise : noise level of the white kernel
            "Trapz1d": 1D Trapezoid model with constant continuum and edge fitted by a constant slope
            
        Returns
        ----------
        res : residual spectra
        wavl_rebin : rebinned wavl in log linear
        """
        
        res, wavl_rebin, cont_fit = fit_cont(spec, self.wavl, 
                                               model=model, edge_ratio=edge_ratio,
                                               kernel_scale=kernel_scale,  kenel_noise=kenel_noise, 
                                               print_out=False, plot=plot)
            
        return res, wavl_rebin, cont_fit   
            
    def fit_continuum_all(self, model='GP', edge_ratio=None,
                          kernel_scale=100, kenel_noise=1e-3, 
                          plot=False, save_path=None):   
        """Fit continuum for all the detections
        
        Parameters
        ----------
        model : continuum model used to subtract from spectrum
            "GP": flexible non-parametric continuum fitting
                edge_ratio : filter band edge threshold (above which the edge is replaced with a median to avoid egde issue). 
                             If None, use the meadian edge ratio determined from spectra of all detections.
                kernel_scale : kernel scale of the RBF kernel
                kenel_noise : noise level of the white kernel
            "Trapz1d": 1D Trapezoid model with constant continuum and edge fitted by a constant slope
        """
        
        if edge_ratio is None:
            # Stacked spectra to estimate edge ratio
            self.spec_stack = (self.obj_specs_opt/self.obj_specs_opt.max(axis=1).reshape(-1,1)).sum(axis=0)
            self.edge_ratio_stack = np.sum((self.spec_stack - np.median(self.spec_stack) \
                                                    <= -2.5*mad_std(self.spec_stack))) / len(self.wavl)
            edge_ratio = 0.5*self.edge_ratio_stack
            
        for num in self.Tab_SE["NUMBER"]:
            r_petro = self.Tab_SE[self.Tab_SE["NUMBER"]==num]["PETRO_RADIUS"]
            if r_petro==0:
                blank_array = np.zeros(len(self.wavl)+1)
                res, wavl_rebin, cont_fit = (blank_array,blank_array,blank_array)
            else:    
                spec = self.obj_specs_opt[self.obj_nums==num][0]
                res, wavl_rebin, cont_fit = self.fit_continuum(spec, 
                                                     model=model, edge_ratio=edge_ratio,
                                                     kernel_scale=kernel_scale,  kenel_noise=kenel_noise, 
                                                     plot=plot)
            self.obj_res_opt = np.vstack((self.obj_res_opt, res)) if num>1 else [res]
            self.obj_cont_fit_opt = np.vstack((self.obj_res_opt, cont_fit)) if num>1 else [cont_fit]
            if plot:
                plt.savefig(save_path+"SE#%d.png"%(num),dpi=100)
                plt.close()
            print ("#%d spectra continuum fitted"%(num))
            
        self.wavl_rebin = wavl_rebin
       
            
    def save_spec_fits(self, save_path='./', suffix=""):  
        """Save extracted spectra and aperture (in the unit of SE pertrosian radius) as fits"""
        hdu_num = fits.PrimaryHDU(data=self.obj_nums)
        
        hdu_spec_cen = fits.ImageHDU(data=self.obj_specs_cen)
        hdu_aper_cen = fits.BinTableHDU.from_columns([fits.Column(name="k_aper_cen",array=self.obj_aper_cen,format="E")])
        
        hdu_spec_opt = fits.ImageHDU(data=self.obj_specs_opt)
        hdu_aper_opt = fits.BinTableHDU.from_columns([fits.Column(name="k_aper_opt",array=self.obj_aper_opt,format="E")])
        
        hdu_res_opt = fits.ImageHDU(data=self.obj_res_opt)
        hdu_wavl_rebin = fits.ImageHDU(data=self.wavl_rebin)
        
        hdu_cont_fit_opt = fits.ImageHDU(data=self.obj_cont_fit_opt)
        
        hdul = fits.HDUList(hdus=[hdu_num, hdu_spec_cen, hdu_aper_cen, hdu_spec_opt, hdu_aper_opt, 
                                  hdu_res_opt, hdu_wavl_rebin, hdu_cont_fit_opt])
        hdul.writeto(save_path+'%s-spec%s.fits'%(self.name, suffix), overwrite=True)
        
        
    def read_spec(self, file_path):
        """Read generated extracted spectra and aperture fits"""
        hdu_spec = fits.open(file_path)
        
        self.obj_nums = hdu_spec[0].data
        
        self.obj_specs_cen = hdu_spec[1].data
        self.k_apers_cen = hdu_spec[2].data['k_aper_cen']
        
        self.obj_specs_opt = hdu_spec[3].data
        self.k_apers_opt = hdu_spec[4].data['k_aper_opt']
        
        self.obj_res_opt = hdu_spec[5].data
        self.wavl_rebin = hdu_spec[6].data
#         self.obj_cont_fit_opt = hdu_spec[7].data
        
        
    def generate_template(self, n_ratio=50, n_stddev=10, n_intp=2,
                          temp_type="Ha-NII", 
                          temp_model="gauss", 
                          ratio_prior="uniform", 
                          z_type="cluster",
                          ratio_range = (1., 8), sigma_max = 300,
                          box_width=4, plot=False):
        
        """Generate template(s) for cross-correlation.
        
        Parameters
        ----------
        temp_type : template type ("Ha-NII" or "Hb-OIII" or "OII")
        temp_model : template line profile model ("gauss", "sincgauss" or "delta")
        n_stddev/n_ratio : number of template=n_stddev*n_ratio, if >1 generate a template library 
        n_intp : time of interpolation (controling the smoothness of CC function), 1 means no interpolation
        
        ratio_prior : template line ratios prior type ("sdss": sdss line ratio from MPA-JHU catalog, "uniform": uniform grid in log)  
        ratio_range : range of line ratio (lower, upper) (only for the template library with "uniform" prior)
        sigma_max : upper range of broadening in km/s
        
        # For Gaussian models, fwhm(=2.355*sigma) between spectral resolution to galaxy broadening v=300km/s.
        # sigma_v = (v)/sqrt(3), sigma_gal = sigma_v/3e5 * 6563*(1+z_cluster) 
        
        # For delta function width=box_width"""

        
        self.n_intp = n_intp
        
        box_wid = np.max([box_width, self.d_wavl])
        
        if temp_type=="OII":
            number = n_stddev
        else:
            number = n_ratio * n_stddev
        print("Template: %s_%s  Total Number: %d"%(temp_type, temp_model, number))
        self.num_temp = number
        
        sig_gal_m = sigma_max/np.sqrt(3)/3e5 * 6563*(1+self.z0)
        if temp_model=="gauss":
            sig_min = self.d_wavl/2.355
        elif temp_model=="sincgauss":
            sig_min = 0.1
        sig_max = np.sqrt(sig_min**2 + sig_gal_m**2)

        if ratio_prior=="sdss":
            self.Stddevs = stats.uniform.rvs(size=number) * (sig_max-sig_min) + sig_min
            dist_line_ratio = np.loadtxt("./%s/SDSS-DR7_line_ratio.txt"%self.name)
            line_ratios = np.random.choice(dist_line_ratio, size=number)   # note: Ha_NII6584 ratios
#                 line_ratios = stats.lognorm.rvs(0.64, loc=1.81, scale=1.24, size=number)

        elif ratio_prior=="uniform":    
            stddevs = np.linspace(sig_min, sig_max, n_stddev)
            if temp_type=="OII":    # Single peaks
                self.Stddevs = stddevs
            else:    # Multiple peaks
                self.Stddevs = np.repeat(stddevs, n_ratio).reshape(n_ratio, n_stddev).T.ravel()
#                     line_ratio = np.linspace(ratio_range[0], ratio_range[1])[::-1]  # line ratio from high to low
                line_ratio = np.logspace(np.log10(ratio_range[0]), np.log10(ratio_range[1]))[::-1]  # line ratio from high to low
                line_ratios = np.repeat(line_ratio, n_stddev).ravel()

        if z_type=="mid":
            z_ref = (self.wavl[0]+self.wavl[-1])/2./6563.-1  # line located at the middle of the filter
        elif z_type=="cluster":
            z_ref = self.z0

        if temp_type=="Ha-NII":
            self.Line_ratios = np.vstack([[1, 3*ratio, 3] for ratio in line_ratios])
            z_temp = 6563*(1+z_ref)/6563.-1

        elif temp_type=="Hb-OIII":        
            self.Line_ratios = np.vstack([[3, 3*ratio] for ratio in line_ratios])
            z_temp = 6563*(1+z_ref)/5007.-1

        elif temp_type=="OII":

            self.Line_ratios = np.vstack([[1] for k in range(number)])
            z_temp = 6563*(1+z_ref)/3727.-1

        n_temp = int(len(self.wavl)) * n_intp +1
        temps = np.empty((number, n_temp))

        for i, (lr, stddev) in enumerate(zip(self.Line_ratios, self.Stddevs)):
            temps[i], self.wavl_temp = generate_template(self.wavl, z=z_temp, 
                                                         temp_model=temp_model, temp_type=temp_type,
                                                         line_ratio=lr, sigma=stddev, box_wid=box_wid, 
                                                         n_intp=n_intp,  plot=plot)
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
        hdul.writeto(save_path+'Template-%s_%s_%s.fits'%(self.name, typ_mod, suffix), overwrite=True) 

    def Read_Template(self, filename, n_intp=2):
        
        """Read Emission Line Template""" 
                      
        _, temp_type, temp_model, _ = re.split(r'\s|_', filename)
                      
        typ_mod = temp_type + "_" + temp_model
        self.typ_mod = typ_mod
        self.n_intp = n_intp
                      
        print("Read Emission Line Template: %s"%(typ_mod))              
        
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
                          edge=30,
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
            fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1,figsize=(9,7))
        else:
            fig, (ax1,ax2) = None, (None, None)
        
        res = self.obj_res_opt[self.obj_nums==num][0]  # Read residual spectra
        
        # Read the template information, and pass to cross-correlation
        temps = self.Temp_Lib[temp_type + "_" + temp_model]
        wavl_temp = self.wavl_temps[temp_type + "_" + temp_model]
        stddevs = self.Stddev_Temps[temp_type + "_" + temp_model]
        line_ratios = self.Line_Ratios_Temps[temp_type + "_" + temp_model]
        
        ccs, rv, z_ccs, Rs, Contrasts, SNRs, SNR_ps, flag_edge = xcor_SNR(res=res, wavl_rebin=self.wavl_rebin, 
                                                                           temps=temps, wavl_temp=wavl_temp, 
                                                                           d_wavl=self.d_wavl, z_sys=self.z0, 
                                                                           n_intp=self.n_intp, kind_intp=kind_intp,                                                                 
                                                                           temp_type=temp_type, temp_model=temp_model,
                                                                           temps_stddev=stddevs, temps_ratio=line_ratios, 
                                                                           h_contrast=h_contrast, rv=rv, edge=edge,
                                                                           plot=plot, fig=fig, axes=(ax1,ax2))
        if plot: plt.tight_layout()
            
        r_max = np.argmax(Rs)    
        z_best = z_ccs[r_max]
        print ("SE Object #%d  z: %.3f  Peak R: %.3f  Detction S/N: %.3f Peak S/N: %.3f"\
                               %(num, z_best, Rs[r_max], SNRs[r_max], SNR_ps[r_max]))
        
            
        return ccs, rv, z_ccs, Rs, SNRs, SNR_ps, flag_edge

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
                ccs, rv0, z_ccs, Rs, SNRs, SNR_ps, flag_edge = self.cross_correlation(num, h_contrast=h_contrast, 
                                                                              temp_type=temp_type, temp_model=temp_model, 
                                                                              kind_intp=kind_intp, edge=edge, rv=None, plot=False)
                self.rv = rv0
            else:
                ccs, rv, z_ccs, Rs, SNRs, SNR_ps, flag_edge = self.cross_correlation(num, h_contrast=h_contrast, 
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
                self.pos_BCG = (self.Tab_SE[id_BCG]["X_IMAGE"], self.Tab_SE[id_BCG]["Y_IMAGE"])
            elif np.ndim(id_BCG)==1:
                pos_BCG1 = (self.Tab_SE[id_BCG[0]]["X_IMAGE"], self.Tab_SE[id_BCG[0]]["Y_IMAGE"])
                pos_BCG2 = (self.Tab_SE[id_BCG[1]]["X_IMAGE"], self.Tab_SE[id_BCG[1]]["Y_IMAGE"])
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
                          plot=True, print_out=True):
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
        
        obj_SE = Object_SE(self.Tab_SE[ind], k_wid=k_wid, cube=self.datacube,
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
            pa, clus_cen_angle, max_dev = compute_centroid_offset(obj_SE, spec=spec, wavl=self.wavl, 
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
                                                                         plot=plot, print_out=print_out)
            return (d_angle, offset, dist_clus, pa, clus_cen_angle, max_dev)
        
        except (ValueError, TypeError) as error:
            if print_out:
                print("Unable to compute centroid! Error raised.")
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
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
            pa, clus_cen_angle, max_dev = self.centroid_analysis(num, z=z, k_wid=k_wid, 
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
            self.max_devs = np.append(self.max_devs, max_dev)
        
    def construct_control(self, Num_v, mag_cut=None, mag0=25.2, dist_cut=25, bootstraped=False, n_boot=100):
        """Construct control sample for comparison
        
        Parameters
        ----------
        Num_v : number list to be excluded in the control sample in the SE catalog
        mag_cut: magnitude range (lower, upper)
        dist_cut : minium distance (in pixel) to the field edge
        
        """
        # remove nan values that fail in centroid analysis
        Num_not_v = np.setdiff1d(self.Tab_SE["NUMBER"], Num_v)
        ind_not_v = Num_not_v - 1
        Num_c_all = np.setdiff1d(self.Tab_SE["NUMBER"][ind_not_v], 
                                 1 + np.where(np.isnan(self.diff_centroids))[0])
        
        # magnitude cut
        mag = -2.5*np.log10(self.Tab_SE["FLUX_AUTO"])+mag0
        mag_no_nan = mag[~np.isnan(mag)]
        
        if mag_cut is None:
            mag_cut = (mag.min(),mag.max())
            
        Num_mag_ctrl = self.Tab_SE["NUMBER"][(mag>mag_cut[0])&(mag<mag_cut[1])]
        Num_c = np.intersect1d(Num_c_all, Num_mag_ctrl)
        
        # area cut
        thre_iso_area = 10.
        print("Isophotal area threshold: ", thre_iso_area)
        Num_area_ctrl = self.Tab_SE["NUMBER"][self.Tab_SE["ISOAREA_IMAGE"]>thre_iso_area]
        
        Num_c = np.intersect1d(Num_c, Num_area_ctrl)
        
#         # SNR cut
#         thre_snr = np.percentile(self.SNRp_best,90)
#         print("peak S/N threshold: ", thre_iso_area)
#         Num_snr_ctrl = self.Tab_SE["NUMBER"][(self.SNRp_best<thre_snr)]
#         Num_c = np.intersect1d(Num_c, Num_snr_ctrl)
        
        # radius cut
        Num_rp_ctrl = self.Tab_SE["NUMBER"][(self.Tab_SE["PETRO_RADIUS"]>0)&(self.Tab_SE["PETRO_RADIUS"]<10)]        
        Num_c = np.intersect1d(Num_c, Num_rp_ctrl)      
        
        # elongation cut < 2 sigma
        thre_elong = np.percentile(self.Tab_SE["ELONGATION"],97.5)
        Num_elong_ctrl = self.Tab_SE["NUMBER"][self.Tab_SE["ELONGATION"]<thre_elong]
        Num_c = np.intersect1d(Num_c, Num_elong_ctrl)
        
        # edge cut
        Y_pix, X_pix = np.indices(self.mask_edge.shape)
        dist_edge = np.array([np.sqrt((X_pix[~self.mask_edge]-gal["X_IMAGE"])**2 + \
                              (Y_pix[~self.mask_edge]-gal["Y_IMAGE"])**2).min() \
                              for gal in self.Tab_SE])
        self.dist_edge = dist_edge
        print("Distance to field edge threshold: ", dist_cut)
        Num_edge_ctrl = self.Tab_SE["NUMBER"][dist_edge>dist_cut]
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
        