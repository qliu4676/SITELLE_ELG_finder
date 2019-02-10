import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.stats import mad_std
import scipy.stats as stats

from utils import *

class Read_Raw_Datacube: 
    def __init__(self, file_path, name):
        self.hdu = fits.open(file_path)
        self.header = self.hdu[0].header
        self.name = name
        
        # Pixel scale, Step of wavenumber, Start of wavenumber, Number of steps
        self.pix_scale = self.hdu[0].header["PIXSCAL1"] #arcsec/pixel
        self.d_wavn = self.hdu[0].header["CDELT3"] #cm^-1 / pixel
        self.wavn_start = self.hdu[0].header["CRVAL3"] #cm^-1
        self.nstep = self.hdu[0].header["STEPNB"] 
        
        # Total wavenumber
        self.wavn = np.linspace(self.wavn_start, self.wavn_start+(self.nstep-1)*self.d_wavn, self.nstep)  #cm^-1
        
        # Effective data range from 12100 cm^-1 to 12550 cm^-1 
        wavn_eff_range = (self.wavn>12100) & (self.wavn<12550)
        self.wavl = 1./self.wavn[wavn_eff_range][::-1]*1e8
        
        # Raw effective datacube and stack field in wavelength
        self.raw_datacube = self.hdu[0].data[wavn_eff_range][::-1]
        self.raw_stack_field = self.raw_datacube.sum(axis=0)
        
        # Field edge to be masked
        self.mask_edge = self.raw_stack_field < 3*mad_std(self.raw_stack_field)
        
        self.hdu_header_new = self.hdu[0].header.copy()
        self.hdu_header_new["CRVAL3"] = self.wavn[wavn_eff_range][0]  # New minimum wavenumber in cm-1
        self.hdu_header_new["STEPNB"] = np.sum(wavn_eff_range)   # New step
        
    def display(self, img, vmax=None):
        # Display a image
        plt.figure(figsize=(12,12))
        if vmax==None: vmax=vmax_2sig(img)
        plt.imshow(img, vmin=0.,vmax=vmax, origin="lower",cmap="gray")
        plt.show()
        
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
        
    def remove_background(self, box_size=64, filter_size=3, n_iter=5, save_path = './', plot=False):
        # Remove background (low-frequency component) using SEXtractor estimator (filtering + mode)
        self.datacube_bkg_sub = np.empty_like(self.raw_datacube)
        for i, field in enumerate(self.raw_datacube):
            if np.mod(i+1, 10)==0: print("Removing background...Frame: %d"%(i+1))
            
            field_sub, back = background_sub_SE(field, mask=self.mask_edge, 
                                                b_size=box_size, f_size=filter_size, n_iter=n_iter)
            self.datacube_bkg_sub[i] = field_sub
            
            # Save background subtraction
            if plot:
                display_background_sub(field, back)
                plt.savefig(save_path + "bkg_sub_channel%d.png"%(i+1),dpi=150)
                plt.close(fig)  
                
        self.stack_field = self.datacube_bkg_sub.sum(axis=0)

    def remove_fringe(self, skip_frames=[None], box_size=8, filter_size=1, n_iter=5, save_path = './', plot=False):
        # Remove fringe and artifacts (high-frequency component)
        self.datacube_bkg_fg_sub = np.empty_like(self.datacube_bkg_sub)
        for i, field in enumerate(self.datacube_bkg_sub):
            if np.mod(i+1, 10)==0: print("Removing fringe...Frame: %d"%(i+1))
                
            if i in skip_frames:
                self.datacube_bkg_fg_sub[i] = field
            else:
                field_sub, back = background_sub_SE(field, mask=self.mask_edge, 
                                                    b_size=box_size, f_size=filter_size, n_iter=n_iter)
                self.datacube_bkg_fg_sub[i] = field_sub
                
                # Save fringe subtraction
                if plot:
                    display_background_sub(field, back)
                    plt.savefig(save_path + "fg_sub_channel%d.png"%(i+1),dpi=150)
                    plt.close(fig)  
                    
        self.stack_field = self.datacube_bkg_fg_sub.sum(axis=0)
                    
    def save_fits(self, save_path = './', suffix=""):
        """Write Stacked Image and Datacube after background & fringe subtraction"""
        print("Saving background & fringe subtracted datacube and stacked field...")
        
        hdu_cube = fits.PrimaryHDU(data = self.datacube_bkg_fg_sub, header=self.hdu_header_new)
        hdu_cube.writeto(save_path + '%s_cube%s.fits'%(self.name, suffix),overwrite=True)

        hdu_stack = fits.PrimaryHDU(data = self.stack_field, header=self.hdu_header_new)
        hdu_stack.writeto(save_path + '%s_stack%s.fits'%(self.name, suffix),overwrite=True)
        
class Read_Datacube:
    def __init__(self, file_path, name, SE_catalog, z0=None):
        self.hdu = fits.open(file_path)
        self.header = self.hdu[0].header
        self.datacube = self.hdu[0].data
        self.name = name
        self.z0 = z0
        
        # Pixel scale, Step of wavenumber, Start of wavenumber, Number of steps
        self.pix_scale = self.hdu[0].header["PIXSCAL1"] #arcsec/pixel
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
        
    def read_mask(self, file_path):
        # Open edge mask
        hdu_mask = fits.open(file_path)
        self.mask_edge = (hdu_mask[0].data == 0)
        
    def read_seg(self, file_path):
        # Open SE segmentation mask
        hdu_seg = fits.open(file_path)
        self.img_seg = hdu_seg[0].data
        
    def calculate_seeing(self, R_pix = 10, mag_cut=0.25, sigma_guess=1., plot=False):
        """Use the brigh star-like objects (class_star>0.7) to calculate 
        # median seeing fwhm by fitting gaussian profiles
        # R_pix : max range of profiles to be fitted (in pixel)
        # magcut : percentage of cut in isophotal magnitude
        # sigma_guess : initial guess of sigma of gaussian profile (in pixel) 
        # plot : plot radial profile for each object in use"""
        
        from scipy.optimize import curve_fit
        def gaussian_func(x, a, sigma):
            # Define a gaussian function with offset
            return a * np.exp(-x**2/(2*sigma**2))

        # Cut in CLASS_STAR, measurable, roundness and mag threshold
        mag_0 = np.percentile(-2.5*np.log10(self.Tab_SE["FLUX_ISO"]), mag_cut*100)
        star_cond = (self.Tab_SE["CLASS_STAR"]>0.7) & (self.Tab_SE["PETRO_RADIUS"]>0) \
                    &(self.Tab_SE["B_IMAGE"]/self.Tab_SE["A_IMAGE"]>0.7) \
                    &(-2.5*np.log10(self.Tab_SE["FLUXERR_ISO"])<mag_0) \
                    & (np.sqrt(self.Tab_SE["A_IMAGE"]**2+self.Tab_SE["B_IMAGE"]**2)<self.Tab_SE["KRON_RADIUS"])
        
        self.Tab_star = self.Tab_SE[star_cond]
        
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
        # num : SE object number
        # ks : a set of aperture ring (in SE petrosian radius) to determined the optimized aperture
        # k1, k2 : inner/outer radius (in SE petrosian radius) of annulus for evaluate background RMS
        # print_out : print out the optimized k and S/N
        # plot : plot the S/N vs k curve
        # display: display the image thumbnail with apertures"""
        
        tab = self.Tab_SE[self.Tab_SE["NUMBER"]==num]
        obj_SE = Object_SE(tab, cube=self.datacube, img_seg=self.img_seg, mask_field=self.mask_edge)
        k_opt, snr_opt = obj_SE.compute_aper_opt(ks=ks, k1=k1, k2=k2,
                                                 ext_type=ext_type, print_out=print_out, plot=plot)
        spec = obj_SE.extract_spec(k_opt, ext_type=ext_type, k1=k1, k2=k2, wavl=self.wavl, plot=plot)
        if display:
            obj_SE.img_display()
        return spec, k_opt, snr_opt, obj_SE.apers
    
    def spec_extraction_all(self, ks = np.arange(1.,4.5,0.2), k1=5., k2=8., display=True, save_path='./fig/img_thumb/'):
        """Extract spectra all SE detections with two optimized aperture"""
        self.obj_nums = np.array([])
        self.obj_aper = np.array([])
        self.obj_aper_opt = np.array([])
        for num in self.Tab_SE["NUMBER"]:
            
            spec, k, snr, apers = self.spec_extraction(num=num, ext_type='sky', 
                                                       ks=ks, k1=k1, k2=k2,
                                                       print_out=False, plot=False, display=False)
            self.obj_specs = np.vstack((self.obj_specs, spec)) if len(self.obj_nums)>0 else [spec]
            self.obj_aper = np.append(self.obj_aper, k)
            
            spec_opt, k_opt, snr_opt, apers_opt = self.spec_extraction(num=num, ext_type='opt', 
                                                                       ks=ks, k1=k1, k2=k2,
                                                                       print_out=False, plot=False, display=display)
            self.obj_specs_opt = np.vstack((self.obj_specs_opt, spec_opt)) if len(self.obj_nums)>0 else [spec_opt]
            self.obj_aper_opt = np.append(self.obj_aper_opt, k_opt)
            
            
            # Display the image marking the two apertures and annulus
            if display:
                apers[0].plot(color='limegreen', lw=1.5, ls='--')
                plt.savefig(save_path+"SE#%d.png"%(num),dpi=150)
                plt.close()
            
            self.obj_nums = np.append(self.obj_nums, num)
            print ("#%d spectra extracted"%(num))
            
    def save_spec_plot(self, save_path='./fig/spec/'):
        """Save normalized extracted spectra by the two apertures"""
        for num, spec, spec_opt in zip(self.obj_nums, self.obj_specs, self.obj_specs_opt):
            plt.plot(self.wavl, spec/spec.max(), color='gray', label="Aperture cen", alpha=0.7)
            plt.plot(self.wavl, spec_opt/spec_opt.max(), color='k', label="Aperture opt", alpha=0.9)
            plt.xlabel("wavelength",fontsize=12)
            plt.ylabel("normed flux",fontsize=12)
            plt.legend(fontsize=12)
            plt.savefig(save_path+"SE#%d.png"%num,dpi=150)
            plt.close()
            
    def save_spec_fits(self, save_path='./', suffix=""):  
        """Save extracted spectra and aperture (in the unit of SE pertrosian radius) as fits"""
        hdu_spec = fits.PrimaryHDU(data=self.obj_specs)
        hdu_aper = fits.BinTableHDU.from_columns([fits.Column(name="k_aper_cen",array=self.obj_aper,format="E")])
        hdu_spec_opt = fits.ImageHDU(data=self.obj_specs_opt)
        hdu_aper_opt = fits.BinTableHDU.from_columns([fits.Column(name="k_aper_opt",array=self.obj_aper_opt,format="E")])
        hdu_num = fits.BinTableHDU.from_columns([fits.Column(name="obj_nums",array=self.obj_nums,format="J")])

        hdul = fits.HDUList(hdus=[hdu_spec, hdu_aper, hdu_spec_opt, hdu_aper_opt, hdu_num])
        hdul.writeto(save_path+'%s-spec%s.fits'%(self.name, suffix), overwrite=True)
        
        
    def read_spec(self, file_path):
        """Read generated extracted spectra and aperture fits"""
        hdu_spec = fits.open(file_path)
        self.obj_specs = hdu_spec[0].data
        self.k_apers = hdu_spec[1].data['k_aper_cen']
        self.obj_specs_opt = hdu_spec[2].data
        self.k_apers_opt = hdu_spec[3].data['k_aper_opt']
        self.obj_nums = hdu_spec[4].data['obj_nums']
        
        # Stacked spectra to estimate edge ratio
        self.spec_stack = (self.obj_specs_opt/self.obj_specs_opt.max(axis=1).reshape(-1,1)).sum(axis=0)
        self.edge_ratio_stack = 0.7*np.sum((self.spec_stack - np.median(self.spec_stack) \
                                            <= -2.5*mad_std(self.spec_stack))) / len(self.wavl)
        
    def generate_template(self, temp_type="Ha-NII", zrange=(0.21,0.26), line_ratio=[1,9,3], width=4, sig_gauss=2.5, model="gaussian", number=200):
        """Generate template for with line ratio [Ha, NII6548, NII6584] and width [width]
        # - For Gaussian models, fwhm(=2.355*sigma) between spectral resolution to galaxy broadening v=250km/s.
        # sigma_v = (v)/sqrt(3), sigma_gal = sigma_v/3e5 * 6563*(1+z_cluster) 
        # - For delta function width=line_width"""
        
        self.template_model = model
        box_wid = np.max([width, self.d_wavl])
        self.temp, self.wavl_temp = generate_template(self.wavl, line_ratio=line_ratio, temp_type=temp_type,
                                                      sig_gauss=sig_gauss, line_width=box_wid, model=model)
        if number>1:
            if temp_type=="Ha-NII":        
                Ha_NII6584_ratios = stats.lognorm.rvs(0.64, loc=1.81, scale=1.24, size=number)
                Line_ratios = np.vstack([[1, 3*ratio, 3] for ratio in Ha_NII6584_ratios])
            elif temp_type=="Hb-OIII":        
                Hb_OIII5007_ratios = stats.lognorm.rvs(0.64, loc=0.29, scale=0.46, size=number)
                Line_ratios = np.vstack([[3*ratio, 1, 3] for ratio in Hb_OIII5007_ratios])
            sig_gal = 250./np.sqrt(3)/3e5 * 6563*(1+self.z0)
            sig_min = self.d_wavl/2.355
            sig_max = np.sqrt(sig_min**2 + sig_gal**2)
            Stddevs = stats.uniform.rvs(size=number) * (sig_max-sig_min) + sig_min
            z_min, z_max = zrange
            self.temp_redshifts = stats.uniform.rvs(size=number) * (z_max-z_min) + z_min
            
            temp_lib, wavl_temps = np.empty((number, len(self.wavl))), np.empty((number, len(self.wavl)))
            for i,(z, lr, stddev) in enumerate(zip(self.temp_redshifts, Line_ratios, Stddevs)):
                temp_lib[i], wavl_temps[i] = generate_template(self.wavl, z=z, temp_type=temp_type,
                                                    line_ratio=lr, sig_gauss=stddev, 
                                                    line_width=box_wid, model=model, plot=False)
                
            self.temps, self.wavl_temps = temp_lib, wavl_temps
            
    def cross_correlation(self, num, w_l=35, ext_type='opt', model='GP',
                          kernel_scale=100, kenel_noise=1e-3, edge_ratio=None, rv=None, plot=True):
        """Cross-correlation for one SE detection
        # rv : relative redshift and velocity to z=z0, keep None if not known (the first object in batch)
        # w_l : half-width (in AA, rest-frame) of vicinity around Ha to be excluded when measuring noise RMS""" 
        if ext_type=='sky':
            spec = self.obj_specs[self.obj_nums==num][0]
        elif ext_type=='opt':
            spec = self.obj_specs_opt[self.obj_nums==num][0]
        else: print('Extraction type: "sky" or "opt"?')
            
        if plot:    
            fig, (ax0,ax1,ax2) = plt.subplots(nrows=3, ncols=1,figsize=(9,8))
        else:
            fig, (ax0,ax1,ax2) = None, (None, None, None)
            
        if edge_ratio is None:
            edge_ratio = self.edge_ratio_stack
            
        y, x, cont_range = fit_continuum(spec, self.wavl, model=model, 
                                         kernel_scale=kernel_scale,  kenel_noise=kenel_noise, edge_ratio=edge_ratio,
                                         print_out=False, plot=plot, fig=fig, ax=ax0)
        ccs, rv, z_ccs, SNR_ps = xcor_SNR(y, x, temps=self.temps, wavl_temp=self.wavl_temps, 
                                          cont_range=cont_range, w_l=w_l, d_wavl=self.d_wavl, 
                                          z0=self.z0, rv=rv, plot=plot, fig=fig, axes=(ax1,ax2))
        if plot: plt.tight_layout()
        
        z_best = z_ccs[np.argmax(SNR_ps)]
        line_at_edge = ((1+z_best)*6563.<(self.wavl[0]+25)) | ((1+z_best)*6563.>(self.wavl[-1]-25))
                                
        print ("SE Objects #%d   S/N med: %.3g  max: %.3g"%(num, np.median(SNR_ps), np.max(SNR_ps)))
        
        if line_at_edge: 
            flag = 2  # near the wavelength edge, raise caution
        else: 
            flag = 0
            
        return ccs, rv, z_ccs, SNR_ps, flag

    def cross_correlation_all(self, ext_type='opt', kernel_scale=100, kenel_noise=1e-3,  edge_ratio=None):
        self.cc_nums = np.array([])
        self.CC_zccs = np.zeros((len(self.obj_nums), len(self.temps)))
        self.CC_SNRs = np.zeros((len(self.obj_nums), len(self.temps)))
        self.cc_flags = np.array([])
        
        if edge_ratio is None:
            edge_ratio = self.edge_ratio_stack
            
        for j, num in enumerate(self.obj_nums):
            if j==0:
                ccs, rv0, z_ccs, SNR_ps, flag = self.cross_correlation(num, rv=None, ext_type=ext_type, 
                                                                       kernel_scale=kernel_scale,  kenel_noise=kenel_noise,
                                                                       edge_ratio=edge_ratio, plot=False)
                self.rv = rv0
            else:
                ccs, rv, z_ccs, SNR_ps, flag = self.cross_correlation(num, rv=self.rv, ext_type=ext_type, 
                                                                      kernel_scale=kernel_scale,  kenel_noise=kenel_noise,
                                                                      edge_ratio=edge_ratio, plot=False)
               
            self.CC_zccs[j] = z_ccs
            self.CC_SNRs[j] = SNR_ps
            self.CCs = np.concatenate([self.CCs, [ccs]]) if len(self.cc_nums)>0 else [ccs]
            self.cc_flags = np.append(self.cc_flags, flag)
            
            self.cc_nums = np.append(self.cc_nums, num)
               
               
               
#             spec = self.obj_specs[self.obj_nums==num][0]
#             y, x, cont = fit_continuum(spec, self.wavl, print_out=False, plot=False)
#             cc, rv, z_cc, SNR_p, var_fitp, cc_pmax = xcor_SNR(y, self.temp, x, self.wavl_temp, 
#                                                               wavl=self.wavl, d_wavl=self.d_wavl, 
#                                                               z0=z0, plot=False)
#             spec_opt = self.obj_specs_opt[self.obj_nums==num][0]
#             y, x, cont = fit_continuum(spec_opt, self.wavl, print_out=False, plot=False)
#             cc, rv, z_cc, SNR_p, var_fitp, cc_pmax = xcor_SNR(y, self.temp, x, self.wavl_temp, 
#                                                               wavl=self.wavl, d_wavl=self.d_wavl, 
#                                                               z0=z0, plot=False)
#             print ("S/N: %.3g   Fit peak variance: %.3g"%(SNR_p, var_fitp))
            
#             self.cc_zccs_opt = np.append(self.cc_zccs_opt, z_cc) if len(self.cc_nums)>0 else np.array([z_cc])
#             self.cc_SNRs_opt = np.append(self.cc_SNRs_opt, SNR_p) if len(self.cc_nums)>0 else np.array([SNR_p])
#             self.cc_vars_opt = np.append(self.cc_vars_opt, var_fitp) if len(self.cc_nums)>0 else np.array([var_fitp])
#             self.CCs_opt = np.vstack((self.CCs_opt, cc)) if len(self.cc_nums)>0 else [cc]

#             self.cc_nums = np.append(self.cc_nums, num)

    def save_cc_fits(self, save_path='./', suffix=""): 
        """Save cross-correlation results (SNR, z, CC funtion, flag, index) as fits"""
        hdu_CC_SNRs = fits.PrimaryHDU(data=self.CC_SNRs)
        hdu_CC_zccs = fits.ImageHDU(data=self.CC_zccs)
        hdu_CCs = fits.ImageHDU(data=self.CCs)
        hdu_cc_flag = fits.BinTableHDU.from_columns([fits.Column(name="cc_flags",array=self.cc_flags,format="J")])
        hdu_cc_nums = fits.BinTableHDU.from_columns([fits.Column(name="cc_nums",array=self.cc_nums,format="J")])

        hdul = fits.HDUList(hdus=[hdu_CC_SNRs, hdu_CC_zccs, hdu_CCs, hdu_cc_flag, hdu_cc_nums])
        hdul.writeto(save_path+'%s-cc%s.fits'%(self.name, suffix), overwrite=True) 
        
    def read_cc(self, filepath):
        hdu_cc = fits.open(filepath)
        
        self.CC_SNRs = hdu_cc[0].data
        self.CC_SNR_max = np.max(self.CC_SNRs, axis=1)
        sn_max = np.argmax(self.CC_SNRs, axis=1)
        
        self.CC_zccs = hdu_cc[1].data
        self.z_best = np.array([self.CC_zccs[i, sn_max[i]] for i in range(len(self.CC_SNRs))])
        
        self.CCs = hdu_cc[2].data
        self.CC_best = np.array([self.CCs[i, sn_max[i]] for i in range(len(self.CC_SNRs))])
        
        self.cc_flags = hdu_cc[3].data['cc_flags']
        self.cc_nums = hdu_cc[4].data['cc_nums']
    
    def read_cluster_boundary(self, filepath):
        """Read Cluster boundary map and BCG position, only for double cluster"""
        self.boundary_map = fits.open(filepath)[0].data 
        
    def assign_BCG_position(self, id_BCG):
        """Assign BCG position (x, y) in pixel from id_BCG, ((x1,y1),(x2,y2)) for double cluster"""
        if np.ndim(id_BCG)==0:
            self.pos_BCG = (self.Tab_SE[id_BCG]["X_IMAGE"], self.Tab_SE[id_BCG]["Y_IMAGE"])
        elif np.ndim(id_BCG)==1:
            pos_BCG1 = (self.Tab_SE[id_BCG[0]]["X_IMAGE"], self.Tab_SE[id_BCG[0]]["Y_IMAGE"])
            pos_BCG2 = (self.Tab_SE[id_BCG[1]]["X_IMAGE"], self.Tab_SE[id_BCG[1]]["Y_IMAGE"])
            self.pos_BCG = np.vstack([pos_BCG1,pos_BCG2])
        
    def centroid_analysis(self, num, z=None, k_wid=5, centroid_type="ISO2", plot=True):
        ind = (self.obj_nums==num)
        obj_SE = Object_SE(self.Tab_SE[ind], k_wid=k_wid, cube=self.datacube,
                           img_seg=self.img_seg, mask_field=self.mask_edge)
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
            d_angle, offset, dist_clus = compute_centroid_offset(obj_SE, spec=spec, wavl=self.wavl, 
                                                                 z_cc=z, centroid_type=centroid_type, k_aper=k_aper, 
                                                                 pos_BCG = self.pos_BCG, affil_map=boundary_map, plot=plot)
            return (d_angle, offset, dist_clus)
        
        except (ValueError, TypeError) as error:
            print("Unable to compute centroid! Error raised.")
            return (np.nan, np.nan, np.nan)
    
    def centroid_analysis_all(self, k_wid=5, snr_control=3, centroid_type="ISO2", plot=False):
        """Compute centroid offset and angle for all SE detections"""
        self.centroid_type = centroid_type
        self.diff_angles = np.array([]) 
        self.diff_centroids = np.array([])
        self.dist_clus_cens = np.array([])
        for num in self.obj_nums:
            print("#EL-%1d"%num)
            ind = (self.obj_nums==num)
            snr_max = self.CC_SNR_max[ind]

            if snr_max < snr_control:  # For non-emission galaxies, pick a random window excluding the edge
                z = np.random.uniform(low=(self.wavl[0]+75)/6563.-1, 
                                      high=(self.wavl[-1]-75)/6563.-1)
            else: # For emission galaxies, pick a window 70A aorund emission
                z = self.z_best[ind]          

            diff_angle, centroid_offset, dist_clus_cen = self.centroid_analysis(num, z=z, k_wid=k_wid, 
                                                                                centroid_type=centroid_type, plot=False)
            self.diff_angles = np.append(self.diff_angles, diff_angle)
            self.diff_centroids = np.append(self.diff_centroids, centroid_offset)
            self.dist_clus_cens = np.append(self.dist_clus_cens, dist_clus_cen)
        
    def construct_control(self, bootstraped=False, mag_per=(2.5, 97.5), snr_control=3, n_boot=100):
        """Construct control sample for comparison"""
        # remove nan values that fail in centroid analysis
        Num_c = np.setdiff1d(self.Tab_SE["NUMBER"][self.CC_SNR_max<snr_control], 
                             1+np.where(np.isnan(self.diff_centroids))[0])
        
        # magnitude cut
        mag = -2.5*np.log10(self.Tab_SE["FLUX_AUTO"])
        mag_no_nan = mag[~np.isnan(mag)]
        mag_cut = (np.percentile(mag_no_nan, mag_per[0]), np.percentile(mag_no_nan, mag_per[1]))
        num_mag = self.Tab_SE["NUMBER"][(mag>mag_cut[0])&(mag<mag_cut[1])]
        Num_c = np.intersect1d(Num_c, num_mag)
        
        print('Control Sample : n=%d'%len(Num_c))
        ind_c = Num_c - 1
        self.diff_angles_c = self.diff_angles[ind_c]
        self.diff_centroids_c = self.diff_centroids[ind_c]
        self.dist_clus_cens_c = self.dist_clus_cens[ind_c]
        
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