import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate, signal, integrate, stats

from skimage import exposure
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic

from astropy.stats import SigmaClip, mad_std, gaussian_fwhm_to_sigma
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel, MexicanHat2DKernel
from astropy.modeling import models, fitting
from astropy.visualization import make_lupton_rgb

from photutils import Background2D, SExtractorBackground
from photutils import EllipticalAnnulus, EllipticalAperture, aperture_photometry

from photutils import centroid_com, centroid_2dg
from photutils import detect_sources, detect_threshold, deblend_sources, source_properties


def vmax_2sig(img):
    # upper limit of visual imshow defined by 2 sigma above median
    return np.median(img)+2*np.std(img)

def vmax_3sig(img):
    # upper limit of visual imshow defined by 3 sigma above median
    return np.median(img)+3*np.std(img)

def vmax_hist(img):
    # upper limit of visual imshow defined by 99.9% histogram
    img_cdf, bins = exposure.cumulative_distribution(img)
    return bins[np.argmin(abs(img_cdf-0.999))]

def printsomething():
    print("Print")
    
def coord_ImtoArray(X_IMAGE, Y_IMAGE):
    # Convert image coordniate to array coordinate
    x_arr, y_arr = Y_IMAGE-1, X_IMAGE-1
    return x_arr, y_arr

def background_sub_SE(field, mask=None, b_size=64, f_size=3, n_iter=5):
    # Subtract background using SE estimator with mask
    Bkg = Background2D(field, mask=mask, bkg_estimator=SExtractorBackground(),
                       box_size=(b_size, b_size), filter_size=(f_size, f_size),
                       sigma_clip=SigmaClip(sigma=3., iters=n_iter))
    back = Bkg.background * ~mask
    field_sub = field - back
    return field_sub, back

def display_background_sub(field, back):
    # Display and save background subtraction result with comparison 
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(16,5))
    ax1.imshow(field, origin="lower",cmap="gray", vmin=0., vmax=vmax_2sig(field))
    ax2.imshow(back, origin='lower', cmap='gray', vmin=0., vmax=vmax_2sig(field))
    ax3.imshow(field - back, origin='lower', cmap='gray', vmin=0., vmax=vmax_2sig(field))
    plt.tight_layout()

    
# SE Object Pipeline start from here
class Object_SE:
    def __init__(self, obj, k_wid=8, cube=None, img_seg=None, mask_field=None):
        self.num = obj["NUMBER"]
        self.x_c, self.y_c = coord_ImtoArray(obj["X_IMAGE"], obj["Y_IMAGE"])
        self.width = np.max([obj["A_IMAGE"],2])
        self.a_image = obj["A_IMAGE"]
        self.b_image = obj["B_IMAGE"]
        self.theta = obj["THETA_IMAGE"]*np.pi/180.
        
        self.x_min, self.x_max = (np.max([int(self.x_c - k_wid*self.width), 0]), 
                                 np.min([int(self.x_c + k_wid*self.width), cube.shape[1]]))
        self.y_min, self.y_max = (np.max([int(self.y_c - k_wid*self.width), 0]), 
                                 np.min([int(self.y_c + k_wid*self.width), cube.shape[2]]))
        self.center_pos=(self.y_c-self.y_min, self.x_c-self.x_min)
            
        self.mask_thumb = mask_field[self.x_min:(self.x_max+1), self.y_min:(self.y_max+1)]
        
        self.seg_thumb = img_seg[self.x_min:(self.x_max+1), self.y_min:(self.y_max+1)]
        self.mask_seg = (self.seg_thumb==obj["NUMBER"]) | (self.seg_thumb==0)
        
        self.cube_thumb = cube[:,self.x_min:(self.x_max+1), self.y_min:(self.y_max+1)]
        self.img_thumb = self.cube_thumb.sum(axis=0)
        
    def img_display(self, vmin=0., vmax=None):
        fig,ax=plt.subplots(1,1)
        if vmax==None: 
            vmax=vmax_2sig(self.img_thumb)
        s = ax.imshow(self.img_thumb, origin="lower",cmap="hot", vmin=vmin, vmax=vmax)
        ax.plot(self.center_pos[0], self.center_pos[1], "lime", marker="+", ms=15, mew=2, alpha=0.7)
        ax.set_xticklabels(ax.get_xticks().astype('int64')+self.y_min)
        ax.set_yticklabels(ax.get_yticks().astype('int64')+self.x_min)
        plt.colorbar(s)
        plt.tight_layout()
        try:
            self.apers[0].plot(color='lime', linewidth=2, alpha=0.7)
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
            plt.figure()
            plt.plot(ks, snr_ks)
            plt.axhline(snr_opt, color='k', ls='--', alpha=0.8)
            plt.xlabel("Radius (R_petro)", fontsize=14)
            plt.xlabel("S/N", fontsize=14)
            plt.title("Extraction: %s"%ext_type, fontsize=14)
            plt.show()
            
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
        if plot:
            fig,ax=plt.subplots(1,1)
            ax.plot(wavl,spec/spec.max(),lw=1)
        return spec

def generate_template(wavl, z=0, line_ratio=[1,9,3], temp_type="Ha-NII", line_width=4, sig_gauss=3, model="box", plot=True):
    """Generate a template for with line ratio [NII6548, Ha, NII6584]/[Hb, OIII4959, OIII5007] and width [line_width]
    # For Gaussian models sigma(stddev)=sig_gauss
    # For delta function width=line_width"""
    wav_wid = wavl[-1]-wavl[0]
    if temp_type=="Ha-NII":
        wavl_temp = np.e**np.linspace(np.log(6563*(1+z)-wav_wid/2.),np.log(6563*(1+z)+wav_wid/2.),int(len(wavl)))
        line_pos = [6548.*(1+z), 6563.*(1+z), 6584.*(1+z)]
    elif temp_type=="Hb-OIII":
        wavl_temp = np.e**np.linspace(np.log(4934*(1+z)-wav_wid/2.),np.log(4934*(1+z)+wav_wid/2.),int(len(wavl)))
        line_pos = [4861.*(1+z), 4959.*(1+z), 5007.*(1+z)]
    if model == "box":
        Mod = models.Box1D
        s = np.sum([models.Box1D(amplitude=lr, x_0=lp, width=line_width) for (lr, lp) in zip(line_ratio, line_pos)])
    elif model == "gaussian":
        s = np.sum([models.Gaussian1D(amplitude=lr, mean=lp, stddev=sig_gauss) for (lr, lp) in zip(line_ratio, line_pos)])
    else: print('Model type: "box" or "gaussian"?')
    temp = s(wavl_temp)
    temp /= temp.max()
    
    if plot:
        plt.figure()
        plt.step(wavl_temp, temp,"g",where="mid")
        for lp in  line_pos:
            plt.axvline(lp, color="k", ls="-.", alpha=0.7) 
        plt.show()
        
    return temp, wavl_temp    
    
    

def fit_continuum(spec, wavl, model='GP', kernel_scale=100, kenel_noise=1e-3, edge_ratio=0.15,
                  print_out=True, plot=True, fig=None, ax=None):
    """Fit normalized continuum with Trapezoid1D model or Gaussian Process"""
    
    # rebin wavelength linearly in log scale
    logwavl = np.log(wavl)
    logwavl_intp = np.linspace(logwavl[0],logwavl[-1],len(wavl)+1)
    f_interp = interpolate.interp1d(logwavl, spec, kind="linear", fill_value="extrapolate")
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
        if np.sum((y_intp - np.median(y_intp) <= -2.5*mad_std(y_intp)))> edge_ratio*len(wavl):  # For some special source with sharp edge
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
        ax.plot(np.e**logwavl_intp, cont, "s", color='indianred', ms=4, label="cont.",alpha=0.7)
        ax.plot(wavl_rebin, cont_fit, "orange", label="cont. fit", alpha=0.9)        
        ax.step(wavl_rebin, res, "royalblue",label="residual", where="mid",zorder=5, alpha=0.9)
        ax.set_xlabel('wavelength ($\AA$)')
        ax.set_ylabel('norm Flux')
        ax.legend(ncol=2)
          
    res -= np.median(res)
    res /= res.max()
    
    return res, wavl_rebin, cont_range


def xcor_SNR(y, x, temps, wavl_temps, 
             cont_range, d_wavl, z0=0.228, 
             zmin=0.21, zmax=0.26, w_l=35., 
             rv=None, plot=False, fig=None, axes=None):
    """Cross-correlate with template and return rative velocity to z0 and corr function
    
    # y: normalized residual spectra
    # x : rebinned wavelength
    # zmin/zmax : z range from range of wavlength
    # rv : relative redshift and velocity to z=z0, None if not known (the first object in batch)
    # w_l : half-width (in AA, rest-frame) of vicinity around Ha to be excluded when measuring noise RMS""" 
    
    temps = np.atleast_2d(temps)
     
    # Compute relative redshift and velocity to z=z0, if rv not given 
    if rv is None:
        cc = signal.correlate(y, temps[0], mode="full", method="direct")
        rz = np.linspace((x[0] - d_wavl * temps.shape[1]/2.) / 6563. - 1, 
                         (x[-1] + d_wavl * temps.shape[1]/2.) / 6563 - 1, 
                         len(cc)) - z0
        rv = rz * 3e5
    
    # For all template
    ccs = np.zeros((temps.shape[0],len(rv)))
    z_ccs = np.zeros(temps.shape[0])
    SNR_ps = np.zeros(temps.shape[0])
    
    
    rv_egde = ((np.max([x[cont_range][0], x[0]+30])/6563.-1-z0) * 3e5, 
               (np.min([x[cont_range][-1], x[-1]-30])/6563.-1-z0) * 3e5) 
    
    for i, temp in enumerate(temps):
        cc = signal.correlate(y, temp, mode="full", method="direct")
        ccs[i] = cc/cc.max()
        
        # Peak redshift (via quadratic interpolation around the peak)
        # note: difference with a gaussian fitting to the peak is <0.0001 in z
        rv_p = rv[np.argmax(cc)]
        Interp_Peak = interpolate.interp1d(rv, cc, kind='quadratic')
        rv_intp = np.linspace(rv_p - w_l/6563.*3e5, rv_p + w_l/6563.*3e5, 100)
        cc_intp = Interp_Peak(rv_intp)
        rv_match = rv_intp[np.argmax(cc_intp)]
        z_cc = z0 + (rv_match/3e5)
        z_ccs[i] = z_cc
        
        S_p = cc_intp.max()
        z_range = (rv>(zmin-z0)*3e5) & (rv<(zmax-z0)*3e5)
        noise_range = ((rv < (rv_match - w_l/6563.*3e5)) | (rv > (rv_match + w_l/6563.*3e5))) &\
                        (rv > rv_egde[0]) & (rv < rv_egde[1])  # not use edge
        N_p = np.std(cc[z_range & noise_range])
#         N_p = mad_std(cc[z_range & noise_range])
        SNR_p = S_p/N_p
        SNR_ps[i] = SNR_p
        sn_max = np.argmax(SNR_ps)
    
    
    if plot:
        if fig==None:
            fig = plt.figure(figsize=(8,6))
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)
        else:
            ax1, ax2 = axes
        ax1.plot(x, y, c="royalblue", label="residual", alpha=0.9)
#         ax1.step(wavl_temp*(1+z_ccs[sn_max]),temps[sn_max]+np.min(y), color="seagreen", where="mid",
#                  alpha=0.9, label="Template: z=%.3g"%z_ccs[sn_max], zorder=3)
        ax1.step(wavl_temps[sn_max],temps[sn_max]+np.min(y), color="seagreen", where="mid",
                 alpha=0.9, label="Template: z=%.3g"%z_ccs[sn_max], zorder=3)
        ax1.axvline(6563*(1+z0), color="k", ls="--", alpha=0.8,lw=1)
        ax1.axvline(6563*(1+z_ccs[sn_max]), color="g", ls="-.", alpha=0.7)
        ax1.set_xlim(x[0]-10, x[-1]+10)
        ax1.set_xlabel('wavelength ($\AA$)')
        ax1.set_ylabel('norm Flux')
        ax1.legend()
        
        for i, temp in enumerate(temps):
            
            ax2.plot(rv, ccs[i], "firebrick", alpha=np.min([0.9, 0.9/len(temps)*3]))

            if fit_peak:
                ax2.plot(rv_ps, g_fit(rv_ps), "k-.", alpha=0.8, label="peak fit")
                
        ax2.axvline((z_ccs[sn_max]-z0)*3e5, color="darkred", ls="--", alpha=0.9)
        ax2.axvline(0, color="k", ls="--", lw=1, alpha=0.8)
        
        rv_left, rv_right = ((z_ccs[sn_max]-z0 - w_l/6563.)*3e5, (z_ccs[sn_max]-z0 + w_l/6563.)*3e5)
        ax2.axvspan(rv_left, rv_right, alpha=0.05, color='firebrick')
        if rv_egde[0]<rv_left:  
            ax2.axvspan(rv_egde[0], rv_left, alpha=0.05, color='gray')
        if rv_egde[1]>rv_right:  
            ax2.axvspan(rv_right, rv_egde[1], alpha=0.05, color='gray')
        
        ax2.text(0.86, 0.88, "S/N:%.1f"%SNR_ps.max(), color='darkred',fontsize=13, transform=ax2.transAxes)
        ax2.set_xlabel('Relative Velocity to z=%.3g (km/s)'%z0)
        ax2.set_xlim((zmin-z0)*3e5, (zmax-z0)*3e5)
        axb = ax2.twiny()
        axb.plot(rv/3e5 + z0, np.ones_like(rv)) # Dummy plot
        axb.cla()
        axb.set_xlim(zmin,zmax)
        axb.set_xlabel("redshift")      
        
    return ccs, rv, z_ccs, SNR_ps
    
    
def compute_centroid_offset(obj_SE, k_aper, spec, wavl, z_cc, pos_BCG, 
                            niter_aper=10, ctol_aper=0.01, iso_snr=1.2, 
                            centroid_type="APER", affil_map=None, plot=True):    
    # Compute the centroid of emission and continua, and plot
    em = (wavl>6548*(1+z_cc)-25)&(wavl<6583*(1+z_cc)+25)
    img_em = obj_SE.cube_thumb[em,:,:].sum(axis=0)/len(em[em])
    img_con = obj_SE.cube_thumb[~em,:,:].sum(axis=0)/len(em[~em])
    
    if centroid_type == "APER":
        print("optimal aperture:%.1f"%k_aper)
        x, y = obj_SE.center_pos
        for i in range(niter_aper):
            aper = EllipticalAperture(positions=(x,y), 
                                      a=k_aper*obj_SE.a_image, b=k_aper*obj_SE.b_image, theta=obj_SE.theta)
            m0 = aper.to_mask(method='exact')[0]
            mask_aper = m0.to_image(obj_SE.img_thumb.shape)
            data_img = obj_SE.img_thumb * mask_aper * obj_SE.mask_seg
            x_new, y_new = centroid_com(data_img)
            if np.sqrt((x-x_new)**2+(y-y_new)**2) < ctol_aper:
                continue
            else:
                x, y = x_new, y_new
        
        data_em = img_em * mask_aper * obj_SE.mask_seg
        data_con = img_con * mask_aper * obj_SE.mask_seg
        x1, y1 = centroid_com(data_em)
        x2, y2 = centroid_com(data_con)
        
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
        
    centroid_offset = np.sqrt((x1-x2)**2+(y1-y2)**2)
    
    if np.ndim(pos_BCG)==1:
        (x_BCG, y_BCG) = pos_BCG
    else:
        lab = affil_map[obj_SE.x_c[0].astype("int64"), obj_SE.y_c[0].astype("int64")]
        (x_BCG, y_BCG) = pos_BCG[0] if lab==1 else pos_BCG[1]     
        
    dist_clus_cen = np.sqrt((x_BCG-obj_SE.y_c[0]-1)**2 + (y_BCG-obj_SE.x_c[0]-1)**2)  
    
    pa = np.arctan2(x2-x1,y2-y1)
    clus_cen_angle = np.arctan2((x_BCG-obj_SE.y_c[0]-1),(y_BCG-obj_SE.x_c[0]-1))
    
    print("PA: %.3f,  Cluster-centric angle: %.3f"%(pa*180/np.pi,clus_cen_angle*180/np.pi))
    diff_angle = np.abs(pa*180/np.pi - clus_cen_angle*180/np.pi)
    if diff_angle>180:
        diff_angle = 360 - diff_angle
    
    if plot:
        plt.figure(figsize=(12,12))
        ax0 = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=1)
        ax0.plot(wavl, spec, "k",alpha=0.8)
        ax0.plot(wavl[em],spec[em],"seagreen",alpha=1)
        ax0.axvline(np.max([wavl[0], 6548*(1+z_cc)-15]), color="seagreen", lw=1, ls="-.", alpha=0.5)
        ax0.axvline(np.min([wavl[-1], 6583*(1+z_cc)+15]), color="seagreen", lw=1, ls="-.", alpha=0.5)
        ax0.set_xlabel('wavelength ($\AA$)')
        ax0.set_ylabel('Flux')

        ax1 = plt.subplot2grid((4, 3), (1, 1), colspan=1, rowspan=1)
        s = ax1.imshow(img_em, origin="lower",cmap="viridis",vmin=0.,vmax=vmax_3sig(img_em))
        ax1.plot(x1, y1, "seagreen", marker="+", ms=10, mew=2)

        ax1.set_xticklabels(ax1.get_xticks().astype("int")+obj_SE.y_min)
        ax1.set_yticklabels(ax1.get_yticks().astype("int")+obj_SE.x_min)
        ax1.set_title("emisson")
        plt.colorbar(s)
        if centroid_type != "APER":
            ax1s = plt.subplot2grid((4, 3), (1, 0), colspan=1, rowspan=1)
            ax1s.imshow(data_em, origin="lower",cmap="viridis",vmin=0.,vmax=vmax_3sig(img_em))
            ax1s.set_xticklabels(ax1s.get_xticks().astype("int")+obj_SE.y_min)
            ax1s.set_yticklabels(ax1s.get_yticks().astype("int")+obj_SE.x_min)
            ax1s.set_title("em. seg.")
            
        ax2 = plt.subplot2grid((4, 3), (2, 2), colspan=1, rowspan=1)
        s = ax2.imshow(img_con, origin="lower",cmap="hot",vmin=0.,vmax=vmax_3sig(img_con))
        ax2.plot(x2, y2, "k", marker="+", ms=10, mew=2)

        ax2.set_xticklabels(ax2.get_xticks().astype("int")+obj_SE.y_min)
        ax2.set_yticklabels(ax2.get_yticks().astype("int")+obj_SE.x_min)
        ax2.set_title("continuum")
        plt.colorbar(s)
        if centroid_type != "APER":
            ax2s = plt.subplot2grid((4, 3), (3, 2), colspan=1, rowspan=1)
            ax2s.imshow(data_con, origin="lower",cmap="hot",vmin=0.,vmax=vmax_3sig(img_con))
            ax2s.set_xticklabels(ax2s.get_xticks().astype("int")+obj_SE.y_min)
            ax2s.set_yticklabels(ax2s.get_yticks().astype("int")+obj_SE.x_min)
            ax2s.set_title("cont. seg.")
        
        if centroid_type == "APER":
            
            ax3 = plt.subplot2grid((4, 3), (1, 2), colspan=1, rowspan=1)
            ax3.imshow(mask_aper * obj_SE.mask_seg, origin="lower", cmap="gray")
            ax3.set_title("segmentation")
        
        ax = plt.subplot2grid((4, 3), (2, 0), colspan=2, rowspan=2)
        img_rgb = make_lupton_rgb(img_con, (img_con+img_em)/2., img_em, stretch=0.05, Q=10)
        ax.imshow(img_rgb, origin="lower",vmin=0.,vmax=vmax_3sig(img_rgb))
        xlim,ylim = ax.get_xlim(),ax.get_ylim()
        if centroid_type == "APER":
            aper.plot(color='violet',lw=2,ax=ax1)
            aper.plot(color='violet',lw=2,ax=ax2)
            aper.plot(color='violet',lw=2,ax=ax)
        ax.contour(data_em,colors="w",alpha=0.7,
                  levels = [0.1*data_em.max(),0.3*data_em.max(),0.5*data_em.max(),0.7*data_em.max(),data_em.max()])
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        

        ax.plot(x1, y1, color="seagreen", marker="+", ms=25,mew=3,alpha=0.9, zorder=4)
        ax.plot(x2, y2, "k", marker="+", ms=25,mew=3,alpha=0.9, zorder=3)
        ax.set_xticklabels(ax.get_xticks().astype("int")+obj_SE.y_min)
        ax.set_yticklabels(ax.get_yticks().astype("int")+obj_SE.x_min)
        ax.arrow(0.9,0.1,np.sin(pa)/15,np.cos(pa)/15,color="skyblue",
                 head_width=0.02, head_length=0.02,transform=ax.transAxes)
        ax.arrow(0.9,0.1,np.sin(clus_cen_angle)/15,np.cos(clus_cen_angle)/15,color="gold",
                 head_width=0.02, head_length=0.02,transform=ax.transAxes)
        ax.text(0.05,0.05,"$\\theta$: %.1f"%diff_angle,color="w",fontsize=15,transform=ax.transAxes)
        if np.ndim(pos_BCG)==2:
            text, color = ('NW', 'lightcoral') if lab==1 else ('SE', 'lightblue')
            ax.text(0.05, 0.9, text, color=color, fontsize=15, transform=ax.transAxes)
        plt.tight_layout()
    
    return diff_angle, centroid_offset, dist_clus_cen





def draw_centroid_offset(diff_centroids_v, diff_centroids_c, crit=[0.5, 0.8]):
    import seaborn as sns
    sns.distplot(diff_centroids_c[diff_centroids_c<np.percentile(diff_centroids_c,97.5)],
                 color="gray",label="All")
    sns.distplot(diff_centroids_v,color="seagreen",label="Candidate")
    p = np.percentile(diff_centroids_c, 100*crit[0])
    q = np.percentile(diff_centroids_c, 100*crit[1])
    plt.axvline(p, color="k",label="Med of Control = %.2f"%p)
    plt.axvline(q, color="k",ls='--',label="%d%% of Control = %.2f"%(100*crit[1],q))
    plt.xlabel("centroid offset",fontsize=14)
    plt.legend(loc='best',fontsize=12)
    
def draw_angle_control(diff_angles_c, diff_centroids_c, crit=[0.5, 0.8], b=7):
    import seaborn as sns
    p = np.percentile(diff_centroids_c, 100*crit[0])
    q = np.percentile(diff_centroids_c, 100*crit[1])

    plt.hist(diff_angles_c, bins=np.linspace(0,180,b), color="k",
             normed=True, histtype="step", linestyle=":",linewidth=3,alpha=0.9, label="centroid offset all")    
    plt.hist(diff_angles_c[diff_centroids_c>p], bins=np.linspace(0,180,b), color="k",
             normed=True, histtype="step", linestyle="--",linewidth=3,alpha=0.9, label="centroid offset > %.1f pix"%p)    
    plt.hist(diff_angles_c[diff_centroids_c>q], bins=np.linspace(0,180,b), color="k",
             normed=True, histtype="step", linewidth=3, alpha=0.9, label="centroid offset > %.1f pix"%q)  
    plt.xlabel("$\\theta$",fontsize=14)
    plt.legend(loc=8,fontsize=12)
    
def draw_angle_candidate(diff_angles_v, diff_centroids_v, diff_centroids_c, crit=[0.5, 0.8], b=7):    
    import seaborn as sns
    p = np.percentile(diff_centroids_c, 100*crit[0])
    q = np.percentile(diff_centroids_c, 100*crit[1])
    print("# of offset > %d%% :"%(100*crit[0]),np.sum(diff_centroids_v>p))
    print("# of offset > %d%% :"%(100*crit[1]),np.sum(diff_centroids_v>q))

    plt.hist(diff_angles_v[diff_centroids_v>p], bins=np.linspace(0,180,b),
             normed=True, histtype="step", linewidth=4, alpha=0.5, label="offset > %.1f pix"%p)
    plt.hist(diff_angles_v[diff_centroids_v>q], bins=np.linspace(0,180,b),
             normed=True, histtype="step", linewidth=4, alpha=0.5, label="offset > %.1f pix"%q)
    plt.hist(diff_angles_v, bins=np.linspace(0,180,b),
             normed=True, histtype="step", linewidth=4, alpha=0.5, label="candidate all")
    plt.xlabel("$\\theta$",fontsize=14)
    plt.legend(loc=9,fontsize=12)
    
def draw_angle_compare(diff_angles_v, diff_centroids_v, diff_angles_c, diff_centroids_c, crit=0.8, b=7):
    q = np.percentile(diff_centroids_c, crit*100)
    plt.hist(diff_angles_v,bins=np.linspace(0,180,b), color='orange',
        normed=True, histtype="step", linestyle=":", linewidth=4, alpha=0.5, label="candidate all")
    plt.hist(diff_angles_v[diff_centroids_v>q], bins=np.linspace(0,180,b),color='orange',
         normed=True, histtype="step", linewidth=4, alpha=0.5, label="candidate > %.1f pix"%q) 
    plt.hist(diff_angles_c, bins=np.linspace(0,180,b), color="k",
         normed=True, histtype="step", linestyle=":", linewidth=3, alpha=0.9, label="control all")  
    plt.hist(diff_angles_c[diff_centroids_c>q], bins=np.linspace(0,180,b), color="k",
         normed=True, histtype="step", linewidth=3, alpha=0.9, label="control > %.1f pix"%q)

    plt.xlabel("$\\theta$",fontsize=14)
    plt.legend(loc=9,fontsize=12)