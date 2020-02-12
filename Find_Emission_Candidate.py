import sys
import getopt
from pipeline import *
from utils import *
    
def main(argv):
    # File Path
    output_dir = './output'
    data_path = argv[0]
    deep_path = None
    
    # Cluster Info
    name = 'A2390C'
    z0 = 0.228
    
    wavl_mask = [[7950,8006], [8020,8040], [8230,8280]]
    
    # Parameter for Processing Raw Datacube
    box_size = 128
    kernel_size = [12, 3]

    # Parameter for Source Extraction
    mode = "MMA"
    sn_thre = 3
    nlevels = 64
    contrast_deblend = 0.01
    MMA_box = [5,3,3]
    suffix = ''
    
    plot_verbose = False
    verbose = False
    
    PROC_RAW = True
    EXTR_SPEC = True
    MAKE_TEMP = True
    CROS_CORR = True
    PLOT_CAND = True
    WRITE_TABLE = False
    
    # Get Script Options
    try:
        optlists, args = getopt.getopt(argv[1:], "n:z:m:d:K:vpw",
                                       ["NAME=", "Z=", "MODE=", "WAVL_MASK=",
                                        "OUT_DIR=", "DEEP_FRAME=", "skip=",
                                        "BOX_SIZE=", "KERNEL_SIZE=", "SN_THRE=", 
                                        "N_LEVELS=", "CONTRAST=", "MMA_BOX=", 
                                        "VERBOSE", "PLOT_VERBOSE", "WRITE", "suffix"])
        opts = [opt for opt, arg in optlists]        
        
    except getopt.GetoptError:
        sys.exit('Wrong Option.')
        
    for opt, arg in optlists:
        if opt in ("-n", "--NAME"):
            name = arg
        elif opt in ("-m", "--MODE"):
            mode = arg
            if mode not in ['MMA', 'stack']:
                sys.exit("Not available mode. Use: 'MMA' or 'stack'")
        elif opt in ("-z", "--Z"):
            z0 = np.float(arg)
        elif opt in ("-m", "--WAVL_MASK"):    
            wavl_mask = np.array(re.findall(r"\d*\.\d+|\d+", arg), dtype=float).reshape(-1,2)
        elif opt in ("--OUT_DIR"):
            output_dir = arg
        elif opt in ("-d", "--DEEP_FRAME"):
            deep_path = arg
        elif opt in ("--skip"):
            if 'raw' in arg : PROC_RAW = False
            if 'spec' in arg : EXTR_SPEC = False
            if 'temp' in arg : MAKE_TEMP = False
            if 'cc' in arg : CROS_CORR = False
            if 'out' in arg : PLOT_CAND = False
        elif opt in ("--BOX_SIZE"):
            box_size = np.int(arg)
        elif opt in ("-K","--KERNEL_SIZE"):
            kernel_size = np.array(re.findall(r"\d*\.\d+|\d+", arg), dtype=int)
            print(kernel_size)
        elif opt in ("--SN_THRE"):
            sn_thre = np.float(arg)
        elif opt in ("--N_LEVELS"):
            nlevels = np.float(arg)
        elif opt in ("--CONTRAST"):
            contrast_deblend = np.float(arg)
        elif opt in ("--MMA_BOX"):
            MMA_BOX = np.array(re.findall(r"\d*\.\d+|\d+", arg), dtype=int)
        elif opt in ("--OUT_DIR"):
            suffix = arg
            
    if ("--VERBOSE" in opts)|('-v' in opts): verbose = True
    if ("--PLOT_VERBOSE" in opts)|('-p' in opts): plot_verbose = True
    if ("--WRITE" in opts)|('-w' in opts):
        PROC_RAW, EXTR_SPEC, MAKE_TEMP, CROS_CORR, PLOT_CAND = [False] * 5
        WRITE_TABLE = True
    
    save_path = os.path.join(output_dir, name)
    check_save_path(save_path)
    
    cube_path = os.path.join(save_path, '%s_cube.fits'%name)
    table_path = os.path.join(save_path, '%s_%s_lpf.dat'%(name, mode))
    seg_path = os.path.join(save_path, '%s_segm_%s_lpf.fits'%(name, mode))
    mask_edge_path = os.path.join(save_path, 'Raw_stack_%s_mask.fits'%name)
    spec_path = os.path.join(save_path, '%s-spec-%s_lpf.fits'%(name, mode))
    template_path = os.path.join(output_dir, 'template')
    CC_res_path = os.path.join(save_path, '%s-cc-%s_lpf.pkl'%(name, mode))
    candidate_path = os.path.join(save_path, 'pic/candidate_%s_lpf'%mode)
    
    
    # output_dir = './output' 
    # name = 'A2390C'
    # name = 'A2390E'
    # name = 'A2390W'

    # data_path = "/home/qliu/data/A2390C4new.fits"
    # data_path = "/home/qliu/data/A2390F/A2390SEC4.fits"
    # data_path = "/home/qliu/data/A2390F/A2390NWC4.fits"

    #################################################
    # 1. Read raw cube and remove background + fringe
    #################################################

    def process_raw_cube():
        # Read raw SITELLE datacube
        raw_datacube = Read_Raw_SITELLE_datacube(data_path, name, wavl_mask=wavl_mask)

        # Mask map around field edges
        raw_datacube.save_mask_edge(save_path=save_path)

        # Remove large-scale background in each channel
        bkg_plot_path = os.path.join(save_path, 'pic/bkg_%d'%box_size)
        raw_datacube.remove_background(box_size, plot=plot_verbose,
                                       save_path=bkg_plot_path)

        # Save background subtracted cube
        raw_datacube.save_fits(save_path=save_path, suffix="")

        # Low-pass filtering in all channels to remove fringes
        channels, _ = raw_datacube.get_channel()
        raw_datacube.remove_fringe(channels, k_size=kernel_size,
                                   sn_source=3, method='LPF',
                                   save_path=bkg_plot_path,
                                   parallel=True, verbose=verbose,
                                   plot=plot_verbose)

        # Save low-pass filtered cube
        raw_datacube.save_fits(save_path=save_path, suffix="_lpf")

        del raw_datacube

    #################################################
    # 2. Extract source and spectra
    #################################################

    def extract_spectra():
        cube_detect_path = os.path.join(save_path, '%s_cube_lpf.fits'%name)

        # Read Processed datacube
        datacube = Read_Datacube(cube_path, name, z0, mode=mode,
                                 cube_detection=cube_detect_path,
                                 wavl_mask=wavl_mask,
                                 mask_edge=mask_edge_path)

        # Source detction use method according to 'mode'
        src_map, seg_map = datacube.ISO_source_detection(sn_thre=sn_thre,
                                                         b_size=box_size,
                                                         nlevels=nlevels,
                                                         contrast=contrast_deblend, 
                                                         box=MMA_box,
                                                         closing=False,
                                                         save=True, suffix="_lpf",
                                                         save_path=save_path)
        # Extract spectra of the detected source
        datacube.ISO_spec_extraction_all(seg_map, verbose=verbose)

        # Fit and remove continuum for cross-correlation below
        cont_plot_path = os.path.join(save_path, 'pic/fit_cont_%s'%mode)
        datacube.fit_continuum_all(model='GP', edge_ratio=0.15, 
                                   save_path=cont_plot_path,
                                   plot=plot_verbose, verbose=verbose)

        # Save spectra and continuum removed spectra
        datacube.save_spec_fits(save_path=save_path, suffix="_lpf")

        del datacube

    #################################################
    # 3. Build templates
    #################################################

    def build_template():
        datacube = Read_Datacube(cube_path, name, z0, mode=mode)

        # build gaussian templates for different lines
        datacube.generate_template(n_ratio=1, n_stddev=10, n_intp=2,
                                   temp_type="OII", temp_model='gauss',
                                   ratio_prior="log-uniform", verbose=verbose)
        datacube.generate_template(n_ratio=8, n_stddev=10, n_intp=2, verbose=verbose,
                                   ratio_range = (2., 4.), ratio_prior="log-uniform", 
                                   temp_type="Hb-OIII", temp_model='gauss')
        datacube.generate_template(n_ratio=40, n_stddev=10, n_intp=2, verbose=verbose,
                                   ratio_range = (1., 8), ratio_prior="log-uniform", 
                                   temp_type="Ha-NII", temp_model='gauss')

        # Save templates
        for line in ["Ha-NII", "Hb-OIII", "OII"]:
            datacube.save_template(save_path=template_path,
                                   temp_type=line, temp_model='gauss')
        del datacube

    #################################################
    # 4. Cross-correlation
    #################################################

    def cross_correlation():
        # Read Processed datacube
        datacube = Read_Datacube(cube_path, name, z0, mode=mode,
                                 table=table_path, mask_edge=mask_edge_path)

        # Read spectra and template
        datacube.read_spec(spec_path)
        datacube.read_template(template_path, n_intp=2, name=name, verbose=verbose)

        # Cross-correlation
        for line in ["Ha-NII", "Hb-OIII", "OII"]:
            datacube.cross_correlation_all(temp_type=line, temp_model="gauss",
                                           edge=20, verbose=False)
        print("Cross-correlation Finished!\n")
 
        # Save CC results
        datacube.save_cc_result(save_path=save_path, suffix="_lpf")

        del datacube

    #################################################
    # 5. Select Candidate
    #################################################

    def save_candidate(save=True):

        datacube = Read_Datacube(cube_path, name, z0, mode=mode,
                                 wavl_mask=wavl_mask, table=table_path,
                                 deep_frame=deep_path, 
                                 mask_edge=mask_edge_path)

        datacube.read_spec(spec_path)
        datacube.read_template(template_path, n_intp=2, name=name, verbose=verbose)
        datacube.read_cc_result(CC_res_path)

        src_map_path = os.path.join(save_path, '%s_%s_lpf.fits'%(name, mode))
        datacube.src_map = fits.getdata(src_map_path)

        # S/N from cross correlation
        SNR_best_Ha_gauss = datacube.get_CC_result_best('SNR', 'Ha-NII_gauss')
        SNR_best_OIII_gauss = datacube.get_CC_result_best('SNR', 'Hb-OIII_gauss')
        SNR_best_OII_gauss = datacube.get_CC_result_best('SNR', 'OII_gauss')

        z_best_OIII_gauss = datacube.get_CC_result_best('z_best', 'Hb-OIII_gauss')

        # Match with SDSS stars
        cat_match = datacube.match_sdss_star(sep=3*u.arcsec,
                                             search_radius=7*u.arcmin, band='rmag')
        not_star = ~np.array([num in cat_match["NUMBER"]
                              for num in datacube.obj_nums], dtype=bool)

        # Remove sources which are too close to field edges
        dist_to_edge = measure_dist_to_edge(datacube.table, datacube.mask_edge, pad=200)
        edge_cond = dist_to_edge > 5

        # Remove sources with low Equivalent Width (of combined lines)
        datacube.estimate_EW_all(MC_err=True, verbose=verbose)
        EW_cond = (datacube.EWs > 5)&(datacube.EWs > datacube.EW_stds)

        # Remove dubious source with max in flagged wavl but low continuum 
        bad_channel, _ = datacube.get_channel(wavl_mask)
        max_channel = np.argmax(datacube.obj_specs_opt, axis=1) + 1
        flag_max = np.array([max_ch in bad_channel for max_ch in max_channel])
        low_cont = np.median(datacube.obj_specs_opt, axis=1) < 0.1
        good_line = ~(flag_max & low_cont)
        cond0 = edge_cond & EW_cond & good_line & not_star
        
        # S/N Condition

        # SN Ha > 10, SN OII > 3
        SNR_cond_A = (SNR_best_Ha_gauss > 10) & (SNR_best_OII_gauss>3)

        # SN Ha > 5, SN OII > 5
        SNR_cond_B = (SNR_best_Ha_gauss > 5) & (SNR_best_Ha_gauss <= 10) & (SNR_best_OII_gauss>5)


        # SN OIII > 5, SN OIII > SN Ha, SN OIII > SN OII 
        zmin_OIII = datacube.wavl.min()/4959-1
        SNR_cond_C = (SNR_best_OIII_gauss > np.max([5*np.ones_like(datacube.obj_nums),
                                                    SNR_best_Ha_gauss], axis=0)) \
                    & (SNR_best_OII_gauss > 3) & (z_best_OIII_gauss > zmin_OIII)

        # Plot
        print("\nSeparate candidates into three subsamples based on S/N.")
        SN_cond_v = {'A':SNR_cond_A, 'B':SNR_cond_B, 'C':SNR_cond_C}
        type_v = {'A':"Ha-NII", 'B':"Ha-NII", 'C':"Hb-OIII"}

        for v in ['A', 'B', 'C']:
            cond = SN_cond_v[v] & cond0
            num_c = datacube.obj_nums[cond]

            # Remove stars and possible artifacts
            num_c = np.setdiff1d(num_c, np.concatenate([cat_match["NUMBER"].data,
                                                        datacube.num_spurious]))
            if verbose:
                print("%s: %d candidates:\n"%(v, len(num_c)), num_c)

            if save:
                check_save_path(candidate_path+'/%s'%v, clear=True)
                with tqdm(total=len(num_c), file=sys.stdout, desc='Candidate: ') as pbar:
                    for k, num in enumerate(num_c):

                        datacube.plot_candidate(num, mode=mode,
                                                temp_type=type_v[v], temp_model="gauss")
                        plt.savefig(os.path.join(candidate_path+'/%s'%v,
                                                 "%s#%d.png"%(name, num)), dpi=75)
                        plt.close()

                        if (np.mod(k+1, 5)==0):
                            pbar.update(5)
                    else:
                        pbar.update(np.mod(k+1, 5))

        if save:
            check_save_path(candidate_path+'/V', clear=True)
            
    def write_table(suffix=""):
        
        table_all = Table.read(table_path, format='ascii')
        candidate_path_C = os.path.join(candidate_path,"C/%s#*.png"%name)
        dir_C = glob.glob(candidate_path_C)
        Num_C = np.sort(np.array([re.compile(r'\d+').findall(el)[-1] for el in dir_C]).astype("int"))
        
        candidate_path_V = os.path.join(candidate_path,"V/%s#*.png"%name)
        dir_V = glob.glob(candidate_path_V)
        Num_V = np.sort(np.array([re.compile(r'\d+').findall(el)[-1] for el in dir_V]).astype("int"))
        
        if len(Num_V)==0:
            use_all = input("%s is empty. Use visual inspection? [y/n]"%candidate_path_V)
            if use_all == 'y':
                sys.exit("Check %s and manually move candidate to %s/V/"\
                         %(candidate_path, candidate_path))
            else:
                candidate_path_sub = os.path.join(candidate_path,"?/%s#*.png"%name)
                dir_V = glob.glob(candidate_path_sub)
                Num_V = np.unique(np.sort(np.array([re.compile(r'\d+').findall(el)[-1] for el in dir_V]).astype("int")))
                
        table_target = join(Table({"NUMBER":Num_V}), table_all)
        table_target['flag'] = np.ones(len(Num_V), dtype=int)
        
        for num in Num_C:
            irow = np.where(table_target["NUMBER"]==num)[0][0]
            table_target['flag'][irow] = 2
        
        f_name = os.path.join(save_path,'%s_ELG_list%s.txt'%(name,suffix))
        table_target.write(f_name, format='ascii', overwrite=True)
        print("Save candidate list as :", f_name)
        
    # Execute
    process_raw_cube() if PROC_RAW else print("Skip processing raw datacube.")
    extract_spectra() if EXTR_SPEC else print("Skip extraction of spectra.")
    build_template() if MAKE_TEMP else print("Skip building template.")
    cross_correlation() if CROS_CORR else print("Skip cross-correlation.")
    save_candidate(save=True) if PLOT_CAND else print("Skip ploting ELG candidate.")
        
    if WRITE_TABLE: write_table(suffix)

    return opts
    
    
if __name__ == "__main__":
    main(sys.argv[1:])