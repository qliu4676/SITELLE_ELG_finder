from pipeline import *
from utils import *

# Arugument
name = sys.argv[1]
sn_thre = np.float(sys.argv[2])
sum_type = sys.argv[3]

z0 = 0.228
mode = "MMA"
output_dir = './output'
LPF = False

suffix = "_%s_sn%.1f"%(sum_type, sn_thre)
if LPF:
    print("Use LPF cube")
    suffix += '_lpf'

wavl_mask = [[7950,8006], [8020,8040], [8230,8280]]

save_path = os.path.join(output_dir, name)
centroid_path = os.path.join(output_dir, 'centroid', name)
check_save_path(centroid_path)

if LPF:
    cube_path = os.path.join(save_path, '%s_cube_lpf.fits'%name)
else:
    cube_path = os.path.join(save_path, '%s_cube.fits'%name)
    
table_path = os.path.join(save_path, '%s_%s_lpf.dat'%(name, mode))
seg_path = os.path.join(save_path, '%s_segm_%s_lpf.fits'%(name, mode))
deep_path = os.path.join(save_path, '%s_DF.fits'%(name))
mask_edge_path = os.path.join(save_path, 'Raw_stack_%s_mask.fits'%name)
spec_path = os.path.join(save_path, '%s-spec-%s_lpf.fits'%(name, mode))
template_path = os.path.join(output_dir, 'template')
CC_res_path = os.path.join(save_path, '%s-cc-%s_lpf.pkl'%(name, mode))
candidate_path = os.path.join(save_path, 'pic/candidate_%s_lpf'%mode)

# Read
datacube = Read_Datacube(cube_path, name, z0, 
                         mode=mode, wavl_mask=wavl_mask,
                         table=table_path, seg_map=seg_path,
                         deep_frame=deep_path,
                         mask_edge=mask_edge_path)
                         
datacube.get_wcs()
cube_detect_path = os.path.join(save_path, '%s_cube_lpf.fits'%name)
datacube.src_map = fits.getdata(cube_detect_path)
datacube.read_spec(spec_path)
datacube.read_template(template_path, n_intp=2, name=name, verbose=False)
datacube.read_cc_result(CC_res_path)

# BCG
datacube.assign_BCG_coordinate((328.40351201,17.69544013))
X_BCG, Y_BCG = datacube.wcs.all_world2pix(datacube.coord_BCG.ra, datacube.coord_BCG.dec, 0)
datacube.pos_BCG = (X_BCG, Y_BCG)

# Candidate
candidate_path_V = os.path.join(candidate_path,"V/%s#*.png"%name)
dir_V = glob.glob(candidate_path_V)
Num_V = np.sort(np.array([re.compile(r'\d+').findall(el)[-1] for el in dir_V]).astype("int"))

# Measure
print("Measure Centroid...")
datacube.centroid_analysis_all(Num_V, nums_obj=Num_V, centroid_type="ISO-D",
                               sn_thre=sn_thre, sum_type=sum_type, morph_cen=False, verbose=False)

datacube.centroid_analysis_all(Num_V, nums_obj=Num_V, centroid_type="ISO-D",
                               sn_thre=sn_thre, sum_type=sum_type, morph_cen=True, verbose=False)

# Save
datacube.save_centroid_measurement(Num_V, save_path=save_path,
                                   suffix=suffix, ID_field=name[-1])

# Plot histogram
z_V =  datacube.get_CC_result_best('z_best', 'Ha-NII_gauss', Num_V)

diff_angle_iso_d = datacube.get_centroid_result('diff_angle', 'ISO-D', fill_value=0)
diff_angle_std_iso_d = datacube.get_centroid_result('diff_angle_std', 'ISO-D', fill_value=99)
cen_off_iso_d = datacube.get_centroid_result('cen_offset', 'ISO-D', fill_value=0)
cen_off_std_iso_d = datacube.get_centroid_result('cen_offset_std', 'ISO-D', fill_value=99)

diff_angle_iso_dm = datacube.get_centroid_result('diff_angle', 'ISO-Dm', fill_value=0)
cen_off_iso_dm = datacube.get_centroid_result('cen_offset', 'ISO-Dm', fill_value=0)
cen_off_std_iso_dm = datacube.get_centroid_result('cen_offset_std', 'ISO-Dm', fill_value=99)

def condition_1(cen_off, cen_off_std, z_V):
    return (cen_off>1) & (cen_off>3*cen_off_std) & (abs(z_V-0.228)<0.015)

def condition_2(cen_off, cen_off_std, z_V):
    return (cen_off>1+3*cen_off_std) & (abs(z_V-0.228)<0.015)

d_angle_d1 = diff_angle_iso_d[condition_1(cen_off_iso_d, cen_off_std_iso_d, z_V)]
d_angle_dm1 = diff_angle_iso_dm[condition_1(cen_off_iso_dm, cen_off_std_iso_dm, z_V)]

d_angle_d2 = diff_angle_iso_d[condition_2(cen_off_iso_d, cen_off_std_iso_d, z_V)]
d_angle_dm2 = diff_angle_iso_dm[condition_2(cen_off_iso_dm, cen_off_std_iso_dm, z_V)]

fig, (ax1,ax2)=plt.subplots(1,2,figsize=(11,4))
ax1.hist(d_angle_d1-2, bins=np.linspace(0,180,7)-2,histtype="step",hatch="/", lw=4, alpha=0.7,label='ISO-1')
ax1.hist(d_angle_dm1, bins=np.linspace(0,180,7),histtype="step", lw=4, hatch="", alpha=0.7,label='morph-1')
ax1.legend()
ax2.hist(d_angle_d2-2, bins=np.linspace(0,180,7)-2,histtype="step",hatch="/", lw=4, alpha=0.7,label='ISO-2')
ax2.hist(d_angle_dm2, bins=np.linspace(0,180,7),histtype="step", lw=4, hatch="", alpha=0.7,label='morph-2')
ax2.legend()
