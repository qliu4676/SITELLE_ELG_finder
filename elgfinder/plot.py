import numpy as np
import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as plticker
import matplotlib.patheffects as PathEffects
from matplotlib import patches

plt.rcParams['text.usetex'] = True
plt.rcParams['image.origin'] = 'lower'
rcParams.update({'xtick.major.pad': '6.0'})
rcParams.update({'xtick.major.size': '8'})
rcParams.update({'xtick.major.width': '1.2'})
rcParams.update({'xtick.minor.pad': '6.0'})
rcParams.update({'xtick.minor.size': '6'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '6.0'})
rcParams.update({'ytick.major.size': '8'})
rcParams.update({'ytick.major.width': '1.2'})
rcParams.update({'ytick.minor.pad': '6.0'})
rcParams.update({'ytick.minor.size': '6'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'axes.labelsize': 14})
rcParams.update({'font.size': 14})

def colorbar(mappable, pad=0.2, size="5%", loc="right",
             direction='out', labelsize=10, length=10, **kwargs):
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
    cb = fig.colorbar(mappable, cax=cax, orientation=orent, **kwargs)
    cb.ax.tick_params(direction=direction, length=length, labelsize=labelsize)
    return cb

def draw_scale_bar(ax, X_bar=20, Y_bar=15, y_text=10,
                   scale=2*u.arcsec, pixel_scale=0.322,
                   lw=4, fontsize=10, color='w',
                   border_color='k', border_lw=0.5, alpha=1):
    """ Draw a scale bar """
    import matplotlib.patheffects as PathEffects
    if (scale).unit.is_equivalent(u.arcsec):
        L_bar = scale.to(u.arcsec).value/pixel_scale
        txt = "%d''"%(scale.value)
    elif (scale).unit.is_equivalent(u.kpc):
        L_bar = scale.to(u.kpc).value/pixel_scale
        txt = "%d kpc"%(scale.value)

    ax.plot([X_bar-L_bar/2, X_bar+L_bar/2], [Y_bar,Y_bar],
            color=color, alpha=alpha, lw=lw,
            path_effects=[PathEffects.SimpleLineShadow(), PathEffects.Normal()])
    ax.text(X_bar, y_text, txt, color=color, alpha=alpha,
            ha='center', va='center', fontweight='bold', fontsize=fontsize,
            path_effects=[PathEffects.SimpleLineShadow(),
            PathEffects.withStroke(linewidth=border_lw, foreground=border_color)])
            
            
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


def display_image(image_filename, width='600px'):
    from IPython.display import Image, HTML, display
    Image="<img style='width: {:s};' src='{:s}' />".format(width, image_filename)
    display(HTML(Image))

def display_images(image_filenames, width='600px'):
    from IPython.display import Image, HTML, display
    ImagesList=''.join( ["<img style='width: {:s}; margin: 0px; float: left; border: 1px solid black;' src='{:s}' />".format(width, filename)
                         for filename in sorted(image_filenames)])
    display(HTML(ImagesList))


def make_pdf(dir_name, name='A2390C', save_dir='./'):
    # collect images in dir_name and converted into a single pdf
    from fpdf import FPDF

    image_list = glob.glob(dir_name)

    pdf = FPDF('P', 'cm', 'A4') # 21x27.9
    pdf.set_font('Arial', '', 10)
    # imagelist is the list with all image filenames
    for k, fn in enumerate(image_list[:20]):
        d = k%3
        if d==0:
            pdf.add_page()
        pdf.image(fn, x=3.22, y=9.75*d+0.15, w=14.55, h=9.7)

        ind = os.path.basename(fn).lstrip(name).rstrip('.png')
        pdf.text(x=3.22, y=9.75*d+0.25, txt=ind)

    pdf.output(os.path.join(save_dir, '%s_candidates.pdf'%(name)), "F")
