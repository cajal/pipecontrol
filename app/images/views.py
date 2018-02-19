import sys

from ._utils import fill_nans
from ..schemata import tune, xcorr
from io import BytesIO
from . import images
import numpy as np
from ..schemata import experiment, shared, reso, meso, stack, pupil, treadmill, tune
from flask import render_template, redirect, url_for, flash, request, session, send_from_directory, make_response, \
    Response
import matplotlib.pyplot as plt
import mpld3
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import datajoint as dj
from ..schemata import stimulus

size_factor = dict(
    thumb=2, small=4, report=4, smedium=6.5, medium=8, marge=10, large=16, huge=32
)
corr_cmap = sns.blend_palette(['dodgerblue', 'steelblue', 'k', 'lime', 'orange'], as_cmap=True)

dj.config['external-analysis'] = dict(
    protocol='file',
    location='/mnt/scratch05/datajoint-store/analysis')


def savefig(fig, **kwargs):
    canvas = FigureCanvas(fig)
    png_output = BytesIO()
    canvas.print_png(png_output, dpi=50, **kwargs)
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    plt.close(fig)
    return response


@images.route("/oracle-<int:animal_id>-<int:session>-<int:scan_idx>-<int:field>_<size>.png")
def oracle_map(animal_id, session, scan_idx, field, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, field=field)
    print('Oracle', key)

    img = (tune.OracleMap() & key).fetch1('oracle_map')

    sz = tuple(i / max(*img.shape) * size_factor[size] for i in reversed(img.shape))

    fig, ax = plt.subplots(figsize=sz)

    ax.imshow(img, origin='lower', interpolation='nearest', cmap=corr_cmap, vmin=-1, vmax=1)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return savefig(fig)


@images.route("/correlation-<int:animal_id>-<int:session>-<int:scan_idx>-<int:field>_<size>.png")
def correlation_image(animal_id, session, scan_idx, field, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, field=field)
    base = meso if meso.ScanInfo() & key else reso

    img = (base.SummaryImages.Correlation() & key).fetch1('correlation_image')

    sz = tuple(i / max(*img.shape) * size_factor[size] for i in reversed(img.shape))
    fig, ax = plt.subplots(figsize=sz)

    ax.imshow(img, origin='lower', interpolation='nearest', cmap=corr_cmap, vmin=-1, vmax=1)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return savefig(fig)


@images.route("/average-<int:animal_id>-<int:session>-<int:scan_idx>-<int:field>_<size>.png")
def average_image(animal_id, session, scan_idx, field, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, field=field)
    base = meso if meso.ScanInfo() & key else reso
    img = (base.SummaryImages.Average() & key).fetch1('average_image')

    sz = tuple(i / max(*img.shape) * size_factor[size] for i in reversed(img.shape))
    fig, ax = plt.subplots(figsize=sz)

    ax.imshow(img, origin='lower', interpolation='nearest', cmap='gray')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return savefig(fig)


@images.route("/contrast_intensity-<int:animal_id>-<int:session>-<int:scan_idx>-<int:field>_<size>.png")
def contrast_intensity(animal_id, session, scan_idx, field, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, field=field)
    base = meso if meso.ScanInfo() & key else reso
    inten, contr = (base.Quality.MeanIntensity() * base.Quality.Contrast() & key).fetch1('intensities',
                                                                                         'contrasts')

    sz = tuple(i * size_factor[size] for i in [.9, .5])
    with  sns.plotting_context('talk' if size == 'huge' else 'paper'):
        with sns.axes_style('ticks'):
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=sz, sharex=True)
            ax.plot(inten, label='mean intensity', color='dodgerblue', lw=1)
            ax.set_ylabel('intensity')
            ax2.set_ylabel('contrast')
            ax2.plot(contr, label='contrast', color='deeppink', lw=1)
            ax2.set_xlabel('frame number')
            fig.tight_layout()
            fig.subplots_adjust(left=.2)
            sns.despine(fig, trim=True)
            for a in [ax, ax2]:
                a.spines['bottom'].set_linewidth(1)
                a.spines['left'].set_linewidth(1)
                a.tick_params(axis='both', length=3)
    return savefig(fig)


@images.route("/cos2map-<int:animal_id>-<int:session>-<int:scan_idx>-<int:field>_<size>.png")
def cos2map(animal_id, session, scan_idx, field, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, field=field)

    a, m = (tune.Cos2Map() & key).fetch1('direction_map', 'amplitude_map')
    h = (a / np.pi / 2) % 1
    v = np.minimum(m / np.percentile(m, 99.9), 1)

    img = mcolors.hsv_to_rgb(np.stack((h, v, v), axis=2))
    sz = tuple(i / max(*img.shape) * size_factor[size] for i in reversed(img.shape[:2]))
    fig, ax = plt.subplots(figsize=sz)
    ax.imshow(img, origin='lower', interpolation='nearest')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return savefig(fig)


@images.route("/oraclecourse-<int:animal_id>-<int:session>-<int:scan_idx>-<int:field>_<size>.png")
def oraclecourse(animal_id, session, scan_idx, field, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, field=field)
    t, movie, pearson = (tune.MovieOracleTimeCourse.OracleClipSet() \
                         * stimulus.Clip() & key).fetch('time', 'movie_name', 'pearson')
    sz = tuple(i * size_factor[size] for i in [1, 9 / 16])
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(figsize=sz)

        for t, oracle, label in zip(t, pearson, movie):
            mu, std = oracle.mean(axis=1), oracle.std(axis=1) / np.sqrt(oracle.shape[1])
            ax.fill_between(t, mu - std, mu + std, alpha=.1)
            ax.plot(t, mu, 'o-', label=label)
        ax.legend(ncol=3, loc='upper left', bbox_to_anchor=(0, 1.15))
        sns.despine(offset=5)
        ax.set_xlabel('scan time [s]')
        ax.set_ylabel('oracle correlation')
        fig.subplots_adjust(bottom=.2)
    return savefig(fig)


@images.route("/eye-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def eye(animal_id, session, scan_idx, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    frames = (pupil.Eye() & key).fetch1('preview_frames')

    sz = tuple(i * size_factor[size] for i in [1, 9 / 16])
    with sns.axes_style('white'):
        fig, ax = plt.subplots(4, 4, figsize=sz, sharex=True, sharey=True)
        vmin, vmax = frames.min(), frames.max()
        for fr, a in zip(frames.transpose([2, 0, 1]), ax.ravel()):
            a.imshow(fr, vmin=vmin, vmax=vmax, cmap='gray')
            a.axis('off')
        fig.set_facecolor('k')
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    return savefig(fig)


@images.route("/eye_tracking-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def eye_tracking(animal_id, session, scan_idx, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    r, center = (pupil.FittedContour.Ellipse() & key).fetch('major_r', 'center', order_by='frame_id ASC')
    detectedFrames = ~np.isnan(r)
    xy = np.full((len(r), 2), np.nan)
    xy[detectedFrames, :] = np.vstack(center[detectedFrames])
    xy = np.vstack(map(fill_nans, xy.T))
    pupil_radius = fill_nans(r.squeeze())
    eye_time = (pupil.Eye() & key).fetch1('eye_time').squeeze()
    eye_time = eye_time - eye_time[0]

    sz = tuple(i * size_factor[size] for i in [.9, .5])
    with  sns.plotting_context('talk' if size == 'huge' else 'paper'):
        with sns.axes_style('ticks'):
            fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=sz, sharex=True)
            ax.plot(eye_time, pupil_radius, color='k', lw=1)
            ax.set_ylabel('pupil radius')

            ax2.plot(eye_time, xy[0, :], color='k', lw=1)
            ax2.set_ylabel('x pos. [px]')

            ax3.plot(eye_time, xy[1, :], color='k', lw=1)
            ax3.set_ylabel('y pos. [px]')

            ax3.set_xlabel('scan time [s]')
            fig.tight_layout()
            fig.subplots_adjust(left=.1)
            sns.despine(fig, trim=True)
            for a in [ax, ax2]:
                a.spines['bottom'].set_linewidth(1)
                a.spines['left'].set_linewidth(1)
                a.tick_params(axis='both', length=3)
    return savefig(fig)


@images.route("/sta-<int:animal_id>-<int:session>-<int:scan_idx>_<int:t>_<quantile>_<size>.png")
def sta(animal_id, session, scan_idx, t, quantile, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, stimulus_type='stimulus.Monet2')

    if quantile == 'upper':
        rfs, keys = (tune.STA.Map() * tune.STAQual() & key).fetch('map', dj.key, order_by='snr DESC', limit=49)
    elif quantile == 'middle':
        med = np.median((tune.STAQual() & key).fetch('snr'))
        rfs, keys = (tune.STA.Map() * tune.STAQual() & key).fetch('map', dj.key,
                                                                  order_by='ABS(snr - {}) ASC'.format(med), limit=49)
    elif quantile == 'lower':
        rfs, keys = (tune.STA.Map() * tune.STAQual() & key).fetch('map', dj.key, order_by='snr ASC', limit=49)

    cmap = plt.cm.get_cmap('bwr')
    cmap._init()
    alphas = np.abs(np.linspace(-1.0, 1.0, cmap._lut[:, -1].shape[-1]))
    cmap._lut[:, -1] = alphas

    sz = tuple(i * size_factor[size] for i in [.9, .5])
    tt = np.linspace(0, 2 * np.pi, 100)
    cx, cy = np.cos(tt), np.sin(tt)
    with  sns.plotting_context('talk' if size == 'huge' else 'paper'):
        with sns.axes_style('ticks'):
            fig, ax = plt.subplots(7, 7, figsize=sz)
            for a, rf, k in zip(ax.ravel(), rfs, keys):
                rf = rf[..., t]
                v = np.abs(rf).max()
                a.matshow(rf, vmin=-v, vmax=v, cmap=cmap)
                if (tune.STAExtent() & k):
                    x, y, r = (tune.STAExtent() & k).fetch1('x', 'y', 'radius')
                    a.plot(x, y, 'o', ms=2 * size_factor[size], lw=.5, color='k', mfc='k', alpha=.2)
                    # a.plot(x + r * cx, y + r * cy, '--k')

                a.axis('off')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05, wspace=0.01, top=1, bottom=0, left=0, right=1)
    return savefig(fig)


@images.route("/sta_loc-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def sta_loc(animal_id, session, scan_idx, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    x, y = (tune.STAExtent() & key).fetch('x', 'y')
    sta = (tune.STA.Map() & key).fetch('map', limit=1)[0]
    ly, lx = sta.shape[:2]
    sz = tuple(i * size_factor[size] for i in [.9, .5])
    with  sns.plotting_context('talk' if size == 'huge' else 'paper'):
        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(figsize=sz)
            ax.scatter(x, y)
            ax.set_xlim((0, lx))
            ax.set_ylim((0, ly))
            ax.set_title('Estimated RF positions of n={} cell with RF-SNR>5'.format(len(x)))
            ax.set_xlabel('stimulus x-dimension')
            ax.set_ylabel('stimulus y-dimension')
        fig.tight_layout()
    return savefig(fig)


@images.route("/rf_snr-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def rf_snr(animal_id, session, scan_idx, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    snr = (tune.STAQual() & key).fetch('snr')
    sz = tuple(i * size_factor[size] for i in [.9, .5])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=2):
        with sns.axes_style('ticks'):
            fig, ax = plt.subplots(figsize=sz)
            g = sns.distplot(snr,
                             hist_kws=dict(cumulative=True),
                             kde_kws=dict(cumulative=True), ax=ax)
        sns.despine(ax=ax, trim=True)
        ax.set_xlabel('RF SNR')
        ax.set_ylabel('Cumulative Distribution')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.tick_params(axis='both', length=3, width=1)
        fig.tight_layout()
    return savefig(fig)


@images.route("/signal_xcorr-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def signal_xcorr(animal_id, session, scan_idx, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)
    v = (xcorr.XSNR() & key).fetch1('xsnr')

    sz = tuple(i * size_factor[size] for i in [.9, .5])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=2):
        with sns.axes_style('ticks'):
            fig, ax = plt.subplots(figsize=sz)
            g = sns.distplot(v,
                             hist_kws=dict(cumulative=True),
                             kde_kws=dict(cumulative=True), ax=ax)
        sns.despine(ax=ax, trim=True)
        ax.set_xlabel(r'$\frac{\sigma^2_{inter trial}}{\sigma^2_{inner trial} - \sigma^2_{inter trial}}$')
        ax.set_ylabel('Cumulative Distribution')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.tick_params(axis='both', length=3, width=1)
        fig.tight_layout()
    return savefig(fig)


@images.route("/pixelwiseori-<int:animal_id>-<int:session>-<int:scan_idx>-<int:field>_<size>.png")
def pixelwiseori(animal_id, session, scan_idx, field, size):
    from matplotlib import colors
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, field=field)

    def make_ori_map(xc):
        return colors.hsv_to_rgb(np.minimum(1, np.stack((np.angle(xc) / np.pi / 2 % 1, abs(xc), abs(xc)), axis=-1)))

    xc_monet, xc_trippy = (tune.PixelwiseOri() & key).fetch1('monet_map', 'trippy_map')

    sz = tuple(i * size_factor[size] for i in [.5, .8])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1):
        with sns.axes_style('ticks'):
            fig, ax = plt.subplots(2, 2, figsize=sz)
            ax = ax.ravel()
            ax[0].imshow(
                np.maximum(0,
                           np.minimum(1, 4 * (np.stack((abs(xc_monet), abs(xc_trippy), 0 * abs(xc_monet)), axis=-1)))))
            ax[0].set_title('Combo ori selectivity')
            ax[1].imshow(make_ori_map(5 * xc_trippy.conj()) ** 1.3)
            ax[1].set_title('Trippy tuning')
            ax[2].imshow(make_ori_map(5 * xc_monet) ** 1.3)
            ax[2].set_title('Monet tuning')
            ax[3].imshow(make_ori_map(3 * (xc_monet + xc_trippy)) ** 1.3)
            ax[3].set_title('Average tuning')
            [a.axis('off') for a in ax.ravel()]
            fig.subplots_adjust(top=.9, hspace=.1, wspace=.05, left=.01, right=.99, bottom=.01)
            plt.suptitle('Parametric tuning.  {animal_id}-{session}-{scan_idx} [{field}]'.format(**key))
    return savefig(fig)


@images.route("/cellwiseori-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def cellori(animal_id, session, scan_idx, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)
    r2, angle = (tune.Ori.Cell() & key).fetch('r2', 'angle')
    sz = tuple(i * size_factor[size] for i in [.5, .5])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1.3):
        with sns.axes_style('ticks'):
            sns.set_palette(sns.color_palette("Set2", 5))
            fig = plt.figure(figsize=sz)
            ax = fig.add_subplot(111)
            bins = np.linspace(-np.pi / 2, np.pi / 2, 15)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            for p in np.arange(20, 120, 20):
                h, _ = np.histogram(angle[r2 < np.percentile(r2, p)], normed=True, bins=bins)
                ax.plot(bin_centers,  h,  label='<{}% percentile R$^2$'.format(p))
            ax.legend(ncol=2)
            ax.set_xticks([-np.pi/2, -np.pi/4,  0, np.pi/4, np.pi/2])
            ax.set_xticklabels([r'$-\frac{\pi}{2}$',r'$\frac{\pi}{4}$', '0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'])
        sns.despine(trim=True)
        ax.set_ylabel('normalized histogram of orientations')
        ax.set_title('cardinal bias for {animal_id}-{session}-{scan_idx}'.format(**key))
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.tick_params(axis='both', length=3, width=1)

    return savefig(fig)

@images.route("/ori_r2-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def ori_r2(animal_id, session, scan_idx, size):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)

    r2 = (tune.Ori.Cell() & key).fetch('r2')
    sz = tuple(i * size_factor[size] for i in [.9, .5])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=2):
        with sns.axes_style('ticks'):
            fig, ax = plt.subplots(figsize=sz)
            g = sns.distplot(r2,
                             hist_kws=dict(cumulative=True),
                             kde_kws=dict(cumulative=True), ax=ax)
        sns.despine(ax=ax, trim=True)
        ax.set_xlabel(r'$R^2$')
        ax.set_ylabel('Cumulative Distribution')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.tick_params(axis='both', length=3, width=1)
        fig.tight_layout()
    return savefig(fig)
