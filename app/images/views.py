import sys

from ._utils import fill_nans
from ..schemata import tune, xcorr, stack
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
import pandas as pd

SETTINGS = dict(spike_method=5, segmentation_method=6)

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

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, field=field, **SETTINGS)

    img = (tune.OracleMap() & key).fetch1('oracle_map')

    sz = tuple(i / max(*img.shape) * size_factor[size] for i in reversed(img.shape))

    fig, ax = plt.subplots(figsize=sz)

    ax.imshow(img, origin='lower', interpolation='nearest', cmap=corr_cmap, vmin=-1, vmax=1)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return savefig(fig)


@images.route('/correlation-<int:animal_id>-<int:session>-<int:scan_idx>-<int:field>-<int:channel>_<size>.png')
def correlation_image(animal_id, session, scan_idx, field, channel, size):

    key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx,
           'field': field, 'channel': channel}
    pipe = reso if reso.ScanInfo() & key else meso
    img = (pipe.SummaryImages.Correlation() & key).fetch1('correlation_image')

    sz = tuple(i / max(*img.shape) * size_factor[size] for i in reversed(img.shape))
    fig, ax = plt.subplots(figsize=sz)

    ax.imshow(img, origin='lower', interpolation='nearest', cmap=corr_cmap, vmin=-1, vmax=1)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return savefig(fig)


@images.route('/average-<int:animal_id>-<int:session>-<int:scan_idx>-<int:field>-<int:channel>_<size>.png')
def average_image(animal_id, session, scan_idx, field, channel, size):

    key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx,
           'field': field, 'channel': channel}
    pipe = reso if reso.ScanInfo() & key else meso
    img = (pipe.SummaryImages.Average() & key).fetch1('average_image')

    sz = tuple(i / max(*img.shape) * size_factor[size] for i in reversed(img.shape))
    fig, ax = plt.subplots(figsize=sz)

    ax.imshow(img, origin='lower', interpolation='nearest', cmap='gray')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return savefig(fig)


@images.route('/contrast_intensity-<int:animal_id>-<int:session>-<int:scan_idx>-<int:field>-<int:channel>_<size>.png')
def contrast_intensity(animal_id, session, scan_idx, field, channel,size):

    key = {'animal_id':animal_id, 'session':session, 'scan_idx':scan_idx, 'field':field, 'channel': channel}
    pipe = reso if reso.ScanInfo() & key else meso
    intensities= (pipe.Quality.MeanIntensity()  & key).fetch1('intensities')
    contrasts= (pipe.Quality.Contrast() & key).fetch1('contrasts')

    sz = tuple(i * size_factor[size] for i in [.9, .5])
    with  sns.plotting_context('talk' if size == 'huge' else 'paper'):
        with sns.axes_style('ticks'):
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=sz, sharex=True)
            ax.plot(intensities, color='dodgerblue', lw=1)
            ax.set_ylabel('intensity')
            ax.set_title('mean intensity')
            ax2.set_ylabel('contrast')
            ax2.plot(contrasts, color='deeppink', lw=1)
            ax2.set_xlabel('frame number')
            ax2.set_title('contrast')
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

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, field=field, **SETTINGS)

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

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, field=field, **SETTINGS)
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

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, **SETTINGS)

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

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, **SETTINGS)

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

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, stimulus_type='stimulus.Monet2', **SETTINGS)

    snr = (tune.STA.Map() * tune.STAQual() & key).fetch('snr')

    quantile = np.percentile(snr, quantile)

    rfs, keys = (tune.STA.Map() * tune.STAQual() & key & 'snr<{}'.format(quantile)).fetch('map', dj.key, order_by='snr DESC', limit=49)

    cmap = plt.cm.get_cmap('bwr')
    cmap._init()
    alphas = np.abs(np.linspace(-1.0, 1.0, cmap._lut[:, -1].shape[-1]))
    cmap._lut[:, -1] = alphas

    sz = tuple(i * size_factor[size] for i in [.9, .5])
    # tt = np.linspace(0, 2 * np.pi, 100)
    # cx, cy = np.cos(tt), np.sin(tt)
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

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, **SETTINGS)

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

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, **SETTINGS)

    snr = (tune.STAQual() & key).fetch('snr')
    perc_low, perc_high = np.percentile(snr, [1, 99])
    sz = tuple(i * size_factor[size] for i in [.9, .5])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=2):
        with sns.axes_style('ticks'):
            fig, ax = plt.subplots(figsize=sz)
            g = sns.distplot(snr,
                             hist_kws=dict(cumulative=True),
                             kde_kws=dict(cumulative=True), ax=ax)
            ax.set_xlim((perc_low, perc_high))
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

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, **SETTINGS)
    subsets, rr = (xcorr.XNR() & key).fetch('subset', 'xnr')

    sz = tuple(i * size_factor[size] for i in [.9, .5])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=2):
        with sns.axes_style('ticks'):
            fig, ax = plt.subplots(figsize=sz)
            for r, lab in zip(rr, subsets):
                ax.hist(r[~np.isnan(r)], np.r_[0:0.5:1000j], cumulative=-1,
                        normed=True, histtype='step', label=lab, lw=3)
        ax.set_ylabel('# cells')
        ax.set_xlim([0, .5])
        ax.legend()
        ax.set_title('{animal_id}-{session}-{scan_idx}'.format(**key))
        ax.set_xlabel('Total signal correlation')
        ax.grid(True)
        sns.despine(trim=True)

        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.tick_params(axis='both', length=3, width=1)
        fig.subplots_adjust(bottom=.2)
    return savefig(fig)


@images.route("/pixelwiseori-<int:animal_id>-<int:session>-<int:scan_idx>-<int:field>_<size>.png")
def pixelwiseori(animal_id, session, scan_idx, field, size):

    from matplotlib import colors
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, field=field, **SETTINGS)

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

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, **SETTINGS)
    osi, r2, angle = (tune.Ori.Cell() & key).fetch('selectivity', 'r2', 'angle')
    sz = tuple(i * size_factor[size] for i in [.5, .5])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1.3):
        with sns.axes_style('ticks'):
            sns.set_palette(sns.color_palette("Set2", 5))
            fig = plt.figure(figsize=sz)
            ax = fig.add_subplot(111)
            bins = np.linspace(-np.pi / 2, np.pi / 2, 15)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            for p in np.arange(0, 100, 20):
                # h, _ = np.histogram(angle[r2 < np.percentile(r2, p)], normed=True, bins=bins)
                h, _ = np.histogram(angle[osi > np.percentile(osi, p)], normed=True, bins=bins)
                ax.plot(bin_centers, h, label='>{}% OSI'.format(p), lw=3)
            ax.legend(ncol=2)
            ax.set_xticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
            ax.set_xticklabels([r'$-\frac{\pi}{2}$', r'$\frac{\pi}{4}$', '0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'])
        sns.despine(trim=True)
        ax.set_ylabel('normalized histogram of orientations')
        ax.set_title('cardinal bias for {animal_id}-{session}-{scan_idx}'.format(**key))
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.tick_params(axis='both', length=3, width=1)

    return savefig(fig)


@images.route("/ori_r2-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def ori_r2(animal_id, session, scan_idx, size):

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, **SETTINGS)

    r2 = (tune.Ori.Cell() & key & dict(ori_type='ori')).fetch('r2')
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


@images.route("/ori_r2-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def dir_r2(animal_id, session, scan_idx, size):

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, **SETTINGS)

    r2 = (tune.Ori.Cell() & key & dict(ori_type='dir')).fetch('r2')
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


@images.route("/osi-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def osi(animal_id, session, scan_idx, size):

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, **SETTINGS)

    osi = (tune.Ori.Cell() & key & dict(ori_type='ori')).fetch('selectivity')
    perc = np.percentile(osi, 98)
    sz = tuple(i * size_factor[size] for i in [.9, .5])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=2):
        with sns.axes_style('ticks'):
            fig, ax = plt.subplots(figsize=sz)
            g = sns.distplot(osi,
                             hist_kws=dict(cumulative=True),
                             kde_kws=dict(cumulative=True), ax=ax)
        ax.set_xlim((osi.min(), perc))
        sns.despine(ax=ax, trim=True)
        ax.set_xlabel(r'$R^2$')
        ax.set_ylabel('Cumulative Distribution')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.tick_params(axis='both', length=3, width=1)
        fig.tight_layout()
    return savefig(fig)


@images.route("/dsi-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def dsi(animal_id, session, scan_idx, size):

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, **SETTINGS)
    dsi = (tune.Ori.Cell() & key & dict(ori_type='dir')).fetch('selectivity')
    perc = np.percentile(dsi, 98)
    sz = tuple(i * size_factor[size] for i in [.9, .5])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=2):
        with sns.axes_style('ticks'):
            fig, ax = plt.subplots(figsize=sz)
            g = sns.distplot(dsi,
                             hist_kws=dict(cumulative=True),
                             kde_kws=dict(cumulative=True), ax=ax)
        ax.set_xlim((dsi.min(), perc))
        sns.despine(ax=ax, trim=True)
        ax.set_xlabel(r'$R^2$')
        ax.set_ylabel('Cumulative Distribution')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.tick_params(axis='both', length=3, width=1)
        fig.tight_layout()
    return savefig(fig)


@images.route("/osi_vs_r2-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def osi_vs_r2(animal_id, session, scan_idx, size):

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, **SETTINGS)

    r2, osi = (tune.Ori.Cell() & key & dict(ori_type='ori')).fetch('r2', 'selectivity')

    sz = tuple(i * size_factor[size] for i in [.7, .7])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1.3):
        with sns.axes_style('ticks'):
            # g = sns.jointplot(osi, r2, marginal_kws=dict(hist_kws=dict(cumulative=True),
            #                                              kde_kws=dict(cumulative=True)), )
            g = sns.JointGrid(osi, r2)
            g = g.plot_joint(plt.scatter, color='#334f6d', s=1)
            g = g.plot_marginals(sns.distplot, color='#334f6d', hist_kws=dict(cumulative=True),
                                 kde_kws=dict(cumulative=True))
            perc = np.percentile(osi, 98)
            g.ax_joint.set_xlim((osi.min(), perc))
            perc = np.percentile(r2, 98)
            g.ax_joint.set_ylim((r2.min(), perc))
            plt.setp(g.ax_marg_y.get_xticklabels(), visible=True, rotation=-60)
            plt.setp(g.ax_marg_y.xaxis.get_majorticklines(), visible=True)
            plt.setp(g.ax_marg_y.xaxis.get_minorticklines(), visible=True)
            g.ax_marg_y.set_xticks(np.linspace(0, 1, 5))
            g.ax_marg_y.tick_params(axis='both', length=3, width=1)
            g.ax_marg_y.grid('on')
            sns.despine(left=False, trim=True, ax=g.ax_marg_y)

            plt.setp(g.ax_marg_x.get_yticklabels(), visible=True)
            plt.setp(g.ax_marg_x.yaxis.get_majorticklines(), visible=True)
            plt.setp(g.ax_marg_x.yaxis.get_minorticklines(), visible=True)
            g.ax_marg_x.set_yticks(np.linspace(0, 1, 5))
            g.ax_marg_x.tick_params(axis='both', length=3, width=1)
            g.ax_marg_x.grid('on')
            sns.despine(left=False, trim=True, ax=g.ax_marg_x)

            sns.despine(ax=g.ax_joint, trim=True)
            g.ax_joint.grid('on')
        g.fig.set_size_inches(sz)
        ax = g.ax_joint
        ax.set_xlabel('OSI')
        ax.set_ylabel(r'$R^2$')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.tick_params(axis='both', length=3, width=1)
        g.fig.subplots_adjust(left=.2, bottom=.15)
    return savefig(g.fig)


@images.route("/dsi_vs_r2-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def dsi_vs_r2(animal_id, session, scan_idx, size):

    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx, **SETTINGS)

    r2, dsi = (tune.Ori.Cell() & key & dict(ori_type='dir')).fetch('r2', 'selectivity')
    sz = tuple(i * size_factor[size] for i in [.7, .7])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1.3):
        with sns.axes_style('ticks'):
            g = sns.JointGrid(dsi, r2)
            g = g.plot_joint(plt.scatter, color='#334f6d', s=1)
            perc = np.percentile(dsi, 98)
            g.ax_joint.set_xlim((dsi.min(), perc))
            perc = np.percentile(r2, 98)
            g.ax_joint.set_ylim((r2.min(), perc))

            g = g.plot_marginals(sns.distplot, color='#334f6d', hist_kws=dict(cumulative=True),
                                 kde_kws=dict(cumulative=True))

            plt.setp(g.ax_marg_y.get_xticklabels(), visible=True, rotation=-60)
            plt.setp(g.ax_marg_y.xaxis.get_majorticklines(), visible=True)
            plt.setp(g.ax_marg_y.xaxis.get_minorticklines(), visible=True)
            g.ax_marg_y.set_xticks(np.linspace(0, 1, 5))
            g.ax_marg_y.tick_params(axis='both', length=3, width=1)
            g.ax_marg_y.grid('on')
            sns.despine(left=False, trim=True, ax=g.ax_marg_y)

            plt.setp(g.ax_marg_x.get_yticklabels(), visible=True)
            plt.setp(g.ax_marg_x.yaxis.get_majorticklines(), visible=True)
            plt.setp(g.ax_marg_x.yaxis.get_minorticklines(), visible=True)
            g.ax_marg_x.set_yticks(np.linspace(0, 1, 5))
            g.ax_marg_x.tick_params(axis='both', length=3, width=1)
            g.ax_marg_x.grid('on')
            sns.despine(left=False, trim=True, ax=g.ax_marg_x)

            sns.despine(ax=g.ax_joint, trim=True)
            g.ax_joint.grid('on')
        g.fig.set_size_inches(sz)
        ax = g.ax_joint
        ax.set_xlabel('DSI')
        ax.set_ylabel(r'$R^2$')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.tick_params(axis='both', length=3, width=1)
        g.fig.subplots_adjust(left=.2, bottom=.15)
    return savefig(g.fig)


@images.route("/mouse_per_scan_oracle-<int:animal_id>_<size>.png")
def mouse_per_scan_oracle(animal_id, size):

    key = dict(animal_id=animal_id, **SETTINGS)
    sz = tuple(i * size_factor[size] for i in [.7, .7])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1.3):
        with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
            df = pd.DataFrame((tune.MovieOracle.Total() & key).fetch(order_by='session ASC, scan_idx ASC'))
            df['scan'] = ['{}-{}-{}'.format(ai, s, sa) for ai, s, sa in zip(df.animal_id, df.session, df.scan_idx)]

            # Initialize the FacetGrid object
            N = len(dj.U('session', 'scan_idx') & (tune.MovieOracle() & key))

            pal = sns.cubehelix_palette(N, rot=-.25, light=.7)
            g = sns.FacetGrid(df, row="scan", hue="scan", palette=pal)

            # Draw the densities in a few steps
            g.map(sns.kdeplot, "pearson", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.01)
            g.map(sns.kdeplot, "pearson", clip_on=False, color="w", lw=2, bw=.01)
            g.map(plt.axhline, y=0, lw=2, clip_on=False)

            # Define and use a simple function to label the plot in axes coordinates
            def label(x, color, label):
                ax = plt.gca()
                low, high = ax.get_xlim()
                low -= .02
                ax.set_xlim((low, high))
                ax.text(0, .2, label, fontweight="bold", color=color,
                        ha="left", va="center", transform=ax.transAxes)

            g.map(label, "scan")

            # Set the subplots to overlap
            g.fig.subplots_adjust(bottom=.1, hspace=-0.25, top=.9)

            g.fig.set_size_inches(sz)

            # Remove axes details that don't play will with overlap
            g.set_titles("")
            g.fig.suptitle("Movie Oracle Correlations")
            g.set(yticks=[])
            g.despine(bottom=True, left=True)
            g.axes.ravel()[-1].set_xlabel('Pearson Correlation')

    return savefig(g.fig)


@images.route("/kuiper-<int:animal_id>_<size>.png")
def kuiper(animal_id, size):

    key = dict(animal_id=animal_id, **SETTINGS)
    sz = tuple(i * size_factor[size] for i in [.7, .7])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1.3):
        with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
            df = pd.DataFrame((tune.Kuiper() & key).fetch())

            # Draw the densities in a few steps
            g = sns.FacetGrid(df, hue='stimulus_type', col='ori_type', col_order=['ori', 'dir'],
                              margin_titles=True,
                              legend_out=False)

            def scatter(x, y, **kwargs):
                plt.scatter(x, y, **kwargs)
                plt.xlim((0, .4))
                plt.ylim((0, .4))

            g.map(scatter, "kuiper", "widest_gap", s=5)

            def label(**kwargs):
                ax = plt.gca()
                ax.spines['bottom'].set_linewidth(1)
                ax.spines['left'].set_linewidth(1)
                ax.tick_params(axis='both', length=3, width=1)

            g.map(label)

            g.fig.set_size_inches(sz)

            # Remove axes details that don't play will with overlap
            g.set_titles("")
            g.fig.suptitle("circular uniformity")
            g.despine(trim=True)
            g.add_legend()
            g.fig.subplots_adjust(left=.15)
            g.set_xlabels('kuiper statistic')
            g.set_ylabels('widest gap')

    return savefig(g.fig)


@images.route("/mouse_per_stack_oracle-<int:animal_id>_<size>.png")
def mouse_per_stack_oracle(animal_id, size):

    key = dict(animal_id=animal_id, **SETTINGS)
    sz = tuple(i * size_factor[size] for i in [.7, .7])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1.3):
        with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
            rel = (stack.StackSet.Unit()).aggr(stack.StackSet.Match().proj('munit_id', session='scan_session')
                                               * tune.MovieOracle.Total() & key, pearson='MAX(pearson)')
            df = pd.DataFrame(rel.fetch(order_by='stack_session ASC, stack_idx ASC'))
            df['stack'] = ['{}-{}-{}'.format(ai, s, sa) for ai, s, sa in
                           zip(df.animal_id, df.stack_session, df.stack_idx)]

            # Initialize the FacetGrid object
            N = len(dj.U('stack_session', 'stack_idx') & rel)

            pal = sns.cubehelix_palette(N, rot=-.25, light=.7)

            if N > 1:
                g = sns.FacetGrid(df, row="stack", hue='stack', aspect=15, size=.5, palette=pal)

                # Draw the densities in a few steps
                g.map(sns.kdeplot, "pearson", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.01)
                g.map(sns.kdeplot, "pearson", clip_on=False, color="w", lw=2, bw=.01)
                g.map(plt.axhline, y=0, lw=2, clip_on=False)

                # Define and use a simple function to label the plot in axes coordinates
                def label(x, color, label):
                    ax = plt.gca()
                    low, high = ax.get_xlim()
                    low -= .02
                    ax.set_xlim((low, high))
                    ax.text(0, .2, label, fontweight="bold", color=color,
                            ha="left", va="center", transform=ax.transAxes)

                g.map(label, "stack")

                # Set the subplots to overlap
                g.fig.subplots_adjust(hspace=-.25, top=.9)

                g.fig.set_size_inches(sz)

                # Remove axes details that don't play will with overlap
                g.set_titles("")
                g.fig.suptitle("Movie Oracle Correlations")
                g.set(yticks=[])
                g.despine(bottom=True, left=True)
                g.axes.ravel()[-1].set_xlabel('Pearson Correlation')
                return savefig(g.fig)
            else:
                with sns.axes_style('ticks', rc={"axes.facecolor": (0, 0, 0, 0)}):
                    fig, ax = plt.subplots()

                # Draw the densities in a few steps
                sns.kdeplot(df.pearson, shade=True, alpha=1, lw=1.5, bw=.01, label='n={} neurons'.format(len(df)))
                # Set the subplots to overlap
                fig.set_size_inches(sz)

                # Remove axes details that don't play will with overlap
                fig.suptitle("Movie Oracle Correlations Stack " + np.unique(df['stack']).item())
                ax.set(yticks=[])
                low, high = ax.get_ylim()
                ax.set_ylim((0, high))
                sns.despine(fig=fig, trim=True, left=True)
                ax.set_xlabel('Pearson Correlation')
                ax.spines['bottom'].set_linewidth(1)
                ax.tick_params(axis='both', length=3, width=1)

                return savefig(fig)


@images.route("/osi_dsi_per_stack-<int:animal_id>_<size>.png")
def osi_dsi_per_stack(animal_id, size):

    key = dict(animal_id=animal_id, **SETTINGS)
    df1 = pd.DataFrame((stack.StackSet.Match() & key).fetch())
    df1['session'] = df1['scan_session']
    df2 = pd.DataFrame((tune.Ori.Cell() & key).fetch())
    df = df1.merge(df2).groupby(
        ['animal_id', 'stack_session', 'stack_idx',
         'munit_id', 'ori_type', 'stimulus_type']).agg(dict(selectivity=np.max)).reset_index()
    df['stack'] = ['{}-{}-{}'.format(ai, s, sa) for ai, s, sa in
                   zip(df.animal_id, df.stack_session, df.stack_idx)]

    df['stimulus_type'] = [s.split('.')[1] for s in df['stimulus_type']]

    sz = tuple(i * size_factor[size] for i in [1, .7])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1.3):
        with sns.axes_style(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}):
            # Initialize the FacetGrid object
            pal = ['steelblue', 'orange']
            g = sns.FacetGrid(df, row="stack", hue='stimulus_type', col='ori_type', palette=pal,
                              col_order=['ori', 'dir'], margin_titles=True, legend_out=True)
            #
            # # Draw the densities in a few steps
            g.map(sns.kdeplot, "selectivity", shade=False, alpha=.5, lw=3, cumulative=True)
            g.add_legend(title="Stimulus Type", prop={'size': 8})
            # g.map(sns.kdeplot, "selectivity", color="w", lw=2, cumulative=True)

            def cosmetics(x, **kwargs):
                high = np.percentile(x, 99)
                ax = plt.gca()
                ax.set_xlim((0, high))
                ax.spines['bottom'].set_linewidth(1)
                ax.tick_params(axis='both', length=3, width=1)

            g.map(cosmetics, "selectivity")
            g.fig.set_size_inches(sz)
            g.set_ylabels("cumulative distribution")
            g.fig.suptitle("Neuron Selectivities")
            g.fig.subplots_adjust(left=.1, right=.8, top=.85)
            g.despine(trim=True)

    return savefig(g.fig)


@images.route("/preferred_per_stack-<int:animal_id>_<size>.png")
def preferred_per_stack(animal_id, size):

    key = dict(animal_id=animal_id, **SETTINGS)
    df1 = pd.DataFrame((stack.StackSet.Match() & key).fetch())
    df1['session'] = df1['scan_session']
    df2 = pd.DataFrame((tune.Ori.Cell() & key & 'selectivity>=0.4 and r2>=0.004').fetch())
    df = df1.merge(df2)
    idx = df.groupby(['animal_id', 'stack_session', 'stack_idx', 'munit_id', 'ori_type', 'stimulus_type'])[
        'selectivity'].idxmax()
    df = df.iloc[idx]
    df['stack'] = ['{}-{}-{}'.format(ai, s, sa) for ai, s, sa in
                   zip(df.animal_id, df.stack_session, df.stack_idx)]
    df['stimulus_type'] = [s.split('.')[1] for s in df['stimulus_type']]

    sz = tuple(i * size_factor[size] for i in [1, .7])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1.3):
        with sns.axes_style(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}):
            # Initialize the FacetGrid object
            pal = ['steelblue', 'orange']
            g = sns.FacetGrid(df, row="stack", hue='stimulus_type', col='ori_type', palette=pal,
                              col_order=['ori', 'dir'], margin_titles=True, legend_out=True)
            #
            # # Draw the densities in a few steps
            g.map(sns.kdeplot, "angle", shade=False, alpha=.5, lw=3)
            g.add_legend(title="Stimulus Type", prop={'size': 8})

            def cosmetics(x, **kwargs):
                ax = plt.gca()
                ax.set_xlim((-np.pi, np.pi))
                ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
                ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
                ax.spines['bottom'].set_linewidth(1)
                ax.tick_params(axis='both', length=3, width=1)

            g.map(cosmetics, "angle")
            g.fig.set_size_inches(sz)
            g.set_ylabels("distribution")
            g.set_xlabels('preferred')
            g.fig.suptitle("preferred orientation/direction")
            g.fig.subplots_adjust(left=.1, right=.8, top=.85)
            g.despine(trim=True)

    return savefig(g.fig)


@images.route("/rf_snr_stack_stat-<int:animal_id>_<size>.png")
def rf_snr_stack_stat(animal_id, size):

    key = dict(animal_id=animal_id, **SETTINGS)
    df1 = pd.DataFrame((stack.StackSet.Match() & key).fetch())
    df1['session'] = df1['scan_session']

    df2 = pd.DataFrame((tune.STAQual() & key).fetch())
    df = df1.merge(df2).groupby(
        ['animal_id', 'stack_session', 'stack_idx',
         'munit_id', 'stimulus_type']).agg(dict(snr=np.max)).reset_index()
    df['stack'] = ['{}-{}-{}'.format(ai, s, sa) for ai, s, sa in
                   zip(df.animal_id, df.stack_session, df.stack_idx)]

    sz = tuple(i * size_factor[size] for i in [1, .7])
    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1.3):
        with sns.axes_style(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}):
            # Initialize the FacetGrid object
            pal = ['steelblue', 'orange', 'slategray']
            g = sns.FacetGrid(df, row="stack", hue='stimulus_type', palette=pal,
                              margin_titles=True, legend_out=False)
            #
            # # Draw the densities in a few steps
            g.map(sns.kdeplot, "snr", shade=False, alpha=1, lw=2, cumulative=True)
            g.add_legend(title="Stimulus Type")

            # g.map(sns.kdeplot, "snr",  color="w", lw=1,cumulative=True)


            def cosmetics(x, **kwargs):
                low = np.percentile(x, 2)
                high = np.percentile(x, 98)
                ax = plt.gca()
                ax.set_xlim((low, high))
                ax.spines['bottom'].set_linewidth(1)
                ax.tick_params(axis='both', length=3, width=1)
                ax.set_xlabel('receptive field SNR')
                ax.grid(True)

            g.map(cosmetics, "snr")
            g.fig.set_size_inches(sz)
            g.set_ylabels("cumulative distribution")
            g.fig.suptitle("receptive field SNR")
            g.fig.subplots_adjust(left=.1, right=.75)
            g.despine(trim=True)

    return savefig(g.fig)


@images.route("/cell_matches-<int:animal_id>_<size>.png")
def cell_matches(animal_id, size):

    key = dict(animal_id=animal_id, **SETTINGS)
    sz = tuple(i * size_factor[size] for i in [.7, .7])
    df = pd.DataFrame((stack.StackSet.Unit() * dj.U('stack_session') & key).aggr(stack.StackSet.Match() & key,
                                                                                 matches='COUNT(*)').fetch())

    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1.3):
        with sns.axes_style(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}):
            order = sorted(pd.unique(df.matches))

            g = sns.factorplot('matches', kind='count', hue='stack_session', data=df, order=order)
            sns.despine(trim=True, offset=5)

            g.ax.spines['bottom'].set_linewidth(1)
            g.ax.spines['left'].set_linewidth(1)
            g.ax.tick_params(axis='both', length=3, width=1)
            g.ax.set_xlabel('scans neuron was visible in')
            g.ax.set_ylabel('neurons')
    g.fig.set_size_inches(sz)

    return savefig(g.fig)


@images.route("/scan_hours-<int:animal_id>_<size>.png")
def scan_hours(animal_id, size):

    bins = np.arange(0, 22, 2)
    names = np.array(['{}-{}h'.format(*a) for a in zip(bins[:-1], bins[1:])] + ['>{}h'.format(bins[-1])])

    key = dict(animal_id=animal_id, **SETTINGS)
    sz = tuple(i * size_factor[size] for i in [.7, .7])

    reso_times = (reso.ScanInfo() & key).proj(scan_session='session', secs='nframes/fps')
    meso_times = (meso.ScanInfo() & key).proj(scan_session='session', secs='nframes/fps')
    df1 = pd.DataFrame((stack.StackSet.Unit() & key).aggr((stack.StackSet.Match() & key) * meso_times,
                                                          hours='sum(secs)/3600').fetch())
    df2 = pd.DataFrame((stack.StackSet.Unit() & key).aggr((stack.StackSet.Match() & key) * reso_times,
                                                          hours='sum(secs)/3600').fetch())

    df = pd.concat([df1, df2])
    df['hours'] = names[np.digitize(np.array(df['hours']).astype(np.float), bins=bins) - 1]

    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1.3):
        with sns.axes_style(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}):
            g = sns.factorplot('hours', kind='count', hue='stack_session', data=df, order=names)
            sns.despine(trim=True, offset=5)
            g.ax.spines['bottom'].set_linewidth(1)
            g.ax.spines['left'].set_linewidth(1)
            g.ax.tick_params(axis='both', length=3, width=1)
            g.ax.set_xlabel('scan hours')
            g.ax.set_ylabel('neurons')
    plt.setp(g.ax.get_xticklabels(), visible=True, rotation=-60)
    g.fig.subplots_adjust(bottom=.2)
    g.fig.set_size_inches(sz)

    return savefig(g.fig)

@images.route("/registration_over_time-<int:animal_id>-<int:session>-<int:scan_idx>_<size>.png")
def registration_over_time(animal_id, session, scan_idx, size):

    import matplotlib.ticker as ticker
    session_key = dict(animal_id=animal_id, scan_session=session, scan_idx=scan_idx, **SETTINGS)
    sz = tuple(i * size_factor[size] for i in [.7, .7])

    stack_sessions = (dj.U('stack_session') & (stack.StackSet() & session_key)).fetch("KEY")


    with sns.plotting_context('talk' if size == 'huge' else 'paper', font_scale=1.3):
        with sns.axes_style(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}):
            # Get all field keys and timestamps
            fig, ax = plt.subplots(len(stack_sessions), 1, figsize=sz)

    for skey in stack_sessions:
        key = dict(session_key, **skey)
        field_key, field_ts, field_nframes, field_fps = ((experiment.Scan() * meso.ScanInfo() *
                                        meso.ScanInfo.Field().proj()).proj('nframes', 'fps',
                                                                                      'scan_ts',
                                                                            scan_session='session')
                                         & key & 'field < 10').fetch('KEY', 'scan_ts',
                                                                             'nframes', 'fps',
                                                                                 order_by='scan_ts')
        assert len(field_key) > 0, 'Warning: No fields selected for'

        initial_time = str(field_ts[0])
        field_ts = [(ts - field_ts[0]).seconds for ts in field_ts]
        field_duration = field_nframes / field_fps

        for fk, ft, fd in zip(field_key, field_ts, field_duration):
            zs = (stack.RegistrationOverTime.Affine() & key & fk).fetch('reg_z', order_by='frame_num')
            ts = ft + np.linspace(0, 1, len(zs) + 2)[1:-1] * fd
            ax.plot(ts / 60, zs, 'o-', ms=4)
        ax.set_title('Registered zs for {animal_id}-{scan_session} starting {t}'.format(t=initial_time, **key))
        ax.set_ylabel('Registered zs')
        ax.set_xlabel('Time [min]')

        # Plot formatting
        ax.invert_yaxis()
        ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.grid(b=True, which='major', color='darkslategray', linestyle='--')
        ax.grid(b=True, which='minor', color='slategray', linestyle=':')
        sns.despine(bottom=True, left=True)
        fig.tight_layout()


    return savefig(fig)