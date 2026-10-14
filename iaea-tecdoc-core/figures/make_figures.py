"""
figures/make_figures.py
-----------------------
Reference figures for the IAEA-TECDOC-643 Appendix A-2 10 MW LEU core
manuscript (Elsevier CAS, single column, a4paper).

    fig1_core_map.pdf    position designations only -- no material fills
    fig2_core_xy.pdf     OpenMC render, XY slice, full core at f = 0
    fig3_axial_xz.pdf    OpenMC renders, axial elevation: XZ through the
                         absorber slot | YZ on the D4 flux-trap axis
    fig4_elements.pdf    OpenMC renders, SFE | CFE | flux trap at one scale

fig2-fig4 are native OpenMC geometry-plotter output -- flat fills, thin cell
boundaries, no text inside the frame -- the OpenMC equivalent of Wilson et al.,
ANL/RERTR/TM-10-41 Figs 2-8 to 2-13. The dimensioned engineering drawings are
produced in SolidWorks from figures/solidworks_equations.txt and
figures/solidworks_appearances.txt, both emitted here.

    python figures/make_figures.py
    python figures/make_figures.py --only fig4
    python figures/make_figures.py --equations-only

There is no Makefile, so this script IS the figures target: a plain run rebuilds
every figure AND regenerates both SolidWorks files. FIGURE_PROOFS=1 additionally
writes 600/150 dpi rasters and greyscale proofs to figures/_proof/.

=============================================================================
WHY EVERY FIGURE IS A PDF
=============================================================================
Geometry.plot() returns a matplotlib figure in which the render is an imshow
array and the legend is vector text. Saved as PDF the array is embedded as
raster at exactly the requested pixel count AND the legend stays sharp vector;
saved as PNG the vector text is rasterised for no benefit.

interpolation='none' -- NOT 'nearest'. In a vector backend they differ: 'none'
embeds the array at native resolution and lets the viewer scale it, while
'nearest' makes matplotlib resample to the output raster first, blurring the
cell boundaries that make a flat-fill render read as a drawing.

=============================================================================
RESOLUTION
=============================================================================
Pixel colour is sampled at pixel CENTRES, so a region thinner than the pixel
pitch can vanish. Every render satisfies

    pixels >= max( 2 * width_cm / CLAD_THICK_INNER , 600 * printed_width_in )

computed per figure from geometry.py constants and TEXTWIDTH_IN, and asserted
against the EMBEDDED IMAGE dimensions in the emitted PDF -- not against the page
size, which is independent and would pass while the render was downsampled.

=============================================================================
OUTLINE
=============================================================================
plot(outline=True) draws a matplotlib CONTOUR, whose width is in points, not
pixels. At 600 dpi the default is ~8 px wide and puts 36 % black ink over the
core, destroying the plate lamination. Measured contour ink and continuity
(contour isolated by differencing against outline=False):

    full core   0.08 pt   10.9 % ink, 0.59 continuous  -> OFF
    axial slot  0.08 pt    0.9 % ink, 0.73 continuous  -> ON
    element     0.08 pt    7.3 % ink, 0.85 continuous  -> ON

fig2 is off because its fills already separate by >= 0.15, so every boundary is
a hard edge, and the contour would add 11-23 % ink while rendering only ~60 %
continuous. The other two are cheap. These are per-figure decisions, measured.

=============================================================================
DIMENSION PROVENANCE
=============================================================================
Every dimension comes from model/geometry.py or is derived with the same
arithmetic geometry.py uses. Three quantities are not named constants there and
are derived, not invented: the lattice extent (read off the built RectLattice),
the active-region bbox (from CORE_MAP tokens), and the lateral unfueled cladding
margin, (ACTIVE_STACK_X - MEAT_WIDTH) / 2.

geometry.py's control-element comment still says "Al inner guide 0.150"; the
constant has been 0.127 since 2026-07-20 and the constant is what is used. Cell
naming is inverted against that comment (inner plate is `slider`, outer is
`guide`); the rename is deliberately not done here.

All dimensions in cm.
"""

import argparse
import datetime
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'model'))
sys.path.insert(0, THIS_DIR)

import openmc
from geometry import (
    PITCH_X, PITCH_Y, ELEM_X, ELEM_Y, ELEM_Z, GAP_X, GAP_Y,
    SIDE_PLATE_THICK, ACTIVE_STACK_X, LATERAL_CLAD_MARGIN,
    N_PLATES_STD, PLATE_THICK_INNER, PLATE_THICK_OUTER,
    CLAD_THICK_INNER, CLAD_THICK_OUTER, MEAT_THICK, MEAT_WIDTH,
    WATER_CHAN_THICK, STD_STACK_HEIGHT, STD_END_WATER,
    N_CTRL_FUEL_PLATES, CTRL_PLATE_PITCH, CTRL_FUEL_STACK_HALF,
    CTRL_FUEL_WIDTH_X, CTRL_SIDE_PLATE_X, CTRL_AL_PLATE_THICK,
    CTRL_FEEDER_CHANNEL, CTRL_BLADE_WATER, CTRL_OUTER_OFFSET,
    CTRL_END_BLOCK, ABSORBER_THICK, ABSORBER_WIDTH, BLADE_TOP_CLAD,
    HALF_Z, HALF_PLATE_Z, PLATE_HEIGHT, MEAT_HEIGHT, CLAD_EXT, ENDBOX_HEIGHT,
    CORE_TOP, CORE_BOTTOM, ENDBOX_ABOVE_TOP, ENDBOX_BELOW_BOT,
    BLADE_LENGTH, ROD_TRAVEL,
    FT_HOLE_RADIUS, CORE_MAP, CORE_MAP_COLS, core_map_label,
    build_core_geometry,
)
import figstyle as S
from figstyle import (
    REGION_FILL, LUM_ORDER, LW_ENVELOPE, LW_PART, LW_DIM, INK,
    FS_LABEL, FS_DIM, FS_PANEL, FS_TICK, textwidth_in, FIG1_WIDTH_FRAC,
    in_to_pdf_pt, openmc_colors, display_name, rgb255,
)

OUT_DIR = THIS_DIR
PAD_INCHES = 0.02

# LATERAL_CLAD_MARGIN was defined here and asserted > 0 in
# check_geometry_consistency(). It is a geometry constant, not a figure
# quantity, so it now lives in geometry.py (imported above) with a
# MIN_RESIDUAL_GAP floor — which runs on every import of the model, not only
# when somebody rebuilds the figures. Same arithmetic, same value.
SIDE_INNER = ACTIVE_STACK_X / 2.0

# fig3 is cut through the ABSORBER SLOT, not y = 0: an axial figure of a control
# element that misses the blade is the wrong slice. Derived outward from the
# follower stack exactly as geometry.py builds the end block.
SLOT_Y = (CTRL_FUEL_STACK_HALF + CTRL_FEEDER_CHANNEL + CTRL_AL_PLATE_THICK
          + CTRL_BLADE_WATER + ABSORBER_THICK / 2.0)

CTRL_END_LAYERS = [
    ('feeder chan.',    CTRL_FEEDER_CHANNEL, 'coolant',       'chan_*_half'),
    ('Al guide, inner', CTRL_AL_PLATE_THICK, 'structural_Al', 'slider_*'),
    ('blade water',     CTRL_BLADE_WATER,    'coolant',       'blade_water_inner_*'),
    ('B4C slot',        ABSORBER_THICK,      'B4C',           'absorber_*'),
    ('blade water',     CTRL_BLADE_WATER,    'coolant',       'blade_water_outer_*'),
    ('Al guide, outer', CTRL_AL_PLATE_THICK, 'structural_Al', 'guide_*'),
    ('offset water',    CTRL_OUTER_OFFSET,   'coolant',       'offset_water_*'),
]

OUTLINE_LW = 0.08          # measured; see the header table

# fig3 z crop. The full model is 180 cm against a 61.6 cm width -- an aspect of
# 2.92, which at any usable width is taller than the a4 text block. Cropping to
# just past the end boxes keeps the boundary condition visible without 45 cm of
# pool. Derived from the end-box height so it tracks the constants.
#
# ENDBOX_HEIGHT is IMPORTED, not recomputed. It used to be derived here as
# ENDBOX_ABOVE_TOP - HALF_Z, which was right only while the plates and the meat
# were the same height. Since B1 the end box runs from HALF_PLATE_Z (+31), so
# that expression yields 15 where the real end box is 14.
FIG3_POOL_MARGIN = 0.4 * ENDBOX_HEIGHT          # 5.6 cm
FIG3_Z_HALF = ENDBOX_ABOVE_TOP + FIG3_POOL_MARGIN

# fig3 panel (b): the y-z elevation is cut through the CENTRAL flux trap, named
# by its CORE_MAP position rather than by coordinate. The other trap is A6, at
# the core edge; D4 is the one the manuscript calls the central trap.
FIG3_TRAP = 'D4'

# ---------------------------------------------------------------------------
# fig3 panel (b) crop -- WHY IT IS NOT PANEL (a)'s CROP
# ---------------------------------------------------------------------------
# Panel (a) is fixed, so the shared printed height is fixed, so at equal aspect
# (b)'s SCALE is height / z-span. Narrowing (b) in Y cannot change that: it
# crops, it does not magnify. At (a)'s 101.2 cm span the plate body -- the light
# clad+meat+clad bar between two coolant channels -- prints 113 um, under the
# ~170 um halftone cell, so the plate stack renders as a dark field with a
# sub-resolution texture rather than as lamination. Only a shorter z span fixes
# it, because only that raises the scale.
#
# 62.0 cm = 2 * HALF_PLATE_Z, the full fuel plate height: a real geometric
# boundary, not a chosen round number. It gives 1:6.90, and:
#     plate pitch  501 um     plate body  184 um  (clears the cell)
#     meat          74 um     -- does NOT resolve, by design
# So each plate reads as one light bar and the clad/meat/clad substructure
# inside it stays merged. That is the intent: lamination reads as lamination,
# not every detail.
#
# 24.3 cm = 3 * PITCH_Y, three whole lattice rows centred on the trap: a full
# standard element, the trap, a full standard element. Whole rows, so no element
# is sliced in half at the frame edge.
FIG3_B_Z_HALF = HALF_PLATE_Z        # 31.0 cm
FIG3_B_ROWS   = 3                   # D3, D4, D5

# EMITTED width. The layout arithmetic still runs on FIG1_WIDTH_FRAC, because
# that is what fixes panel (a)'s printed size and panel (a) must not move; but
# the narrow (b) means the two panels no longer fill that measure, so the figure
# is emitted at its own width instead of carrying 1.4 cm of dead margin. Held as
# a declared fraction rather than measured at run time so that check_figures has
# an independent number to test the emitted MediaBox against -- a target read
# back out of the builder would only ever agree with itself.
# Content measures 8.07 cm -- the panel row, which is now the widest element
# (the legend is 7.45 cm). 0.50 leaves ~0.16 cm of slack, and save() asserts if
# that is ever exceeded. Was 0.53 while the sub-captions overhung their panels.
FIG3_WIDTH_FRAC = 0.50

# CONSEQUENCE, RECORDED BECAUSE IT IS EASY TO MISREAD AS A FIX. This crop drops
# the graphite reflector rows (column D is G,S,S,F,S,S,G) and, with the z crop,
# the end box and pool water as well. The coolant/graphite pair that fig3 used
# to carry is therefore OUT OF FRAME BY GEOMETRY -- the crop resolves nothing.
# What resolved it is the 2026-08-13 palette rebuild, which moved the pair from
# 0.0455 to 0.3076 everywhere it occurs, fig2 included. Two separate facts.

# fig4 split panels: each stands at 0.48 x textwidth so two sit side by side in
# a row of minipages, or one at half measure.
FIG4_SPLIT_FRAC = 0.48

# fig2 compaction. Cropping to a quarter-pitch margin outside the core keeps the
# boundary condition visible -- it is a legend entry -- without the dead space.
# Derived from the pitch so it tracks the constants.
#
# NOTE: this text described the old 8x9 lattice and its one-cell water ring.
# Since B4 the lattice is the 6x7 core only and the water is an explicit 38.5 cm
# pool box, so the figure code below needs a wider review than this comment fix
# -- see the handover note. The crop constant itself is still pitch-derived.
#
# Note the crop alone does NOT shorten the figure: the ring is proportionally
# wider than tall, so removing it raises the aspect ratio from 1.183 to 1.210
# and at fixed width that is MORE height, not less. The height comes off by
# narrowing the figure to 0.8 x textwidth, which also keeps it off its own page.
FIG2_POOL_MARGIN = 0.25 * PITCH_X               # 1.925 cm
FIG2_WIDTH_FRAC = 0.8
PAD_TIGHT = 0.012                               # axes side padding, fraction
LEG_ROW_TIGHT = 0.140                           # legend row height, in
LEG_GAP_TIGHT = 0.05                            # plot-to-legend gap, in


def check_geometry_consistency():
    assert abs(SIDE_INNER - (ELEM_X / 2 - SIDE_PLATE_THICK)) < 1e-12
    assert abs((2 * CLAD_THICK_OUTER + MEAT_THICK) - PLATE_THICK_OUTER) < 1e-12
    assert abs((2 * CLAD_THICK_INNER + MEAT_THICK) - PLATE_THICK_INNER) < 1e-12
    std = (2 * PLATE_THICK_OUTER + (N_PLATES_STD - 2) * PLATE_THICK_INNER
           + (N_PLATES_STD - 1) * WATER_CHAN_THICK)
    assert abs(std - STD_STACK_HEIGHT) < 1e-12
    assert abs(STD_STACK_HEIGHT + 2 * STD_END_WATER - ELEM_Y) < 1e-12
    # `assert LATERAL_CLAD_MARGIN > 0` removed 2026-08-24: strictly implied by
    # the LATERAL_CLAD_MARGIN >= MIN_RESIDUAL_GAP floor in geometry.py, which
    # has already run at import time by the time this function is called.
    ctrl = (N_CTRL_FUEL_PLATES * PLATE_THICK_INNER
            + (N_CTRL_FUEL_PLATES - 1) * WATER_CHAN_THICK)
    assert abs(ctrl / 2 - CTRL_FUEL_STACK_HALF) < 1e-12
    assert abs(sum(t for _l, t, _k, _c in CTRL_END_LAYERS) - CTRL_END_BLOCK) < 1e-12
    assert abs(CTRL_FUEL_STACK_HALF + CTRL_END_BLOCK - ELEM_Y / 2) < 1e-12
    assert abs(GAP_X - (PITCH_X - ELEM_X) / 2) < 1e-12
    assert abs(GAP_Y - (PITCH_Y - ELEM_Y) / 2) < 1e-12
    # ELEM_Z is the PLATE height (62) and 2*HALF_Z the MEAT height (60); since
    # B1 they differ by 2*CLAD_EXT. Assert the relationship that now holds.
    assert abs(ELEM_Z - PLATE_HEIGHT) < 1e-12
    assert abs(2 * HALF_Z - MEAT_HEIGHT) < 1e-12
    assert abs(ELEM_Z - (2 * HALF_Z + 2 * CLAD_EXT)) < 1e-12
    assert abs(HALF_PLATE_Z - (HALF_Z + CLAD_EXT)) < 1e-12
    assert abs((ENDBOX_ABOVE_TOP - HALF_PLATE_Z) - ENDBOX_HEIGHT) < 1e-12
    assert 2 * FT_HOLE_RADIUS < min(PITCH_X, PITCH_Y)
    # the slot cut must land inside the absorber band, not beside it
    lo = CTRL_FUEL_STACK_HALF + CTRL_FEEDER_CHANNEL + CTRL_AL_PLATE_THICK + CTRL_BLADE_WATER
    assert lo < SLOT_Y < lo + ABSORBER_THICK, \
        f'fig3 slot cut y={SLOT_Y} is outside the absorber band'
    assert ENDBOX_ABOVE_TOP < FIG3_Z_HALF < CORE_TOP, \
        'fig3 crop must clear the end box but stay inside the model'
    # fig3 panel (b) must cut a flux trap, and cut it on the AXIS -- that is the
    # only x giving the full 2r chord. Checked against CORE_MAP so that moving
    # or renaming the trap fails here rather than silently shifting the cut.
    _trap = [core_map_label(i, j) for i, r in enumerate(CORE_MAP)
             for j, t in enumerate(r) if t == 'F']
    assert FIG3_TRAP in _trap, \
        f'fig3 panel (b): {FIG3_TRAP} is not a flux trap position (traps: {_trap})'
    S.check_palette()          # hard errors only: unparseable or duplicate colour
    S.check_type_sizes()
    print(f'  [OK] tripwires; slot cut y = {SLOT_Y:.4f} cm')
    # Palette quality is REPORTED, never enforced here -- edit REGION_FILL
    # freely and the build still runs. check_figures.py is where a lost
    # separation counts as a failure.
    for note in S.palette_issues():
        print(f'  [palette] {note}')


def apply_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': FS_DIM,
        'axes.labelsize': FS_LABEL, 'legend.fontsize': FS_DIM,
        'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
        'axes.linewidth': LW_DIM, 'xtick.major.width': LW_DIM,
        'ytick.major.width': LW_DIM, 'xtick.major.size': 2.0,
        'ytick.major.size': 2.0, 'lines.linewidth': LW_DIM,
        'patch.linewidth': LW_PART, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        # one image object per figure; compositing would merge or retile the
        # embedded render and break the pixel-count assertion
        'image.composite_image': False,
    })


def required_pixels(width_cm, printed_in):
    """max(feature-limited, print-limited). Returns (feature, print, required)."""
    t1 = 2.0 * width_cm / CLAD_THICK_INNER
    t2 = 600.0 * printed_in
    return t1, t2, int(np.ceil(max(t1, t2)))


# =============================================================================
# RENDER + SAVE
# =============================================================================

RENDERED = {}          # figure stem -> set of region kinds actually rendered


def _kinds_in(im):
    """Region kinds present in a rendered array, read from its own pixels.

    Ground truth for what a figure contains. A probe grid over geometry.find()
    is sampling-dependent -- a coarse grid silently misses thin regions, which
    under-reports the legend AND the separation matrix. The render is the thing
    being tested, so the render is what gets measured.
    """
    a = np.asarray(im.get_array())[..., :3]
    if a.dtype.kind == 'f':
        a = np.clip(np.round(a * 255), 0, 255).astype(np.uint8)
    key = ((a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8)
           | a[..., 2].astype(np.uint32))
    seen = set(np.unique(key).tolist())
    out = set()
    for k in REGION_FILL:
        r, g, b = rgb255(k)
        if ((r << 16) | (g << 8) | b) in seen:
            out.add(k)
    return out


def render(ax, geom, *, basis, origin, width, pixels, outline, frame=False):
    """One OpenMC render into `ax`. No text, no ticks -- Wilson's convention.

    `frame` keeps the axes spines as a thin black boundary. fig2 and fig3 need
    it: pool water is #FBFDFD, effectively white, and it is the entire
    background of fig3, so without a frame the figure has no edge against the
    page. Chroma cannot fix that -- at L = 0.990 the achievable C* is about 0.7.
    The frame is drawn at OUTLINE_LW so it matches fig4's cell boundaries and
    the set reads as one system.
    """
    kw = dict(basis=basis, origin=origin, width=width, pixels=pixels, axes=ax,
              color_by='material', colors=openmc_colors(), legend=False)
    if outline:
        kw.update(outline=True,
                  contour_kwargs={'linewidths': OUTLINE_LW, 'colors': 'k'})
    geom.plot(**kw)
    ax.set_title('')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(''); ax.set_ylabel('')
    for sp in ax.spines.values():
        sp.set_visible(frame)
        if frame:
            sp.set_linewidth(OUTLINE_LW)
            sp.set_color(INK)
    if not frame:
        ax.set_axis_off()
    for im in ax.images:
        # 'none' embeds the array at native resolution in a vector backend;
        # 'nearest' would resample to the output raster and blur the boundaries
        im.set_interpolation('none')
    return _kinds_in(ax.images[-1])


def material_legend(fig, kinds, *, y_in, ncol):
    """Legend built by hand from figstyle.DISPLAY_NAMES.

    plot(legend=True) would print materials.py's raw identifiers, and those are
    production identifiers that must not be renamed for legend cosmetics.
    """
    handles = [mpatches.Patch(facecolor=REGION_FILL[k], edgecolor=INK,
                              linewidth=LW_DIM, label=display_name(k))
               for k in LUM_ORDER if k in kinds]
    return fig.legend(handles=handles, loc='lower center',
                      bbox_to_anchor=(0.5, y_in / fig.get_figheight()),
                      ncol=ncol, frameon=False, fontsize=FS_DIM,
                      handlelength=1.6, handleheight=1.0, columnspacing=1.1,
                      handletextpad=0.45, borderaxespad=0.0, labelspacing=0.34)


def panel_letter(fig, ax, text, *, dy_in=0.05):
    bb = ax.get_position()
    fig.text(bb.x0, bb.y1 + dy_in / fig.get_figheight(), text,
             fontsize=FS_PANEL, fontweight='bold', ha='left', va='bottom',
             color=INK)


def caption(fig, ax, text, *, dy_in=0.05):
    bb = ax.get_position()
    fig.text(bb.x0 + bb.width / 2, bb.y0 - dy_in / fig.get_figheight(), text,
             fontsize=FS_LABEL, ha='center', va='top', color=INK)


def save(fig, stem, *, target_in, expect_px=None, axes_in=None):
    """Emit a PDF exactly target_in wide, then verify what actually landed."""
    from matplotlib.transforms import Bbox
    path = os.path.join(OUT_DIR, f'{stem}.pdf')
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    assert bb.width <= target_in + 1e-6, (
        f'{stem}: content {bb.width:.4f} in exceeds target {target_in:.4f} in')
    cx = 0.5 * (bb.x0 + bb.x1)
    save_bb = Bbox.from_extents(cx - target_in / 2, bb.y0 - PAD_INCHES,
                                cx + target_in / 2, bb.y1 + PAD_INCHES)
    fig.savefig(path, bbox_inches=save_bb)
    if os.environ.get('FIGURE_PROOFS'):
        d = os.path.join(OUT_DIR, '_proof')
        os.makedirs(d, exist_ok=True)
        for dpi in (600, 150):
            p = os.path.join(d, f'{stem}_{dpi}dpi.png')
            fig.savefig(p, dpi=dpi, bbox_inches=save_bb)
            from PIL import Image
            im = Image.open(p).convert('L')       # greyscale proof
            im.save(os.path.join(d, f'{stem}_{dpi}dpi_grey.png'))
    plt.close(fig)
    msg = f'  Saved: {path}  ({os.path.getsize(path)/1024:.0f} KB)'
    if expect_px:
        msg += f'   render {expect_px[0]}x{expect_px[1]} px'
    if axes_in and expect_px:
        # Effective dpi is pixels over the printed AXES width, not the page
        # width -- the axes sit inside the page by the side padding, so a
        # page-width figure would understate the requirement.
        dpi_eff = expect_px[0] / axes_in
        msg += f'   axes {axes_in:.4f} in = {dpi_eff:.1f} eff dpi'
        assert dpi_eff >= 599.5, (
            f'{stem}: {dpi_eff:.1f} dpi over a {axes_in:.4f} in axes is under '
            f'600; raise the pixel count')
    print(msg)
    return path


# =============================================================================
# FIGURE 1 -- position designations only, no material fills
# =============================================================================

POS = {
    'S': dict(glyph='S',  lw=LW_PART,     box=False, label='standard fuel element'),
    'C': dict(glyph='C',  lw=LW_ENVELOPE, box=True,  label='control fuel element'),
    'G': dict(glyph='G',  lw=LW_PART,     box=False, label='graphite reflector'),
    'F': dict(glyph='FT', lw=LW_PART,     box=False, label='flux trap'),
}


def core_lattice_of(geom):
    for cell in geom.root_universe.cells.values():
        if isinstance(cell.fill, openmc.RectLattice):
            return cell.fill
    raise RuntimeError('no RectLattice in the built geometry')


def active_region_bbox():
    p = [(i, j) for i, r in enumerate(CORE_MAP)
         for j, t in enumerate(r) if t in ('S', 'C', 'F')]
    return (min(x[0] for x in p), max(x[0] for x in p),
            min(x[1] for x in p), max(x[1] for x in p))


def fig1_core_map():
    """Wilson Fig 2-7 role: which position is what, and nothing else.

    No material fills at all, so this figure contributes no regions to the
    palette's co-occurrence constraints and needs no material legend -- its
    legend entries are position designations, which have no material behind them.
    """
    geom = build_core_geometry(withdrawn_fraction=0.0)
    lat = core_lattice_of(geom)
    nx, ny = lat.shape
    px, py = lat.pitch
    llx, lly = lat.lower_left
    urx, ury = llx + nx * px, lly + ny * py
    col_x = lambda j: llx + (j + 0.5) * px
    row_y = lambda i: ury - (i + 0.5) * py
    r0, r1, c0, c1 = active_region_bbox()

    W = FIG1_WIDTH_FRAC * textwidth_in()
    y_leg = 0.0
    pad_b = y_leg + 3 * 0.150 + 0.10 + 0.42
    pl, pr, pt = 0.135, 0.055, 0.03
    aw = W * (1 - pl - pr)
    ah = aw * (ury - lly) / (urx - llx)
    fig_h = ah + pad_b + pt
    fig = plt.figure(figsize=(W, fig_h))
    ax = fig.add_axes([pl, pad_b / fig_h, 1 - pl - pr, ah / fig_h])
    ax.set_xlim(llx, urx); ax.set_ylim(lly, ury); ax.set_aspect('equal')
    ax.set_xlabel('$x$ (cm)'); ax.set_ylabel('$y$ (cm)')

    for j in range(nx + 1):
        ax.plot([llx + j * px] * 2, [lly, ury], color=INK, lw=LW_DIM, alpha=0.5)
    for i in range(ny + 1):
        ax.plot([llx, urx], [lly + i * py] * 2, color=INK, lw=LW_DIM, alpha=0.5)
    for i, row in enumerate(CORE_MAP):
        for j, tok in enumerate(row):
            if tok == 'W':
                continue
            st = POS[tok]
            ax.add_patch(mpatches.Rectangle(
                (llx + j * px, ury - (i + 1) * py), px, py, facecolor='none',
                edgecolor=INK, linewidth=st['lw'], zorder=5))
            bbox = (dict(boxstyle='square,pad=0.22', facecolor='none',
                         edgecolor=INK, linewidth=LW_PART) if st['box'] else None)
            ax.text(col_x(j), row_y(i), st['glyph'], color=INK,
                    fontsize=FS_LABEL, ha='center', va='center', zorder=8,
                    bbox=bbox)
    ax.add_patch(mpatches.Rectangle(
        (llx + c0 * px, ury - (r1 + 1) * py), (c1 - c0 + 1) * px,
        (r1 - r0 + 1) * py, facecolor='none', edgecolor=INK, lw=LW_ENVELOPE,
        zorder=9))

    sx = ax.secondary_xaxis('top')
    sx.set_xticks([col_x(j) for j in range(c0, c1 + 1)])
    sx.set_xticklabels(list(CORE_MAP_COLS[:c1 - c0 + 1]))
    sx.tick_params(length=1.6, width=LW_DIM, pad=1.5)
    sx.set_xlabel('grid column', labelpad=2.0)
    rows = [i for i in range(ny) if core_map_label(i, c0) is not None]
    sy = ax.secondary_yaxis('right')
    sy.set_yticks([row_y(i) for i in rows])
    sy.set_yticklabels([core_map_label(i, c0)[1:] for i in rows])
    sy.tick_params(length=1.6, width=LW_DIM, pad=1.5)
    sy.set_ylabel('grid row', labelpad=3.0)
    ax.tick_params(length=1.6, width=LW_DIM, pad=1.5)

    counts = {t: sum(r.count(t) for r in CORE_MAP) for t in 'SCGF'}
    handles = [mpatches.Patch(facecolor='none', edgecolor=INK,
                              linewidth=POS[t]['lw'],
                              label=f"{POS[t]['glyph']} — {POS[t]['label']} "
                                    f"({counts[t]})") for t in 'SCGF']
    handles.append(mlines.Line2D([], [], color=INK, lw=LW_ENVELOPE,
                   label=f'active core, {r1-r0+1} $\\times$ {c1-c0+1}'))
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, y_leg / fig_h), ncol=2, frameon=False,
               fontsize=FS_DIM, handlelength=1.6, handleheight=1.0,
               columnspacing=1.1, handletextpad=0.45, borderaxespad=0.0,
               labelspacing=0.34)
    return save(fig, 'fig1_core_map', target_in=W)


# =============================================================================
# FIGURES 2-4 -- native OpenMC renders
# =============================================================================

def fig2_core_xy(f=0.0):
    """Full core, XY at the fuel midplane. Outline OFF -- measured."""
    geom = build_core_geometry(withdrawn_fraction=f)
    lat = core_lattice_of(geom)
    nx, ny = lat.shape
    px, py = lat.pitch
    llx, lly = lat.lower_left
    ury = lly + ny * py
    # Cropped to the labelled core (CORE_MAP cols 1-6, rows 1-7) plus a
    # quarter-pitch pool margin, derived rather than a literal extent.
    r0, r1, c0, c1 = active_region_bbox()
    ncol, nrow = c1 - c0 + 1, r1 - r0 + 1 + 2      # + the reflector rows
    m = FIG2_POOL_MARGIN
    W_cm = ncol * px + 2 * m
    H_cm = nrow * py + 2 * m
    cx = llx + (c0 + ncol / 2.0) * px
    cy = ury - (r0 - 1 + nrow / 2.0) * py

    W = FIG2_WIDTH_FRAC * textwidth_in()
    y_leg = 0.0
    pad_b = y_leg + 4 * LEG_ROW_TIGHT + LEG_GAP_TIGHT
    pl = pr = PAD_TIGHT
    aw = W * (1 - pl - pr)
    ah = aw * H_cm / W_cm
    fig_h = ah + pad_b + 0.03
    t1, t2, need = required_pixels(W_cm, aw)
    npx = (need, int(round(need * H_cm / W_cm)))
    print(f'  fig2 {W_cm:.2f}x{H_cm:.2f} cm aspect {H_cm/W_cm:.4f}  '
          f'{W:.3f}x{fig_h:.3f} in  pixels feature {t1:.0f}, print {t2:.0f} '
          f'-> {need} (clad {CLAD_THICK_INNER*need/W_cm:.2f} px)')

    fig = plt.figure(figsize=(W, fig_h))
    ax = fig.add_axes([pl, pad_b / fig_h, 1 - pl - pr, ah / fig_h])
    kinds = render(ax, geom, basis='xy', origin=(cx, cy, 0.0),
                   width=(W_cm, H_cm), pixels=npx, outline=False, frame=True)
    RENDERED['fig2_core_xy'] = kinds
    material_legend(fig, kinds, y_in=y_leg, ncol=2)
    return save(fig, 'fig2_core_xy', target_in=W, expect_px=npx, axes_in=aw)


def flux_trap_centre(lat, label):
    """(x, y) of the flux trap at CORE_MAP position `label`.

    Keyed on the LABEL, not on a row-major index. There are two traps -- D4 and
    A6 -- and the y-z cut has to pass through the axis of a named one, so an
    index would silently pick the other if CORE_MAP were ever reordered.
    """
    nx, ny = lat.shape
    px, py = lat.pitch
    llx, lly = lat.lower_left
    ury = lly + ny * py
    for i, row in enumerate(CORE_MAP):
        for j, tok in enumerate(row):
            if tok == 'F' and core_map_label(i, j) == label:
                return llx + (j + 0.5) * px, ury - (i + 0.5) * py
    raise RuntimeError(f'no flux trap at CORE_MAP position {label}')


def fig3_axial_xz(f=0.0):
    """Axial elevation, two panels. Outline ON.

    (a) XZ through the ABSORBER SLOT so the blade is in the figure.
    (b) YZ through the FIG3_TRAP flux trap, cut on its cylinder AXIS so the
        hole shows its full 2 * FT_HOLE_RADIUS diameter -- the widest chord.
        A y-z plane anywhere else through the trap shows a shorter chord and
        understates it.

    PANEL (a) IS UNCHANGED IN EVERY RESPECT THAT AFFECTS ITS CONTENT: same crop
    (FIG3_Z_HALF), same window (nx * px), same cut (SLOT_Y), same palette, same
    outline/frame flags, same pixel count. Only its PLACED SIZE moved, because a
    second panel now shares the measure. The pixel count is genuinely identical
    rather than coincidentally so: required_pixels() takes the max of a
    feature-limited term and a print-limited one, and at both the old and new
    printed widths the feature term (2 * 46.2 / 0.038 = 2432) dominates.

    The panels are sized to a COMMON HEIGHT at equal aspect, so (b) comes out
    wider than (a): the lattice is 7 rows deep in y and 6 columns across in x,
    so the y window is 56.7 cm against 46.2. Neither window is cropped or
    widened -- each is its axis's full lattice extent.

    KNOWN, DELIBERATE, NOT FIXED HERE: panel (b) is the first thing in fig3 to
    contain graphite (column D is G,S,S,F,S,S,G), so the shared key now carries
    all EIGHT regions and coolant/graphite join the pairs sitting under
    figstyle.VALUE_MIN -- four now, three of which fig3 already had. This is not
    fixable by recolouring: eight levels 0.15 apart need 1.05 of luma span and
    only 1.0 exists, so any real fix merges two regions onto one level, which is
    out of scope. Narrowing (b)'s window to miss the reflector rows would hide
    it rather than fix it, and is deliberately not done.
    """
    geom = build_core_geometry(withdrawn_fraction=f)
    lat = core_lattice_of(geom)
    nx, ny = lat.shape
    px, py = lat.pitch
    llx, lly = lat.lower_left
    W_cm = nx * px                              # 46.2 -- panel (a), unchanged
    Za_cm = 2 * FIG3_Z_HALF                     # 101.2, see FIG3_POOL_MARGIN
    Y_cm = FIG3_B_ROWS * py                     # 24.3 -- panel (b), see above
    Zb_cm = 2 * FIG3_B_Z_HALF                   # 62.0
    cx = llx + W_cm / 2
    tx, ty = flux_trap_centre(lat, FIG3_TRAP)

    # fig4_elements' margins, so the two multi-panel figures read as one system.
    # LAYOUT runs on FIG1_WIDTH_FRAC and nothing else, because that is what pins
    # panel (a)'s printed width; the figure's own width falls out below.
    W_layout = FIG1_WIDTH_FRAC * textwidth_in()
    pl = pr = 0.015 * W_layout
    gap = 0.045 * W_layout
    # Panel (a) is FIXED, and it is what sets the shared height: its printed
    # width is what it was, so ph follows from its own aspect. (b) is then sized
    # to that same height at ITS aspect, which is why the two scales differ.
    pw_a = (W_layout - pl - pr - gap) * W_cm / (W_cm + ny * py)
    ph = pw_a * Za_cm / W_cm
    pw = {'a': pw_a, 'b': ph * Y_cm / Zb_cm}
    # The figure is exactly as wide as its content row. Sizing the canvas to
    # W_layout instead would leave the panels sitting left of centre while the
    # legend, anchored at 0.5 of the FIGURE, centred somewhere else entirely.
    W = pl + pw['a'] + gap + pw['b'] + pr

    # NO SUB-CAPTIONS IN THE FIGURE. They carried each panel's z extent, which
    # is what told a reader that the two panels are at DIFFERENT SCALES --
    # (a) 1:11.26 over the full model height, (b) 1:6.90 over the active region.
    # With them gone that information has to live in the manuscript caption
    # instead; the in-figure cue is only that (b) visibly lacks the end boxes
    # and pool water (a) opens and closes on. Panel letters stay, per
    # fig4_elements' layout.
    panels = [
        ('a', 'xz', (cx, SLOT_Y, 0.0), W_cm, Za_cm),
        ('b', 'yz', (tx, ty, 0.0), Y_cm, Zb_cm),
    ]
    y_leg = 0.0
    pad_b = y_leg + 4 * 0.150 + 0.10
    fig_h = pad_b + ph + 0.22

    fig = plt.figure(figsize=(W, fig_h))
    allk = set()
    x_in = pl
    for key, basis, origin, w_cm, z_cm in panels:
        t1, t2, need = required_pixels(w_cm, pw[key])
        npx = (need, int(round(need * z_cm / w_cm)))
        dpi_eff = npx[0] / pw[key]
        assert dpi_eff >= 599.5, (
            f'fig3 ({key}): {dpi_eff:.1f} dpi over {pw[key]:.4f} in is under 600')
        ax = fig.add_axes([x_in / W, pad_b / fig_h, pw[key] / W, ph / fig_h])
        allk |= render(ax, geom, basis=basis, origin=origin,
                       width=(w_cm, z_cm), pixels=npx, outline=True, frame=True)
        panel_letter(fig, ax, '(%s)' % key)
        print(f'  fig3 ({key}) {basis} {w_cm:.1f}x{z_cm:.1f} cm  '
              f'{pw[key]*2.54:.2f}x{ph*2.54:.2f} cm  1:{z_cm/(ph*2.54):.2f}  '
              f'pixels feature {t1:.0f}, print {t2:.0f} -> {need} '
              f'({npx[0]}x{npx[1]}, {dpi_eff:.0f} dpi)')
        x_in += pw[key] + gap
    RENDERED['fig3_axial_xz'] = allk
    material_legend(fig, allk, y_in=y_leg, ncol=2)
    return save(fig, 'fig3_axial_xz',
                target_in=FIG3_WIDTH_FRAC * textwidth_in())


def _cell_centre(tok, want_index=0):
    """Global (x, y) of the want_index-th CORE_MAP cell carrying `tok`."""
    geom = build_core_geometry(withdrawn_fraction=0.0)
    lat = core_lattice_of(geom)
    nx, ny = lat.shape
    px, py = lat.pitch
    llx, lly = lat.lower_left
    ury = lly + ny * py
    hits = [(i, j) for i, r in enumerate(CORE_MAP)
            for j, t in enumerate(r) if t == tok]
    i, j = hits[want_index]
    return llx + (j + 0.5) * px, ury - (i + 0.5) * py


PANEL_MATS = {'S': {'coolant', 'clad', 'meat', 'structural_Al'},
              'C': {'coolant', 'clad', 'meat', 'structural_Al', 'B4C'},
              'F': {'coolant', 'structural_Al'}}
PANEL_CAP = {'S': 'standard element', 'C': 'control element', 'F': 'flux trap'}


def fig4_split(f=0.0):
    """One PDF per element type, each at FIG4_SPLIT_FRAC x textwidth.

    Each carries only the materials present in ITS panel -- the flux trap legend
    is two entries, not five -- taken from the same derived co-occurrence the
    acceptance test uses, not hand-declared.
    """
    geom = build_core_geometry(withdrawn_fraction=f)
    lat = core_lattice_of(geom)
    px, py = lat.pitch
    W = FIG4_SPLIT_FRAC * textwidth_in()
    pl = pr = 0.02
    aw = W * (1 - pl - pr)
    ah = aw * py / px
    out = []
    for tok, stem in (('S', 'fig4a_sfe'), ('C', 'fig4b_cfe'), ('F', 'fig4c_ft')):
        nrows = (len(PANEL_MATS[tok]) + 1) // 2   # verified against the render below
        y_leg = 0.0
        pad_b = y_leg + nrows * 0.150 + 0.10 + 0.20
        fig_h = pad_b + ah + 0.06
        t1, t2, need = required_pixels(px, aw)
        npx = (need, int(round(need * py / px)))
        fig = plt.figure(figsize=(W, fig_h))
        ax = fig.add_axes([pl, pad_b / fig_h, 1 - pl - pr, ah / fig_h])
        ox, oy = _cell_centre(tok)
        kinds = render(ax, geom, basis='xy', origin=(ox, oy, 0.0),
                       width=(px, py), pixels=npx, outline=True)
        RENDERED[stem] = kinds
        assert kinds == PANEL_MATS[tok], (
            f'{stem}: rendered {sorted(kinds)} != expected {sorted(PANEL_MATS[tok])}')
        caption(fig, ax, PANEL_CAP[tok])
        material_legend(fig, kinds, y_in=y_leg, ncol=2)
        print(f'  {stem}: pixels feature {t1:.0f}, print {t2:.0f} -> {need} '
              f'(clad {CLAD_THICK_INNER*need/px:.2f} px), '
              f'{len(kinds)} legend entries: {sorted(kinds)}')
        out.append(save(fig, stem, target_in=W, expect_px=npx, axes_in=aw))
    return out


def fig4_elements(f=0.0):
    """SFE | CFE | flux trap, one scale, one lattice cell each. Outline ON."""
    geom = build_core_geometry(withdrawn_fraction=f)
    lat = core_lattice_of(geom)
    px, py = lat.pitch
    W = textwidth_in()
    pl = pr = 0.015 * W
    gap = 0.045 * W
    pw = (W - pl - pr - 2 * gap) / 3.0
    ph = pw * py / px

    y_leg = 0.0
    pad_b = y_leg + 3 * 0.150 + 0.10 + 0.20
    fig_h = pad_b + ph + 0.22
    t1, t2, need = required_pixels(px, pw)
    npx = (need, int(round(need * py / px)))
    print(f'  fig4 pixels: feature {t1:.0f}, print {t2:.0f} -> {need} per panel '
          f'(clad {CLAD_THICK_INNER*need/px:.2f} px)')

    fig = plt.figure(figsize=(W, fig_h))
    allk = set()
    panels = [('S', 'standard element'), ('C', 'control element'),
              ('F', 'flux trap')]
    for k, (tok, cap) in enumerate(panels):
        ax = fig.add_axes([(pl + k * (pw + gap)) / W, pad_b / fig_h,
                           pw / W, ph / fig_h])
        ox, oy = _cell_centre(tok)
        allk |= render(ax, geom, basis='xy', origin=(ox, oy, 0.0),
                       width=(px, py), pixels=npx, outline=True)
        panel_letter(fig, ax, '(%s)' % 'abc'[k])
        caption(fig, ax, cap)
    RENDERED['fig4_elements'] = allk
    material_legend(fig, allk, y_in=y_leg, ncol=2)
    return save(fig, 'fig4_elements', target_in=W, expect_px=npx, axes_in=pw)


# =============================================================================
# SOLIDWORKS EXPORT -- dimensions and appearances, both cascaded from source
# =============================================================================

SW_LENGTHS = [
    ('PITCH_X', PITCH_X), ('PITCH_Y', PITCH_Y), ('ELEM_X', ELEM_X),
    ('ELEM_Y', ELEM_Y), ('ELEM_Z', ELEM_Z), ('GAP_X', GAP_X), ('GAP_Y', GAP_Y),
    ('SIDE_PLATE_THICK', SIDE_PLATE_THICK), ('ACTIVE_STACK_X', ACTIVE_STACK_X),
    ('PLATE_THICK_INNER', PLATE_THICK_INNER),
    ('PLATE_THICK_OUTER', PLATE_THICK_OUTER),
    ('CLAD_THICK_INNER', CLAD_THICK_INNER),
    ('CLAD_THICK_OUTER', CLAD_THICK_OUTER), ('MEAT_THICK', MEAT_THICK),
    ('MEAT_WIDTH', MEAT_WIDTH), ('WATER_CHAN_THICK', WATER_CHAN_THICK),
    ('STD_STACK_HEIGHT', STD_STACK_HEIGHT), ('STD_END_WATER', STD_END_WATER),
    ('CTRL_PLATE_PITCH', CTRL_PLATE_PITCH),
    ('CTRL_FUEL_STACK_HALF', CTRL_FUEL_STACK_HALF),
    ('CTRL_FUEL_WIDTH_X', CTRL_FUEL_WIDTH_X),
    ('CTRL_SIDE_PLATE_X', CTRL_SIDE_PLATE_X),
    ('CTRL_AL_PLATE_THICK', CTRL_AL_PLATE_THICK),
    ('CTRL_FEEDER_CHANNEL', CTRL_FEEDER_CHANNEL),
    ('CTRL_BLADE_WATER', CTRL_BLADE_WATER),
    ('CTRL_OUTER_OFFSET', CTRL_OUTER_OFFSET),
    ('CTRL_END_BLOCK', CTRL_END_BLOCK), ('ABSORBER_THICK', ABSORBER_THICK),
    ('FT_HOLE_RADIUS', FT_HOLE_RADIUS), ('HALF_Z', HALF_Z),
    ('CORE_TOP', CORE_TOP), ('CORE_BOTTOM', CORE_BOTTOM),
    ('ENDBOX_ABOVE_TOP', ENDBOX_ABOVE_TOP),
    ('ENDBOX_BELOW_BOT', ENDBOX_BELOW_BOT), ('BLADE_LENGTH', BLADE_LENGTH),
    ('ROD_TRAVEL', ROD_TRAVEL),
    # --- axial build-up, added 2026-08-03 -----------------------------------
    # The plate is 62 tall with 60 of meat inside it; the 1 cm difference at
    # each end is unfueled cladding, and the end-box picks up at 31.
    ('PLATE_HEIGHT', PLATE_HEIGHT),      # full plate height, incl. clad extension
    ('MEAT_HEIGHT', MEAT_HEIGHT),        # active fuel meat only
    ('CLAD_EXT', CLAD_EXT),              # unfueled clad above AND below the meat
    ('ENDBOX_HEIGHT', ENDBOX_HEIGHT),    # homogenized end-box, runs 31 -> 45
    # --- absorber blade ------------------------------------------------------
    # ABSORBER_WIDTH is the B4C blade itself. It is NOT the coolant channel
    # width, which is ACTIVE_STACK_X = 6.64 above — a separate constant that
    # this one was aliased to until B3. The blade is narrower than the slot it
    # rides in; the 0.005 per side is water.
    ('ABSORBER_WIDTH', ABSORBER_WIDTH),  # B4C blade width (channel is 6.64)
    ('BLADE_TOP_CLAD', BLADE_TOP_CLAD),  # Al cap, TOP END ONLY — no side or
                                         # bottom clad, absent by confirmation
]
SW_COUNTS = [('N_PLATES_STD', N_PLATES_STD),
             ('N_CTRL_FUEL_PLATES', N_CTRL_FUEL_PLATES)]
SW_DERIVED = [
    ('LATERAL_CLAD_MARGIN', '( "ACTIVE_STACK_X" - "MEAT_WIDTH" ) / 2',
     LATERAL_CLAD_MARGIN),
    ('SIDE_INNER', '"ACTIVE_STACK_X" / 2', SIDE_INNER),
    ('MEAT_HALF_WIDTH', '"MEAT_WIDTH" / 2', MEAT_WIDTH / 2),
    ('STD_STACK_HALF', '"STD_STACK_HEIGHT" / 2', STD_STACK_HEIGHT / 2),
]


def _provenance():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                         cwd=REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain', 'model/geometry.py'],
            cwd=REPO_ROOT, text=True).strip())
    except Exception:
        commit, dirty = 'unknown', False
    return commit, dirty, datetime.date.today().isoformat()


def _sw_eval(expr, env):
    import re as _re
    return eval(_re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"', r'env["\1"]', expr),
                {'__builtins__': {}}, {'env': env})


def write_solidworks_equations(path=None):
    path = path or os.path.join(OUT_DIR, 'solidworks_equations.txt')
    commit, dirty, stamp = _provenance()
    env = dict(SW_LENGTHS); env.update(dict(SW_COUNTS))
    for n, e, v in SW_DERIVED:
        got = _sw_eval(e, env)
        assert abs(got - v) < 1e-12, f'{n}: expression gives {got}, source {v}'
        env[n] = got
    L = ["' " + '=' * 72,
         "' IAEA-TECDOC-643 Appendix A-2, generic 10 MW LEU research reactor",
         "' SolidWorks external equations -- GENERATED FILE, DO NOT EDIT",
         "' " + '=' * 72,
         f"' Generated  {stamp} by figures/make_figures.py",
         f"' Source     model/geometry.py at commit {commit}"
         + ('  *** WORKING TREE DIRTY ***' if dirty else ''),
         "'",
         "' REGENERATE WHENEVER model/geometry.py CHANGES:",
         "'     python figures/make_figures.py",
         "' then re-link in SolidWorks (Equations > Link to external file).",
         "'",
         "' UNITS: lengths carry an explicit cm suffix. Counts are unitless.",
         "' ORDER matters: SolidWorks resolves globals top to bottom.",
         "' " + '=' * 72, '', "' ---- lengths ----"]
    # round to 6 dp before formatting: derived constants are the end of
    # float chains and SolidWorks should not receive 0.004999999999999893
    L += [f'"{n}" = {round(v, 6):g}cm' for n, v in SW_LENGTHS]
    L += ['', "' ---- counts (unitless) ----"]
    L += [f'"{n}" = {v:g}' for n, v in SW_COUNTS]
    L += ['', "' ---- derived: expressions, so SolidWorks recomputes them ----"]
    L += [f'"{n}" = {e}' for n, e, _v in SW_DERIVED] + ['']
    open(path, 'w').write('\n'.join(L) + '\n')
    print(f'  Saved: {path}  ({len(SW_LENGTHS)+len(SW_COUNTS)+len(SW_DERIVED)} '
          f'equations, geometry.py @ {commit}' + (', DIRTY' if dirty else '') + ')')
    return path


def write_solidworks_appearances(path=None):
    """Region -> RGB, emitted from the same figstyle palette the figures use.

    Cascaded, not hand-synced: change figstyle.REGION_FILL and both the renders
    and the SolidWorks appearances move together.
    """
    path = path or os.path.join(OUT_DIR, 'solidworks_appearances.txt')
    commit, dirty, stamp = _provenance()
    L = ["' " + '=' * 72,
         "' SolidWorks appearance table -- GENERATED FILE, DO NOT EDIT",
         "' " + '=' * 72,
         f"' Generated  {stamp} by figures/make_figures.py",
         f"' Source     figures/figstyle.py REGION_FILL, at commit {commit}"
         + ('  *** WORKING TREE DIRTY ***' if dirty else ''),
         "'",
         "' These are the SAME colours the manuscript renders use, so a part",
         "' shaded from this table matches its figure. SolidWorks has no external",
         "' appearance-link mechanism equivalent to linked equations, so set",
         "' these by hand (Appearance > Colour > RGB) or drive them from a macro.",
         "' " + '=' * 72, '',
         f"'  {'region':16s} {'R':>4s} {'G':>4s} {'B':>4s}   hex        display name"]
    for k in LUM_ORDER:
        r, g, b = rgb255(k)
        L.append(f"'  {k:16s} {r:4d} {g:4d} {b:4d}   {REGION_FILL[k]}   "
                 f"{display_name(k)}")
    L.append('')
    open(path, 'w').write('\n'.join(L) + '\n')
    print(f'  Saved: {path}  ({len(LUM_ORDER)} appearances)')
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description='TECDOC-643 A-2 reference figures')
    p.add_argument('--only', choices=['fig1', 'fig2', 'fig3', 'fig4'],
                   action='append')
    p.add_argument('--f', type=float, default=0.0)
    p.add_argument('--equations-only', action='store_true')
    a = p.parse_args(argv)
    want = set(a.only) if a.only else {'fig1', 'fig2', 'fig3', 'fig4'}
    print('TECDOC-643 A-2 reference figures')
    print(f'  textwidth {S.TEXTWIDTH_PT:.4f} TeX pt = {textwidth_in():.6f} in')
    check_geometry_consistency()
    if a.equations_only:
        write_solidworks_equations(); write_solidworks_appearances(); return
    apply_style()
    if 'fig1' in want: fig1_core_map()
    if 'fig2' in want: fig2_core_xy(f=a.f)
    if 'fig3' in want: fig3_axial_xz(f=a.f)
    if 'fig4' in want:
        fig4_elements(f=a.f)
        fig4_split(f=a.f)
    write_solidworks_equations()
    write_solidworks_appearances()


if __name__ == '__main__':
    main()
