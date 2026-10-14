"""
geometry.py
-----------
Geometry definitions for the IAEA TECDOC-643 Appendix A-2
Generic 10 MW LEU Research Reactor Core (Argonne design).

Reference:
    IAEA-TECDOC-643, "Research Reactor Core Conversion Guidebook,
    Volume 2: Analysis (Appendices A-F)," IAEA, Vienna, 1992.
    Appendix A-2: Generic 10 MW Reactor — Argonne National Laboratory.

Core Layout:
    - CORE: 5 x 6 = 30 positions — 23 standard fuel elements, 5 control
      fuel elements, 2 flux traps. [TECDOC A-2 Table 1, "Active Core
      Geometry: 5 x 6 Positions"]. The 12 graphite reflector positions are
      NOT part of the core.
    - LATTICE as built: 6 (x) x 7 (y) = 42 positions — the 30 core positions
      plus the 12 graphite reflector positions, which must live in the same
      RectLattice. This is a modelling extent, not a core description.
    - GRID PLATE: 8 x 9 positions [TECDOC A-2 Table 1, "Grid Plate: 8 x 9
      Positions"] — counts the surrounding water ring. Cite for that row only.
    - Lattice pitch: 77 mm x 81 mm
    - Active fuel meat height: 60 cm; plate height 62 cm

Axial model structure (symmetric about z=0):
    CORE_BOTTOM = -90 cm  (vacuum)
    [-90, -45]  : 45 cm light water
    [-45, -31]  : 14 cm homogenized end-box (0.25 Al / 0.75 H₂O by volume)
    [-31, -30]  :  1 cm unfueled clad extension
    [-30, +30]  : 60 cm active fuel meat
    [+30, +31]  :  1 cm unfueled clad extension
    [+31, +45]  : 14 cm homogenized end-box
    [+45, +90]  : 45 cm light water
    CORE_TOP    = +90 cm  (vacuum) — COINCIDES with the fully-withdrawn (f=1)
                  blade top; no water cap above the withdrawn blade.
    Sum check: 2 * (45 + 14 + 1 + 30) = 180 cm — tripwired below.

Lateral model structure:
    The 6 x 7 lattice is enclosed in an explicit water-filled pool box whose
    lateral faces sit POOL_WATER_THICK = 38.5 cm outboard of the lattice
    envelope. Vacuum boundary at the pool faces.
        x: 6 * 7.7 + 2 * 38.5 = 123.2 cm
        y: 7 * 8.1 + 2 * 38.5 = 133.7 cm
    Pool water is the 294 K bulk water material, NOT the 316.8 K core coolant.

Control blade model — fixed-length sliding absorber:
    BLADE_LENGTH = 60 cm (rigid; never changes)
    ROD_TRAVEL   = 60 cm (full stroke)
    withdrawn_fraction f in [0, 1]:
        z_bot = -30 + f * 60   → f=0: -30,  f=1: +30
        z_top = z_bot + 60     → f=0: +30,  f=1: +90 (= CORE_TOP at f=1)
    b4c fills the absorber-slot x/y band for z in [z_bot, z_top]. A
    BLADE_TOP_CLAD (1 cm) aluminum clad rides directly on the B4C (top end
    only — side and bottom clad are absent by confirmation), and an
    ENDBOX_HEIGHT (14 cm) homogenized (end_box_homog) end-box cap rides on
    the CLAD. Both fill the slot footprint and clip at CORE_TOP. At f=0 the
    stack is B4C [-30,+30], clad [+30,+31], cap [+31,+45] — the cap COPLANAR
    with the surrounding end-boxes falls out of BLADE_TOP_CLAD == CLAD_EXT
    (2026-07-31; supersedes the A4 Option B water band, which the clad now
    fills). At f=1 the B4C top coincides with CORE_TOP (z_top == +90) — it is
    the B4C, not the aluminum, that touches the boundary — and clad and cap
    are clipped entirely out of the model.
    Plate/clad, structural and channel cells run the full plate height
    z=[-31, +31]; only the fuel meat is restricted to z=[-30, +30].
    End-box/water cells cover z outside [-31, +31].

Standard Fuel Element (LEU, U3Si2-Al, heterogeneous build):
    - Envelope:           76 x 80 mm
    - Side plates:        4.8 mm each (aluminum, in x)
    - Active stack:       66.4 mm wide between side plate inner faces
    - 23 plates:          1.27 mm inner, 1.5 mm outer (outer plates clad on
                          both faces of the meat at the outer 0.495 mm
                          thickness, not just the face away from the stack)
    - Fuel meat:          0.51 mm thick x 63 mm wide x 600 mm tall
    - Plate height:       620 mm (600 mm meat + 10 mm unfueled clad each end)
    - Inner clad:         0.38 mm  |  Outer clad: 0.495 mm

All dimensions in cm.
"""

import openmc
from materials import (fuel, clad, water, water_core, b4c, graphite, aluminum,
                       end_box_homog, N_X_ZONES, N_AXIAL_ZONES, make_zoned_fuel)

# =============================================================================
# NUMERICAL FLOORS
# =============================================================================

# Smallest residual gap this model is willing to emit as a real coolant region.
#
# WHAT THIS IS FOR. Most water gaps here are RESIDUALS — what is left of an
# envelope after the solids inside it are subtracted. A residual has no lower
# bound of its own: edit a thickness upstream and the gap shrinks silently,
# through "thin", through "sub-millimetre", to zero and then negative, and the
# only symptom is a sliver cell that eats tracking time or an inverted region
# that OpenMC reports as a hole. A `> 0` assert catches only the last of those.
# This floor catches the approach.
#
# 0.01 cm (0.1 mm) is [ASSUMED] — our own choice, no external authority. It is
# an order of magnitude under the smallest real coolant dimension in the model
# (WATER_CHAN_THICK = 0.219) and an order of magnitude over float noise on
# these derivation chains (~1e-12), so it separates "someone made a mistake"
# from "this is genuinely a thin channel" without being sensitive to rounding.
#
# ONE DELIBERATE EXEMPTION EXISTS: ABSORBER_SIDE_WATER (0.005) is half this
# floor and is CORRECT — see the exception at its definition, which enforces
# the exemption rather than merely tolerating it.
MIN_RESIDUAL_GAP = 0.01   # cm                                        [ASSUMED]

# =============================================================================
# LATTICE / ELEMENT ENVELOPE
# =============================================================================

# Lattice pitch. NOTE THE SUBSECTION: this is the only geometry constant that
# comes from Table 1's REACTOR Design Description rather than its Fuel Element
# Design Descriptions. [TECDOC A-2 Table 1, Reactor Design Description] (MCNP MATCH)
PITCH_X = 7.7    # cm  (77 mm)
PITCH_Y = 8.1    # cm  (81 mm)

# Element envelope, 76 x 80 mm. ELEM_Y is also the side plate width (8.000).
# [TECDOC A-2 Table 1, Fuel Element Design Descriptions, LEU column] (MCNP MATCH)
ELEM_X = 7.6     # cm  (76 mm)
ELEM_Y = 8.0     # cm  (80 mm)

# --- Axial plate / meat heights ---------------------------------------------
# The reference MCNP model carries 1 cm of UNFUELED cladding above and below
# the active meat: the plates stand 62 cm tall with 60 cm of meat inside them.
# MEAT_HEIGHT and PLATE_HEIGHT are INDEPENDENT primaries; the clad extension is
# derived from the pair and is never written as a literal anywhere.
MEAT_HEIGHT  = 60.0   # cm (600 mm) — active fuel meat height
                      # completes the meat block 0.51 x 63 x 600 mm
                      # [TECDOC A-2 Table 1, Fuel Element] (MCNP MATCH)
PLATE_HEIGHT = 62.0   # cm (620 mm) — fuel / unfueled plate height   [MCNP]
CLAD_EXT     = (PLATE_HEIGHT - MEAT_HEIGHT) / 2.0   # 1.0 cm          [DERIVED]

assert CLAD_EXT > 0, "PLATE_HEIGHT must exceed MEAT_HEIGHT (clad extension <= 0)"

# Element dimension (Z). Side plates, unfueled control plates, flux-trap blocks
# and reflector blocks all run the full plate height, not the meat height.
#
# KNOWN TECDOC / MCNP DIVERGENCE, MCNP GOVERNS. TECDOC-643 A-2 Table 1 gives
# the element dimension as 600 mm; this model carries 620 mm on Kyle's
# confirmation (B1), because the reference MCNP model adds 1 cm of unfueled
# cladding at each end. The divergence is deliberate and is recorded here so
# it stays visible rather than silently contradicting the benchmark.
ELEM_Z = PLATE_HEIGHT   # 62.0 cm                                     [MCNP]

# Inter-element water gap — documentation/tripwire only. Feeds no surface or
# cell directly; every gap cell derives its width from the pitch/envelope
# XPlane/YPlane objects themselves (PITCH_X/Y, ELEM_X/Y above), not from
# these constants. Exists so a future PITCH/ELEM edit that zeroes or inverts
# the gap fails loudly here instead of emitting a zero-width sliver cell.
GAP_X = (PITCH_X - ELEM_X) / 2.0   # cm
GAP_Y = (PITCH_Y - ELEM_Y) / 2.0   # cm
assert GAP_X > 0, "PITCH_X must exceed ELEM_X (zero/negative gap)"
assert GAP_Y > 0, "PITCH_Y must exceed ELEM_Y (zero/negative gap)"

# Solid (non-plate) block dimensions — flux trap and graphite reflector.
# These blocks sit inside the lattice pitch with the SAME 1 mm water gap as the
# fuel elements: 7.6 x 8.0 inside the 7.7 x 8.1 pitch. This is what resolves the
# old "graphite inter-block water channel" question — the channel IS the pitch
# gap (GAP_X in x, GAP_Y in y). There is no separate channel dimension to find.
# The homogenized end-box above and below these blocks stays FULL PITCH.
FT_BLOCK_X   = ELEM_X   # 7.600 cm                                    [MCNP]
FT_BLOCK_Y   = ELEM_Y   # 8.000 cm                                    [MCNP]
REFL_BLOCK_X = ELEM_X   # 7.600 cm                                    [MCNP]
REFL_BLOCK_Y = ELEM_Y   # 8.000 cm                                    [MCNP]

assert abs((PITCH_X - FT_BLOCK_X) / 2.0 - GAP_X) < 1e-12, \
    "flux-trap block does not leave the standard inter-element gap in x"
assert abs((PITCH_Y - FT_BLOCK_Y) / 2.0 - GAP_Y) < 1e-12, \
    "flux-trap block does not leave the standard inter-element gap in y"
assert abs((PITCH_X - REFL_BLOCK_X) / 2.0 - GAP_X) < 1e-12, \
    "reflector block does not leave the standard inter-element gap in x"
assert abs((PITCH_Y - REFL_BLOCK_Y) / 2.0 - GAP_Y) < 1e-12, \
    "reflector block does not leave the standard inter-element gap in y"

# Side plate thickness. NOT in Table 1 — it comes from the dimensioned drawing
# in Fig. 2's TOP-LEFT panel. The other three panels of Fig. 2 are homogenized
# diffusion models, not physical geometry, and are not authoritative for any
# dimension. Cite the panel, never Fig. 2 generally.
# [TECDOC A-2 Fig. 2 top-left] (MCNP MATCH)
SIDE_PLATE_THICK = 0.48   # cm (4.8 mm)

# Interior coolant channel width — the clear span between the side plates.
# 6.640, a MATCH row, and NOT the absorber blade width (see ABSORBER_WIDTH).
ACTIVE_STACK_X   = ELEM_X - 2 * SIDE_PLATE_THICK   # 6.64 cm        [DERIVED]

# =============================================================================
# PLATE / MEAT / CLAD DIMENSIONS
# =============================================================================

# Plate, clad and meat cross-section. Table 1 gives plate 1.27 mm, clad
# 0.38 mm inner / 0.495 mm outer, meat 0.51 x 63 x 600 mm.
# [TECDOC A-2 Table 1, Fuel Element Design Descriptions, LEU column] (MCNP MATCH)
PLATE_THICK_INNER = 0.127    # cm (1.27 mm)  inner plate

CLAD_THICK_INNER = 0.038     # cm (0.38 mm)  inner clad
CLAD_THICK_OUTER = 0.0495    # cm (0.495 mm) outer clad

MEAT_THICK = 0.051           # cm (0.51 mm)  meat thickness
MEAT_WIDTH = 6.3             # cm (63 mm)    meat width

# Outer plates (first/last in the stack) are clad at the outer thickness on
# BOTH faces of the meat, not just the face away from the stack — so their
# total thickness is meat + 2*CLAD_THICK_OUTER, not meat + inner + outer.
#
# HISTORY — READ BEFORE "CORRECTING" THIS BACK TO A LITERAL. Until 4bfd5d5 this
# was hardcoded `PLATE_THICK_OUTER = 0.1385`, i.e. meat + inner + outer, which
# is the wrong construction: it clads the inward face at the INNER thickness.
# 4bfd5d5 replaced the literal with the derivation and moved the value to the
# 0.15 that Table 1 and the reference MCNP model both carry. The literal is
# gone deliberately — it cannot drift out of step with the clad and meat
# constants it is made of, and the assert below is what says so.
#
# WARNING ON THE LOG MESSAGES. The 0.1385 era is still visible in commit
# messages and in older run logs, where the outer plate is reported at
# 1.385 mm. Those numbers describe a superseded geometry. Any log or archived
# result printing 1.385 predates 4bfd5d5 and must not be compared against
# current output as though the two were the same model.
PLATE_THICK_OUTER = MEAT_THICK + 2 * CLAD_THICK_OUTER   # 0.15 cm  [DERIVED]
                                                        # (1.5 mm, MCNP MATCH)

# The INNER plate is the other side of the same construction, and unlike the
# outer plate it is a primary [TECDOC] literal rather than a derivation — so
# nothing forces it to stay consistent with the meat and clad it is made of.
# This is that force. It is what makes a lone MEAT_THICK edit fail loudly
# instead of quietly producing a plate whose parts no longer sum to it.
#
# NOTE FOR THE "thick" CONFIGURATION: the OQ-3 cascade (meat primary, plate
# thicknesses derived from it) is not implemented until the spec layer, so
# until then MEAT_THICK and PLATE_THICK_INNER must be edited TOGETHER. This
# assert refusing a lone MEAT_THICK edit is the assert working, not a defect.
assert abs((PLATE_THICK_INNER - 2 * CLAD_THICK_INNER) - MEAT_THICK) < 1e-12, (
    f"inner plate does not decompose: PLATE_THICK_INNER {PLATE_THICK_INNER} "
    f"- 2 x CLAD_THICK_INNER {CLAD_THICK_INNER} = "
    f"{PLATE_THICK_INNER - 2 * CLAD_THICK_INNER}, not MEAT_THICK {MEAT_THICK}")

# Lateral aluminum margin — the clad shoulder each side of the meat, between
# the 6.3 cm meat width and the 6.64 cm clear span between the side plates.
# Residual, never a literal.
#
# This lived in figures/make_figures.py as a local definition with a bare
# `> 0` assert, which meant the constraint only ran when somebody built the
# figures. It belongs next to the constants it is made of, and it is imported
# from here now — the figure module no longer defines its own.
LATERAL_CLAD_MARGIN = (ACTIVE_STACK_X - MEAT_WIDTH) / 2.0   # 0.17 cm [DERIVED]

assert LATERAL_CLAD_MARGIN >= MIN_RESIDUAL_GAP, (
    f"lateral clad margin is {LATERAL_CLAD_MARGIN:.6f} cm, under the "
    f"{MIN_RESIDUAL_GAP} cm floor — the meat is running out to the side "
    f"plates. Check MEAT_WIDTH ({MEAT_WIDTH}) against ACTIVE_STACK_X "
    f"({ACTIVE_STACK_X}); at MEAT_WIDTH >= 6.62 the shoulder goes sub-floor "
    f"and at 6.64 the meat touches the side plates outright")

# Plate counts. Table 1: 23 plates per standard element, "17 + 4 Al plates"
# per control element. [TECDOC A-2 Table 1, Fuel Element] (MCNP MATCH)
N_PLATES_STD  = 23           # fuel plates per standard element
N_PLATES_CTRL = 17           # fuel plates per control element

# Interior coolant channel, plate to plate. Table 1 gives 2.19 mm.
# [TECDOC A-2 Table 1, Fuel Element] (MCNP MATCH)
WATER_CHAN_THICK = 0.219     # cm (2.19 mm)

# Standard element plate-stack height and the residual end water gap between
# the outermost plate face and the element envelope edge. [DERIVED]
STD_STACK_HEIGHT = (2 * PLATE_THICK_OUTER
                    + (N_PLATES_STD - 2) * PLATE_THICK_INNER
                    + (N_PLATES_STD - 1) * WATER_CHAN_THICK)   # 7.785 cm
# Exterior (element-end) coolant channel: 0.1075. A3 leaves this [DERIVED] —
# the MCNP column is blank, so there is nothing to reconcile it against yet.
STD_END_WATER = (ELEM_Y - STD_STACK_HEIGHT) / 2.0             # 0.1075 cm  [DERIVED]

# Was `> 0`. That only caught the gap after it had already inverted; this is a
# pure residual of a 23-plate stack inside a fixed envelope, so every plate,
# clad and channel thickness above feeds it and any of them can walk it down.
# At the 0.1075 default there is a factor of ten of headroom over the floor.
assert STD_END_WATER >= MIN_RESIDUAL_GAP, (
    f"standard element end water gap is {STD_END_WATER:.6f} cm, under the "
    f"{MIN_RESIDUAL_GAP} cm floor — the {N_PLATES_STD}-plate stack "
    f"({STD_STACK_HEIGHT:.6f} cm) has grown to fill the {ELEM_Y} cm envelope. "
    f"Check the plate, clad and channel thicknesses that feed it")

# Flux trap cylindrical water hole radius. [MCNP] — Kyle confirmed, closes A1
# (2026-07-31). Area-equivalent to the 50 mm square hole: 25 cm^2 gives
# r = 5/sqrt(pi) = 2.8209 cm, carried as 2.820 per the reference MCNP model.
# (The old 2.500 was the inscribed radius of the same square, [ASSUMED].)
FT_HOLE_RADIUS = 2.820       # cm [MCNP]

# The hole must clear the 7.6 x 8.0 flux-trap block on both axes — with 2.820
# the aluminum margins are (7.6 - 5.64)/2 = 0.98 cm in x and
# (8.0 - 5.64)/2 = 1.18 cm in y.
assert 2 * FT_HOLE_RADIUS < FT_BLOCK_X, \
    "flux-trap hole diameter exceeds the block in x"
assert 2 * FT_HOLE_RADIUS < FT_BLOCK_Y, \
    "flux-trap hole diameter exceeds the block in y"

# HALF_Z is the ACTIVE MEAT half-height and must track MEAT_HEIGHT, not ELEM_Z.
# Since B1 the two differ (60 vs 62): the blade travel, the meat cells and the
# depletion zone tiling all key off HALF_Z, and deriving it from ELEM_Z would
# silently move the meat to +/-31 along with the plates.
HALF_Z       = MEAT_HEIGHT / 2.0     # 30.0 cm — active meat half-height  [DERIVED]
HALF_PLATE_Z = PLATE_HEIGHT / 2.0    # 31.0 cm — plate / clad half-height [DERIVED]

assert abs(HALF_PLATE_Z - (HALF_Z + CLAD_EXT)) < 1e-12, \
    "HALF_PLATE_Z must equal HALF_Z + CLAD_EXT"

# =============================================================================
# AXIAL MODEL EXTENTS AND FIXED-LENGTH BLADE PARAMETERS
# =============================================================================

# Absorber blade height. [MCNP] — cross-validation spreadsheet row "Absorber
# (B4C) blade height", MCNP 60.0000, MATCH.
BLADE_LENGTH     = 60.0    # cm — rigid absorber blade (fixed length, translates in z)

# Full stroke. [DERIVED] from MEAT_HEIGHT, NOT from BLADE_LENGTH — the two are
# both 60.0 by coincidence, not by relationship. The stroke is fixed by this
# model's blade convention: at f=0 the blade spans the active meat exactly, and
# at f=1 its bottom sits exactly on the meat top, so
#     z_bot(f=1) = -HALF_Z + ROD_TRAVEL = +HALF_Z  =>  ROD_TRAVEL = MEAT_HEIGHT.
# The convention itself ("fully withdrawn" = blade clear of the active meat) is
# this project's modelling choice; the spreadsheet carries no rod-stroke row.
# Deriving it from MEAT_HEIGHT is what stops a compensating error in
# ROD_TRAVEL and BLADE_LENGTH from satisfying the f=1 pin below.
ROD_TRAVEL       = MEAT_HEIGHT   # 60.0 cm                            [DERIVED]

# Blade TOP cladding — aluminum riding on the B4C, part of the blade assembly.
# [MCNP] Kyle confirmed 2026-07-31: 1.0 cm, TOP END ONLY (side and bottom clad
# are absent by confirmation, not omission). Derived from CLAD_EXT because it
# is the same physical 1 cm as the fuel plates' unfueled extension — if Kyle
# later decouples them, this is a one-line change. Never written as 1.0.
# The B4C itself stays BLADE_LENGTH = 60 and ROD_TRAVEL stays 60: it is the
# B4C that touches CORE_TOP at f=1, and the clad above it clips away at the
# model boundary exactly as the cap does.
BLADE_TOP_CLAD   = CLAD_EXT   # 1.0 cm [MCNP]
# Model axial extent. [MCNP] — cross-validation spreadsheet row "Model
# dimension (Z)", MCNP 180.000, MATCH. Note the confirmed quantity is the
# 180.000 TOTAL; the +/-90 split follows from the symmetry-about-z=0 assert
# below, not from a separate confirmed row.
CORE_TOP         = +90.0   # cm — vacuum boundary                          [MCNP]
                            # COINCIDES with the fully-withdrawn (f=1) blade top
                            # (z_top = -30 + 60 + 60 = +90). No cap above.
CORE_BOTTOM      = -90.0   # cm — vacuum boundary; symmetric with CORE_TOP  [MCNP]

# Homogenized end-box axial extent. The plates gaining 1 cm at each end and the
# end-box losing 1 cm are the same centimetre — +/-45 and +/-90 do not move.
ENDBOX_HEIGHT    = 14.0    # cm — homogenized end-box height            [MCNP]

# BENCHMARK PIN. ENDBOX_HEIGHT is a primary [MCNP] literal, and every axial
# tripwire below it is written in terms of ENDBOX_HEIGHT itself — so they all
# stay satisfied for ANY value it takes, sliding +/-45 up and down the model
# with it. The axial stack still sums to 180 because POOL_WATER_AXIAL is the
# residual that absorbs the change. Nothing downstream notices, and the model
# builds a geometry that is no longer the benchmark.
#
# This pin is the only thing in the file that holds the value itself. It is
# deliberately a bare literal compared against a bare literal: writing it in
# terms of anything else here would make it slide too.
assert abs(ENDBOX_HEIGHT - 14.0) < 1e-12, (
    f"ENDBOX_HEIGHT is {ENDBOX_HEIGHT}, not the 14.0 cm of the reference "
    f"MCNP model. The axial tripwires below are all written in terms of "
    f"ENDBOX_HEIGHT and will happily pass at the wrong value — POOL_WATER_AXIAL "
    f"absorbs the difference and the stack still sums to 180 cm. If this is a "
    f"deliberate revision, change it here and re-run the cross-validation; do "
    f"not delete the pin")

ENDBOX_ABOVE_TOP = HALF_PLATE_Z + ENDBOX_HEIGHT    # +45.0 cm           [DERIVED]
ENDBOX_BELOW_BOT = -ENDBOX_ABOVE_TOP               # −45.0 cm           [DERIVED]
POOL_WATER_AXIAL = CORE_TOP - ENDBOX_ABOVE_TOP     # +45.0 cm           [DERIVED]

# POOL_WATER_AXIAL is the axial residual — CORE_TOP minus everything stacked
# under it. It is what silently absorbs an ENDBOX_HEIGHT or PLATE_HEIGHT edit
# (see the pin above), so it is the one axial quantity that can go to zero or
# invert while every symmetry and sum tripwire below still passes.
assert POOL_WATER_AXIAL >= MIN_RESIDUAL_GAP, (
    f"upper pool water is {POOL_WATER_AXIAL:.6f} cm, under the "
    f"{MIN_RESIDUAL_GAP} cm floor — the end-box top ({ENDBOX_ABOVE_TOP}) has "
    f"reached CORE_TOP ({CORE_TOP}). Check ENDBOX_HEIGHT ({ENDBOX_HEIGHT}) "
    f"and PLATE_HEIGHT ({PLATE_HEIGHT})")

# Symmetry / height tripwires. Documentation + guard only: none of these feed a
# cell or surface directly. Tolerance-based, not ==, since every value here is
# now the end of a float derivation chain.
assert abs(CORE_TOP + CORE_BOTTOM) < 1e-12, "axial model must be symmetric about z=0"
assert abs((CORE_TOP - CORE_BOTTOM) - 180.0) < 1e-12, "total axial height must be 180 cm"
# REMOVED 2026-08-24 — the upper-water assert was vacuous. It read
#     assert abs((CORE_TOP - ENDBOX_ABOVE_TOP) - POOL_WATER_AXIAL) < 1e-12
# and POOL_WATER_AXIAL is DEFINED as CORE_TOP - ENDBOX_ABOVE_TOP twenty lines
# above, so it compared an expression to itself and could not fail for any
# input. It is replaced by nothing, because there is nothing here to check: the
# real constraint on the upper water is the floor assert at POOL_WATER_AXIAL's
# definition. Recorded rather than silently dropped so this is not read as a
# lost check.
#
# The LOWER twin below is KEPT and is NOT the same case. It compares
# ENDBOX_BELOW_BOT - CORE_BOTTOM against POOL_WATER_AXIAL, which is built from
# the TOP pair — so it is vacuous only while the model is symmetric, and that
# symmetry is an assumption enforced by the assert above, not a definition. It
# is a real cross-check that the bottom mirrors the top.
assert abs((ENDBOX_BELOW_BOT - CORE_BOTTOM) - POOL_WATER_AXIAL) < 1e-12, \
    "lower water region must be POOL_WATER_AXIAL"
assert abs((ENDBOX_ABOVE_TOP - HALF_PLATE_Z) - ENDBOX_HEIGHT) < 1e-12, \
    "upper end-box must be ENDBOX_HEIGHT tall"
assert abs((-HALF_PLATE_Z - ENDBOX_BELOW_BOT) - ENDBOX_HEIGHT) < 1e-12, \
    "lower end-box must be ENDBOX_HEIGHT tall"

# B1 tripwire — the whole axial stack, layer by layer, must close on 180 cm:
#   2 * (45 water + 14 end-box + 1 clad extension + 30 half-meat) = 180
_AXIAL_STACK_SUM = 2.0 * (POOL_WATER_AXIAL + ENDBOX_HEIGHT + CLAD_EXT + HALF_Z)
assert abs(_AXIAL_STACK_SUM - 180.0) < 1e-12, (
    f"axial stack sums to {_AXIAL_STACK_SUM} cm, not 180 cm — layers: "
    f"water {POOL_WATER_AXIAL}, end-box {ENDBOX_HEIGHT}, clad ext {CLAD_EXT}, "
    f"half-meat {HALF_Z}")
assert abs(_AXIAL_STACK_SUM - (CORE_TOP - CORE_BOTTOM)) < 1e-12, \
    "axial layer sum disagrees with the CORE_BOTTOM..CORE_TOP model extent"

# Blade travel is unchanged by B1: the f=1 blade top must still land exactly on
# CORE_TOP, which is what makes the withdrawn case create no cap at all.
#
# READ BEFORE SIMPLIFYING EITHER SIDE. This pin combines values from three
# different sources that happen to close exactly:
#   ROD_TRAVEL   = MEAT_HEIGHT   — a PROJECT CONVENTION ("fully withdrawn"
#                                  means the blade clears the active meat);
#                                  no spreadsheet row exists for it
#   BLADE_LENGTH = 60.0          — [MCNP], spreadsheet MATCH
#   CORE_TOP     = +90.0         — [MCNP], from the 180.000 model-Z row
# That the B4C top touches the model top at full withdrawal is real — Kyle
# confirmed it, and it is why the withdrawn case creates no cap. But it is an
# ALIGNMENT OF INDEPENDENTLY SOURCED VALUES, not a derivation: none of the
# three follows from the other two. Do not "simplify" by defining one in terms
# of the others. If any of the three is ever revised on its own the alignment
# breaks, and this assert is what says so.
assert abs((-HALF_Z + ROD_TRAVEL + BLADE_LENGTH) - CORE_TOP) < 1e-12, \
    "f=1 blade top no longer coincides with CORE_TOP — cap logic assumes it does"

# The pin above constrains only the SUM of ROD_TRAVEL and BLADE_LENGTH, so
# offsetting errors in the two would satisfy it. Pin each independently:
# ROD_TRAVEL to the stroke convention it derives from, BLADE_LENGTH to its
# spreadsheet row.
assert abs(ROD_TRAVEL - MEAT_HEIGHT) < 1e-12, (
    f"ROD_TRAVEL is {ROD_TRAVEL}, not MEAT_HEIGHT {MEAT_HEIGHT} — the f=1 "
    f"blade bottom would no longer land on the meat top")
assert abs(BLADE_LENGTH - 60.0) < 1e-12, (
    f"BLADE_LENGTH is {BLADE_LENGTH}, not the 60.000 of the cross-validation "
    f"row 'Absorber (B4C) blade height' (MCNP 60.0000)")

# Shared axial ZPlane surfaces — transmission (NOT vacuum boundaries).
# Defined once at module level and reused in every element universe to avoid
# creating redundant surfaces at identical z-values.
# _z_fuel_* bound the ACTIVE MEAT (+/-30); _z_plate_* bound the PLATES and every
# structural cell (+/-31). The 1 cm between them is unfueled cladding.
_z_fuel_bot     = openmc.ZPlane(z0=-HALF_Z)           # −30.0 cm
_z_fuel_top     = openmc.ZPlane(z0= HALF_Z)           # +30.0 cm
_z_plate_bot    = openmc.ZPlane(z0=-HALF_PLATE_Z)     # −31.0 cm
_z_plate_top    = openmc.ZPlane(z0= HALF_PLATE_Z)     # +31.0 cm
_z_endbox_above = openmc.ZPlane(z0=ENDBOX_ABOVE_TOP)  # +45.0 cm
_z_endbox_below = openmc.ZPlane(z0=ENDBOX_BELOW_BOT)  # −45.0 cm
_z_model_top    = openmc.ZPlane(z0=CORE_TOP)           # +90.0 cm
_z_model_bot    = openmc.ZPlane(z0=CORE_BOTTOM)        # −90.0 cm


# =============================================================================
# FUEL MEAT DEPLETION ZONES — 2D, x (width) x z (axial)
#
# The zone COUNTS and their provenance live in materials.py, which owns
# N_X_ZONES and N_AXIAL_ZONES; read the tagging block there before changing
# anything here. In short: 2 x 10 is [MCNP — Kyle confirmed 2026-08-12], it
# supersedes both an [ASSUMED] 8 x 20 and an unconfirmed [MCNP-VISUAL] reading
# of 5 axial zones, and it is NOT the 8-per-half-core of [TECDOC-643 App. A-2,
# Sec. 3.1]. Kyle's answer confirms the per-plate SUBDIVISION only — the sharing
# of one material across all plates of an element remains unconfirmed.
# Nothing in THIS block is [TECDOC] or [MCNP]; every value here is [DERIVED]
# from MEAT_WIDTH / MEAT_HEIGHT and the counts.
#
# The zone bounds are derived from the ACTIVE MEAT extents themselves, NOT from
# the element envelope. ELEM_Z is the element extent, and since B1 the two HAVE
# diverged (62 cm element vs 60 cm meat); zones derived from ELEM_Z would tile
# 62 cm and silently mis-size every depletion volume. The same trap exists in x
# — ELEM_X is 7.6 and ACTIVE_STACK_X is 6.64, neither of which is the 6.3 cm
# meat. The tiling asserts below test the quantities that matter, in both
# directions.
#
# y (the plate stacking direction) is NOT subdivided — each plate is a single
# meat band in y. Every plate does, however, carry its OWN materials: cells and
# depletable materials are 1:1 [MCNP — Kyle confirmed 2026-08-12]. The former
# element-shared scheme was the last [MCNP-VISUAL] inference and is retired.
#
# Zoning is opt-in (build_core_geometry(depletion_zoning=True)). With it off,
# none of these surfaces are created and the model is unchanged.
# =============================================================================

MEAT_BOT_Z  = _z_fuel_bot.z0     # −30.0 cm — active meat lower bound
MEAT_TOP_Z  = _z_fuel_top.z0     # +30.0 cm — active meat upper bound

# The x meat bounds have no module-level shared surfaces to read back (unlike
# _z_fuel_bot/_z_fuel_top): meat_left/meat_right are built per element universe.
# These constants are therefore the SINGLE SOURCE those per-element planes are
# built from, so the zone tiling and the element geometry cannot diverge.
MEAT_LEFT_X  = -MEAT_WIDTH / 2.0   # −3.15 cm — active meat −x bound
MEAT_RIGHT_X =  MEAT_WIDTH / 2.0   # +3.15 cm — active meat +x bound

# Tolerance for the zone tiling asserts. Needed because MEAT_WIDTH / N_X_ZONES
# is not generally exact in binary: at N_X_ZONES = 2 the residual happens to be
# exactly 0 (division by a power of two), but e.g. N_X_ZONES = 3 leaves ~9e-16.
# 1e-12 is ~3 orders of margin on the worst case measured.
ZONE_TILE_TOL = 1e-12

# MEAT_HEIGHT is a module-level primary (see the envelope block); it used to be
# derived here off these two planes. Since B1 the plates (62 cm) and the meat
# (60 cm) are different heights, so the derivation is inverted into a check:
# the meat planes must still bound exactly MEAT_HEIGHT, or the zone tiling
# below is sizing depletion volumes against the wrong stack.
assert abs((MEAT_TOP_Z - MEAT_BOT_Z) - MEAT_HEIGHT) < ZONE_TILE_TOL, \
    "active meat planes do not bound MEAT_HEIGHT"
assert abs((MEAT_RIGHT_X - MEAT_LEFT_X) - MEAT_WIDTH) < ZONE_TILE_TOL, \
    "active meat x bounds do not span MEAT_WIDTH"

MEAT_ZONE_HEIGHT = MEAT_HEIGHT / N_AXIAL_ZONES   # 6.0 cm  for N_AXIAL_ZONES=10
MEAT_ZONE_WIDTH  = MEAT_WIDTH  / N_X_ZONES       # 3.15 cm for N_X_ZONES=2

# 2D zone volume for ONE plate: width x height x the meat thickness. Every plate
# carries the same meat thickness (outer plates are clad at CLAD_THICK_OUTER on
# both faces, inner at CLAD_THICK_INNER, and both leave exactly MEAT_THICK), so
# one constant is valid for all 614 plates.
MEAT_ZONE_VOLUME_PER_PLATE = MEAT_THICK * MEAT_ZONE_WIDTH * MEAT_ZONE_HEIGHT

assert abs((MEAT_BOT_Z + N_AXIAL_ZONES * MEAT_ZONE_HEIGHT) - MEAT_TOP_Z) < ZONE_TILE_TOL, \
    "axial zones do not tile the active meat height"
assert abs((MEAT_LEFT_X + N_X_ZONES * MEAT_ZONE_WIDTH) - MEAT_RIGHT_X) < ZONE_TILE_TOL, \
    "x zones do not tile the active meat width"

# Interior zone dividers, created ONCE and reused across all 28 fueled element
# universes — never inside the per-element builder, which would put
# (N-1) x 28 coincident redundant planes in the model. Both element types use
# the same MEAT_WIDTH, so one set of x planes serves standard and control alike.
#
# Created lazily rather than at import: module-level surface construction
# consumes the global auto-ID counter and would shift every subsequent surface
# ID, so the zoning-OFF model would stop being byte-for-byte identical to the
# Phase One baseline.
#
# NOTE THE ASYMMETRY between the two directions below. In z the OUTER bounds are
# module-level shared surfaces (_z_fuel_bot/_z_fuel_top), so zone_z_bounds() can
# reuse them itself. In x there is no module-level meat surface — meat_left and
# meat_right are local to each element builder — so zone_x_bounds() takes them
# as arguments. Hoisting them to module level would shift the surface auto-ID
# counter and break the zoning-OFF baseline.
_FUEL_ZONE_PLANES   = None
_FUEL_ZONE_X_PLANES = None


def fuel_zone_planes():
    """The N_AXIAL_ZONES-1 interior zone ZPlanes (−24, −18, … +24 for N=10)."""
    global _FUEL_ZONE_PLANES
    if _FUEL_ZONE_PLANES is None:
        _FUEL_ZONE_PLANES = [
            openmc.ZPlane(z0=MEAT_BOT_Z + k * MEAT_ZONE_HEIGHT)
            for k in range(1, N_AXIAL_ZONES)
        ]
    return _FUEL_ZONE_PLANES


def fuel_zone_x_planes():
    """The N_X_ZONES-1 interior zone XPlanes (a single plane at 0 for N=2).

    At an even N_X_ZONES one of these lands exactly on the meat centreline
    x = 0. No other surface in this model sits at x0 = 0 — every XPlane built
    anywhere in this file is one of a ± symmetric pair — so the centreline
    plane is coincident with nothing.
    """
    global _FUEL_ZONE_X_PLANES
    if _FUEL_ZONE_X_PLANES is None:
        _FUEL_ZONE_X_PLANES = [
            openmc.XPlane(x0=MEAT_LEFT_X + j * MEAT_ZONE_WIDTH)
            for j in range(1, N_X_ZONES)
        ]
    return _FUEL_ZONE_X_PLANES


def zone_z_bounds(k):
    """(lower, upper) ZPlane surfaces bounding axial zone k.

    zone 0 = bottom (z = MEAT_BOT_Z). The outermost bounds reuse the existing
    shared active-fuel planes, so no duplicate surface is made at ±30.
    """
    planes = fuel_zone_planes()
    return (_z_fuel_bot if k == 0 else planes[k - 1],
            _z_fuel_top if k == N_AXIAL_ZONES - 1 else planes[k])


def zone_x_bounds(j, meat_left, meat_right):
    """(lower, upper) XPlane surfaces bounding width zone j.

    zone 0 = −x edge (x = MEAT_LEFT_X). `meat_left`/`meat_right` are the calling
    element's OWN meat edge planes and are reused for the outermost bounds, so
    no duplicate surface is made at ±3.15 — the x analogue of what zone_z_bounds
    does with the shared ±30 planes. See the asymmetry note above for why they
    have to be passed in rather than read from module scope.
    """
    planes = fuel_zone_x_planes()
    return (meat_left  if j == 0 else planes[j - 1],
            meat_right if j == N_X_ZONES - 1 else planes[j])



# =============================================================================
# STANDARD FUEL ELEMENT
# 23 plates stacked in y, running in x. Plates are 62 cm tall (z) with 60 cm
# of meat inside them. All structural cells are bounded to the plate height
# z=[-31, +31]; only the meat stops at z=[-30, +30].
# End-box and water regions fill the full pitch footprint above/below.
# =============================================================================

def make_standard_fuel_element(elem_id, element_id=None, zoned=False):
    """
    Standard ANL/TECDOC A-2 fuel element.

    X = plate meat width direction (side plates bound this)
    Y = plate/channel stack direction (plates stacked here)
    Z = axial (plates -31 to +31 cm; active fuel meat -30 to +30 cm)

    elem_id     integer index, unchanged — drives every cell name.
    element_id  core-map position label ('B4', ...) — depletion zoning only.
    zoned       when True, each plate's meat is split into
                N_X_ZONES x N_AXIAL_ZONES cells on an (x, z) grid, each cell
                filled by its OWN depletable material — cells and materials are
                1:1 and nothing is shared between plates. When False the element
                is built exactly as it always has been.
    """
    if zoned and element_id is None:
        raise ValueError("zoned=True requires a core-map element_id label")

    # Zoned materials are created per PLATE, inside the plate loop below, where
    # the plate index is in scope — one material per meat cell, nothing shared.

    # Pitch cell boundaries
    pitch_left  = openmc.XPlane(x0=-PITCH_X / 2.0)
    pitch_right = openmc.XPlane(x0= PITCH_X / 2.0)
    pitch_front = openmc.YPlane(y0=-PITCH_Y / 2.0)
    pitch_back  = openmc.YPlane(y0= PITCH_Y / 2.0)

    # Element envelope
    box_left  = openmc.XPlane(x0=-ELEM_X / 2.0)
    box_right = openmc.XPlane(x0= ELEM_X / 2.0)
    box_front = openmc.YPlane(y0=-ELEM_Y / 2.0)
    box_back  = openmc.YPlane(y0= ELEM_Y / 2.0)

    # Side plate inner faces
    side_inner_left  = openmc.XPlane(x0=-ELEM_X / 2.0 + SIDE_PLATE_THICK)
    side_inner_right = openmc.XPlane(x0= ELEM_X / 2.0 - SIDE_PLATE_THICK)

    # Fuel meat X boundaries — built from the module-level constants so the zone
    # tiling in zone_x_bounds() and this element's geometry share one source.
    meat_left  = openmc.XPlane(x0=MEAT_LEFT_X)
    meat_right = openmc.XPlane(x0=MEAT_RIGHT_X)

    # Axial bounds — reuse module-level surfaces (avoids redundant surface IDs).
    # meat_z* bound the fuel meat only (+/-30); plate_z bounds the plates and
    # every structural cell (+/-31). The clad cells subtract the full meat
    # region, so the 1 cm bands [+/-30, +/-31] inside the meat footprint come
    # out as unfueled cladding with no extra cell.
    meat_zbot = _z_fuel_bot   # −30 cm
    meat_ztop = _z_fuel_top   # +30 cm
    plate_z   = +_z_plate_bot & -_z_plate_top   # [−31, +31]

    cells = []

    plate_thicks = (
        [PLATE_THICK_OUTER]
        + [PLATE_THICK_INNER] * (N_PLATES_STD - 2)
        + [PLATE_THICK_OUTER]
    )

    stack_height_y = sum(plate_thicks) + (N_PLATES_STD - 1) * WATER_CHAN_THICK
    # Tie the built stack to the module-level derived end-water gap.
    assert abs((ELEM_Y - stack_height_y) / 2.0 - STD_END_WATER) < 1e-9, \
        "standard stack end water gap disagrees with STD_END_WATER"
    y = -stack_height_y / 2.0
    stack_bottom_surf = openmc.YPlane(y0=y)

    for i, plate_thick in enumerate(plate_thicks):
        is_first = (i == 0)
        is_last  = (i == N_PLATES_STD - 1)

        plate_bottom = openmc.YPlane(y0=y)
        plate_top    = openmc.YPlane(y0=y + plate_thick)

        if is_first or is_last:
            # Outer plates: clad at the outer thickness on BOTH faces of the
            # meat (not just the face away from the stack).
            clad_bottom = CLAD_THICK_OUTER
            clad_top    = CLAD_THICK_OUTER
        else:
            clad_bottom = CLAD_THICK_INNER 
            clad_top    = CLAD_THICK_INNER

        meat_bottom = openmc.YPlane(y0=y + clad_bottom)
        meat_top    = openmc.YPlane(y0=y + plate_thick - clad_top)

        # Meat: bounded in x, y, AND z (active zone only)
        # meat_y is the y band alone. Zone cells substitute their own x bounds
        # for meat_left/meat_right rather than intersecting on top of them, so a
        # zone cell carries the same 6 half-spaces an unzoned meat cell does.
        meat_y  = +meat_bottom & -meat_top
        meat_xy = +meat_left & -meat_right & meat_y
        meat_region = meat_xy & +meat_zbot & -meat_ztop

        # Plate region bounded to active zone
        plate_region = (
            +side_inner_left & -side_inner_right &
            +plate_bottom & -plate_top &
            plate_z
        )

        if zoned:
            # One cell per (x, z) zone, and one MATERIAL per cell — plate i gets
            # its own, shared with nothing. The clad cell below still subtracts
            # the FULL meat_region, so the zone cells tile the same volume the
            # single meat cell occupied — no gap, no overlap.
            for j in range(N_X_ZONES):
                x_lo, x_hi = zone_x_bounds(j, meat_left, meat_right)
                for k in range(N_AXIAL_ZONES):
                    z_lo, z_hi = zone_z_bounds(k)
                    cells.append(openmc.Cell(
                        name=f'std{elem_id}_meat_{i}_x{j}_z{k}',
                        fill=make_zoned_fuel(element_id, i, j, k,
                                             MEAT_ZONE_VOLUME_PER_PLATE),
                        region=+x_lo & -x_hi & meat_y & +z_lo & -z_hi
                    ))
        else:
            cells.append(openmc.Cell(
                name=f'std{elem_id}_meat_{i}',
                fill=fuel,
                region=meat_region
            ))
        cells.append(openmc.Cell(
            name=f'std{elem_id}_clad_{i}',
            fill=clad,
            region=plate_region & ~meat_region
        ))

        y += plate_thick

        if not is_last:
            chan_bottom = plate_top
            chan_top    = openmc.YPlane(y0=y + WATER_CHAN_THICK)
            cells.append(openmc.Cell(
                name=f'std{elem_id}_chan_{i}',
                fill=water_core,
                region=(
                    +side_inner_left & -side_inner_right &
                    +chan_bottom & -chan_top &
                    plate_z
                )
            ))
            y += WATER_CHAN_THICK

    stack_top_surf = openmc.YPlane(y0=y)

    # Water below and above the plate stack — active zone only
    cells.append(openmc.Cell(
        name=f'std{elem_id}_water_below_stack',
        fill=water_core,
        region=(
            +box_front & -stack_bottom_surf &
            +side_inner_left & -side_inner_right &
            plate_z
        )
    ))
    cells.append(openmc.Cell(
        name=f'std{elem_id}_water_above_stack',
        fill=water_core,
        region=(
            +stack_top_surf & -box_back &
            +side_inner_left & -side_inner_right &
            plate_z
        )
    ))

    # Side plates — active zone only
    cells.append(openmc.Cell(
        name=f'std{elem_id}_side_left',
        fill=aluminum,
        region=(
            +box_left & -side_inner_left &
            +box_front & -box_back &
            plate_z
        )
    ))
    cells.append(openmc.Cell(
        name=f'std{elem_id}_side_right',
        fill=aluminum,
        region=(
            +side_inner_right & -box_right &
            +box_front & -box_back &
            plate_z
        )
    ))

    # Inter-element water gaps — active zone only
    cells.append(openmc.Cell(
        name=f'std{elem_id}_gap_xleft',
        fill=water_core,
        region=(
            +pitch_left & -box_left &
            +pitch_front & -pitch_back &
            plate_z
        )
    ))
    cells.append(openmc.Cell(
        name=f'std{elem_id}_gap_xright',
        fill=water_core,
        region=(
            +box_right & -pitch_right &
            +pitch_front & -pitch_back &
            plate_z
        )
    ))
    cells.append(openmc.Cell(
        name=f'std{elem_id}_gap_yfront',
        fill=water_core,
        region=(
            +box_left & -box_right &
            +pitch_front & -box_front &
            plate_z
        )
    ))
    cells.append(openmc.Cell(
        name=f'std{elem_id}_gap_yback',
        fill=water_core,
        region=(
            +box_left & -box_right &
            +box_back & -pitch_back &
            plate_z
        )
    ))

    # ── Axial regions above/below the active fuel ──────────────────────────
    # End-box is one solid full-pitch homogenized block — no inter-element
    # water gap subdivision (the end-box material is already a homogenized
    # Al/water mixture, so a physical gap slice within it is not meaningful).
    full_pitch = +pitch_left & -pitch_right & +pitch_front & -pitch_back

    cells.append(openmc.Cell(
        name=f'std{elem_id}_upper_endbox',
        fill=end_box_homog,
        region=full_pitch & +_z_plate_top & -_z_endbox_above   # +31 → +45 cm
    ))
    cells.append(openmc.Cell(
        name=f'std{elem_id}_upper_water',
        fill=water,
        region=full_pitch & +_z_endbox_above & -_z_model_top
    ))
    cells.append(openmc.Cell(
        name=f'std{elem_id}_lower_endbox',
        fill=end_box_homog,
        region=full_pitch & +_z_endbox_below & -_z_plate_bot   # −45 → −31 cm
    ))
    cells.append(openmc.Cell(
        name=f'std{elem_id}_lower_water',
        fill=water,
        region=full_pitch & +_z_model_bot & -_z_endbox_below
    ))

    return openmc.Universe(name=f'std_fuel_elem_{elem_id}', cells=cells)


# =============================================================================
# CONTROL ELEMENT
# Architecture: two end blocks + central 17-plate fuel follower stack, built
# on the SAME standard 0.127 cm plate / 0.219 cm channel pitch as the standard
# fuel element (TECDOC A-2 Table 1: "17 + 4 Al plates").
#
#   Follower fuel stack (17 plates + 16 channels), centered on the element:
#     half-width = (17*PLATE_THICK_INNER + 16*WATER_CHAN_THICK) / 2 = 2.8315 cm
#
#   Each end, from the fuel stack outward to the element wall:
#     [feeder channel 0.219 | Al inner guide 0.150 | blade water g |
#      B4C blade slot 0.310 | blade water g | Al outer guide 0.150 |
#      outer offset water OUTER_OFFSET]
#   The feeder channel is a standard fuel-to-fuel water channel (matches the
#   follower's own plate pitch). The two blade-flanking water gaps are EQUAL
#   (even spacing) and are the residual after every other layer is fixed:
#     g = (END_BLOCK - 2*CTRL_AL_PLATE_THICK - ABSORBER_THICK - CTRL_OUTER_OFFSET
#          - CTRL_FEEDER_CHANNEL) / 2
#   where END_BLOCK = ELEM_Y/2 - CTRL_FUEL_STACK_HALF (1.1685 cm, fixed by the
#   element envelope and the fuel stack half-width above).
#
# Fixed-length sliding blade:
#   The B4C absorber blade is BLADE_LENGTH=60 cm long and translates in z.
#   At fraction f, the blade occupies z=[z_bot, z_top] = [-30+f*60, +30+f*60].
#   b4c fills the absorber-slot x/y band for z in [z_bot, z_top] across the full
#   model height. Below z_bot, water fills the slot down to the plate bottom
#   (-31) — the blade never dips below z=-30, so the lower end-box/water are
#   uniform material with no reserved slot at all. Above z_top the assembly
#   stack rides with the blade: BLADE_TOP_CLAD (1 cm) aluminum clad, then the
#   14 cm cap riding on the clad, then water to CORE_TOP — each clipped at
#   CORE_TOP; at f=1, z_top == CORE_TOP so clad and cap are both clipped away.
#   All guide/slider/fuel/channel cells are bounded to the PLATE height
#   z=[-31, +31]; only the fuel meat stops at z=[-30, +30]; end-box/water cells
#   fill z outside [-31, +31].
# =============================================================================

# Absorber blade thickness (y). [MCNP] — cross-validation spreadsheet row
# "Absorber (B4C) blade thickness", MCNP 0.3100, MATCH.
ABSORBER_THICK  = 0.31   # cm                                          [MCNP]

# Pin ABSORBER_THICK to its spreadsheet value directly. This is the D3 fix: the
# control end-block budget closes BY CONSTRUCTION because CTRL_BLADE_WATER is
# its residual, so an error in this constant would be absorbed silently into
# the blade-water gap and every layer-sum assert would still pass. Until now
# the only thing catching that was the CTRL_BLADE_WATER pin further down —
# a single point of failure. Pinning both ends removes it.
assert ABSORBER_THICK > 0.0, "absorber blade thickness must be positive"
assert abs(ABSORBER_THICK - 0.310) < 1e-12, (
    f"ABSORBER_THICK is {ABSORBER_THICK}, not the 0.310 of the "
    f"cross-validation row 'Absorber (B4C) blade thickness' (MCNP 0.3100)")

# Absorber blade WIDTH (x). This is a different physical quantity from
# ACTIVE_STACK_X = 6.640, the coolant channel width, which is unchanged and
# stays a MATCH row. Before B3 the two were ALIASED: the absorber slot took its
# x-extent straight from CTRL_FUEL_WIDTH_X (= ACTIVE_STACK_X), so the blade was
# 6.640 wide by construction and there was no way to move one without the other.
# They are now independent, and the assert below stops them being re-merged.
#
# KNOWN TECDOC / MCNP DIVERGENCE, MCNP GOVERNS. TECDOC-643 A-2 Fig. 2's
# bottom-right panel shows 6.60; the reference MCNP model gives 6.630 (B3),
# and ACTIVE_STACK_X is a third value again at 6.640. Three different numbers
# for adjacent quantities — recorded so the divergence stays visible.
ABSORBER_WIDTH  = 6.63   # cm                                          [MCNP]

# The blade is narrower than the slot it sits in, so a thin water film runs
# down each side of it. Residual, never a literal.
ABSORBER_SIDE_WATER = (ACTIVE_STACK_X - ABSORBER_WIDTH) / 2.0   # 0.005 cm [DERIVED]

assert ABSORBER_WIDTH != ACTIVE_STACK_X, (
    "ABSORBER_WIDTH has been re-aliased to ACTIVE_STACK_X — the blade width "
    "(6.630) and the coolant channel width (6.640) are different constants")
assert 0.0 < ABSORBER_WIDTH < ACTIVE_STACK_X, \
    "absorber blade must be positive and no wider than the slot it slides in"
assert ABSORBER_SIDE_WATER > 0.0, \
    "absorber side water film is degenerate — check ABSORBER_WIDTH"

# THE ONE EXEMPTION FROM MIN_RESIDUAL_GAP, MADE SELF-ENFORCING.
#
# ABSORBER_SIDE_WATER is 0.005 cm — half the 0.01 cm floor. That is CORRECT and
# deliberate: it is not a residual that got away from us, it is the real
# clearance between a 6.630 blade and the 6.640 channel it slides in, and both
# numbers are [MCNP]. Exempting it with a comment would leave the exemption
# unpoliced, so the exemption is written as its own check instead.
#
# READ THE DIRECTION CAREFULLY — THIS FIRES WHEN THE FILM GETS *BIGGER*.
# The condition is `< MIN_RESIDUAL_GAP`, so it holds while the film stays
# sliver-scale and fires once it stops being one. That is the point: a film at
# or above the floor is no longer the tight blade-in-channel clearance this
# exemption was granted for, which means something upstream moved and the
# exemption needs re-deciding rather than silently widening.
#
# Three ways in, all measured:
#   * SIDE_PLATE_THICK was reduced to about 0.474 or less — ACTIVE_STACK_X
#     widens, the channel opens up around a fixed blade. (At exactly 0.475 the
#     film lands on 0.01 from below by ~2e-16 and this still passes; the
#     threshold is a float edge, not a round number.)
#   * ABSORBER_WIDTH was lowered to about 6.61 or less -- the blade is
#     over-narrow and the side film is no longer sliver-scale. (Widening it
#     back toward ACTIVE_STACK_X does NOT fire this; that drives the film to
#     zero and trips ABSORBER_SIDE_WATER > 0 above.) Like the SIDE_PLATE_THICK
#     row, the threshold is a float edge and not a round number: at exactly
#     6.62 the film lands on 0.01 from below by ~2e-16 and this still passes.
#   * MIN_RESIDUAL_GAP itself was lowered to at or below the film — at 0.004
#     the exemption becomes unnecessary and this says so, because an exemption
#     that no longer exempts anything should not stay on the books.
if not ABSORBER_SIDE_WATER < MIN_RESIDUAL_GAP:
    raise AssertionError(
        f"ABSORBER_SIDE_WATER is {ABSORBER_SIDE_WATER:.6f} cm, no longer below "
        f"the {MIN_RESIDUAL_GAP} cm floor it is exempted from.\n"
        f"This film is deliberately sub-floor (0.005 = the 6.640 channel minus "
        f"the 6.630 blade, halved). It reaching the floor means one of:\n"
        f"  * SIDE_PLATE_THICK reduced to about 0.474 or less "
        f"(now {SIDE_PLATE_THICK}, giving ACTIVE_STACK_X {ACTIVE_STACK_X:.6f})\n"
        f"  * ABSORBER_WIDTH lowered to about 6.61 or less "
        f"(now {ABSORBER_WIDTH}) — an over-narrow blade, NOT an undoing of B3\n"
        f"  * MIN_RESIDUAL_GAP lowered to at or below the film "
        f"(now {MIN_RESIDUAL_GAP}), which retires the exemption\n"
        f"Decide which, then update the exemption — do not delete this check.")

CTRL_FUEL_WIDTH_X   = ACTIVE_STACK_X
CTRL_SIDE_PLATE_X   = SIDE_PLATE_THICK
# Unfueled (control-element) plate thickness. B5: ours is the CORRECT side —
# the MCNP model's 0.150 is being corrected there. Do not change this to match.
# [TECDOC A-2 Table 1, Fuel Element] — the "4 Al plates" of the control
# element's "17 + 4" row, at the same 1.27 mm as the fuel plates.
CTRL_AL_PLATE_THICK = 0.127   # cm (1.27 mm) — was 0.15 (an Argonne
                              # TH-analysis convenience); reverted 2026-07-20.
# CTRL_HF_THICK removed 2026-07-31: a dead Hf-era alias of ABSORBER_THICK,
# defined but never read anywhere in the repository. Use ABSORBER_THICK (y) and
# ABSORBER_WIDTH (x) — an unused second name for a blade dimension is exactly
# how the B3 blade-width/channel-width aliasing happened in the first place.

N_CTRL_FUEL_PLATES  = 17
CTRL_PLATE_PITCH    = PLATE_THICK_INNER + WATER_CHAN_THICK   # 0.346 cm

# N_PLATES_CTRL (module top) and N_CTRL_FUEL_PLATES are two names for the same
# 17-plate follower stack — pre-existing duplication. The follower loop and the
# zoned-material volumes both key off N_CTRL_FUEL_PLATES; this tripwire stops
# the two from drifting apart and silently mis-sizing a depletion volume.
assert N_CTRL_FUEL_PLATES == N_PLATES_CTRL, \
    "N_CTRL_FUEL_PLATES and N_PLATES_CTRL disagree on the follower plate count"

# Follower fuel stack half-width (standard 0.127/0.219 pitch, symmetric)
CTRL_FUEL_STACK_HALF = (N_CTRL_FUEL_PLATES * PLATE_THICK_INNER
                        + (N_CTRL_FUEL_PLATES - 1) * WATER_CHAN_THICK) / 2.0  # 2.8315 cm

# Feeder channel: the follower's outermost fuel plate to the inner guide
# plate is a standard fuel-to-fuel water channel, same width as every
# plate-to-plate channel in the stack above it.
CTRL_FEEDER_CHANNEL = WATER_CHAN_THICK   # 0.219 cm [DERIVED — standard channel]

# Control-element outer offset — the water between the outer guide plate and
# the element wall. De-aliased from STD_END_WATER (2026-07-31, closes A3): the
# standard element's exterior channel (0.1075, a derived residual of the plate
# stack) and the control element's outer offset are physically different
# things and should never have shared a constant, regardless of value.
#
# [MCNP] 0.1305: supplied directly by Kyle as the end water channel value
# (2026-07-31), who stated the reference MCNP model either already carries it
# or will be updated to it. The value may therefore POSTDATE the current state
# of the reference model.
CTRL_OUTER_OFFSET = 0.1305   # cm [MCNP]

# End-block budget: everything between the fuel stack edge and the wall.
CTRL_END_BLOCK = ELEM_Y / 2.0 - CTRL_FUEL_STACK_HALF   # 1.1685 cm

# Blade-flanking water gap — residual, split equally on both sides of the
# blade. Recomputes automatically if CTRL_OUTER_OFFSET (or any layer above)
# changes.
CTRL_BLADE_WATER = (CTRL_END_BLOCK - CTRL_FEEDER_CHANNEL
                    - 2.0 * CTRL_AL_PLATE_THICK - ABSORBER_THICK
                    - CTRL_OUTER_OFFSET) / 2.0
# With CTRL_OUTER_OFFSET = 0.1305 this evaluates to exactly:
#   (1.1685 - 0.219 - 2*0.127 - 0.31 - 0.1305) / 2 = 0.1275 cm

# Kyle quoted 0.1275 alongside 0.1305; hardcoding both would over-determine
# the end-block budget, so 0.1275 stays derived and this assert VALIDATES his
# number against the budget instead of trusting it. The PRO-X drawing's own
# closure (1.305 + 1.27 + 5.65 + 1.27 + 2.19 = 11.685 mm) is the same check
# from the other end.
#
# Labelling (2026-07-31 spreadsheet correction): CTRL_BLADE_WATER is the
# PER-SIDE water gap between a guide plate and the absorber. The spreadsheet
# row "Control guide coolant channel thickness" is NOT this quantity — it is
# the full inner span between the two guide plates that the blade rides in,
# ABSORBER_THICK + 2*CTRL_BLADE_WATER = 0.310 + 0.255 = 0.565 (Kyle: 0.5650,
# MATCH). 0.565 carries no assert of its own: ABSORBER_THICK is a module
# literal and CTRL_BLADE_WATER is pinned to 0.1275 by the assert below, so
# the span cannot move without tripping it.
assert abs(CTRL_BLADE_WATER - 0.1275) < 1e-9, (
    f"CTRL_BLADE_WATER derives to {CTRL_BLADE_WATER:.6f}, not the 0.1275 the "
    f"2026-07-31 A3 decision quotes — the end-block budget no longer closes "
    f"on Kyle's pair (outer offset {CTRL_OUTER_OFFSET})")

# Guide-channel span — the full inner span between the two guide plates that
# the blade rides in. This is the quantity the cross-validation row "Control
# guide coolant channel thickness" carries (Kyle: 0.5650), NOT the per-side
# CTRL_BLADE_WATER. Asserted directly rather than left pinned transitively
# through ABSORBER_THICK and CTRL_BLADE_WATER, so the spreadsheet row has an
# assert that names it.
CTRL_GUIDE_SPAN = ABSORBER_THICK + 2.0 * CTRL_BLADE_WATER   # 0.565 cm [DERIVED]

assert abs(CTRL_GUIDE_SPAN - 0.5650) < 1e-9, (
    f"CTRL_GUIDE_SPAN is {CTRL_GUIDE_SPAN:.6f}, not the 0.5650 of the "
    f"cross-validation row 'Control guide coolant channel thickness'")

assert CTRL_BLADE_WATER >= 0.05, (
    f"CTRL_BLADE_WATER={CTRL_BLADE_WATER:.5f} cm is degenerate for "
    f"CTRL_AL_PLATE_THICK={CTRL_AL_PLATE_THICK}, "
    f"CTRL_OUTER_OFFSET={CTRL_OUTER_OFFSET} — check end-block budget")

# TRIPWIRE, NOT A BOUND. The 0.05 above is the real physical limit for this
# channel and it binds first — at the 0.1275 default there is 0.0775 cm of
# headroom to it and 0.1175 cm to the floor, so in ordinary operation this line
# is unreachable and is MEANT to be.
#
# It is here so that the universal floor applies universally. CTRL_BLADE_WATER
# is a residual of the control end-block budget, and every other residual in
# this file is now floored; leaving this one governed solely by a local 0.05
# would mean a future edit that relaxes or removes that bound drops the channel
# straight out from under the model-wide floor with nothing left to catch it.
# If this ever fires, the 0.05 bound above has been weakened — look there
# first, not here.
assert CTRL_BLADE_WATER >= MIN_RESIDUAL_GAP, (
    f"CTRL_BLADE_WATER={CTRL_BLADE_WATER:.5f} cm is under the "
    f"{MIN_RESIDUAL_GAP} cm floor. This should be UNREACHABLE — the 0.05 bound "
    f"above binds first. Reaching it means that bound was relaxed or removed. "
    f"Check the end-block budget: CTRL_END_BLOCK={CTRL_END_BLOCK}, "
    f"CTRL_FEEDER_CHANNEL={CTRL_FEEDER_CHANNEL}, "
    f"CTRL_AL_PLATE_THICK={CTRL_AL_PLATE_THICK}, "
    f"CTRL_OUTER_OFFSET={CTRL_OUTER_OFFSET}")


def make_control_fuel_element(elem_id, withdrawn_fraction=0.0,
                              element_id=None, zoned=False):
    """
    Control fuel element with a fixed-length (60 cm) B4C absorber blade that
    translates in z.

    withdrawn_fraction f in [0, 1]:
        f=0 → blade at z=[-30, +30] (all-in, blade spans the active meat)
        f=1 → blade at z=[+30, +90] (all-out, blade entirely above active fuel)

    The blade always exists; only its z-position changes.

    elem_id     integer index, unchanged — drives every cell name.
    element_id  core-map position label ('C2', ...) — depletion zoning only.
    zoned       when True, each follower plate's meat is split into
                N_X_ZONES x N_AXIAL_ZONES cells on an (x, z) grid, each cell
                filled by its OWN depletable material — cells and materials are
                1:1, nothing shared between the 17 follower plates. The
                absorber slot lives
                in a different y-band than the meat (see the slot/meat
                disjointness note in the follower section below), so neither the
                axial nor the width cut touches the blade, its slot, or the
                sliding-cap logic. The x cut stays inside the meat, which is
                MEAT_WIDTH (6.3) wide against the ABSORBER_WIDTH (6.63) blade,
                so it cannot reach the blade in x either.
    """
    if zoned and element_id is None:
        raise ValueError("zoned=True requires a core-map element_id label")

    # Zoned materials are created per PLATE, inside the follower plate loop
    # below, where the plate index is in scope — one material per meat cell.

    f = withdrawn_fraction
    z_bot = -HALF_Z + f * ROD_TRAVEL   # blade bottom
    z_top = z_bot + BLADE_LENGTH        # blade top

    assert z_bot >= CORE_BOTTOM, (
        f"ctrl{elem_id}: blade bottom {z_bot:.2f} < CORE_BOTTOM {CORE_BOTTOM}")
    assert z_top <= CORE_TOP, (
        f"ctrl{elem_id}: blade top {z_top:.2f} > CORE_TOP {CORE_TOP}")
    # These two are the sole justification for (a) merging the lower end-box/
    # water into uniform material with no absorber-slot exclusion, and (b) never
    # needing an "above-active" water-gap cell (the blade always reaches at
    # least the top of the active zone). If travel or geometry parameters
    # ever change so these fail, both simplifications below become wrong.
    assert z_bot >= -HALF_Z, (
        f"ctrl{elem_id}: blade_z_bot={z_bot:.2f} < -HALF_Z ({-HALF_Z}) — "
        "blade would enter the lower end-box/water; lower-side merge is invalid")
    assert z_top >= HALF_Z, (
        f"ctrl{elem_id}: blade_z_top={z_top:.2f} < HALF_Z ({HALF_Z}) — "
        "blade would leave a water gap above it inside the active zone")
    print(f"ctrl{elem_id}: f={f:.3f}  blade z=[{z_bot:.2f}, {z_top:.2f}] cm"
          f"  (within [{CORE_BOTTOM}, {CORE_TOP}] ✓)")

    # Axial surfaces for this blade position
    blade_z_bot = openmc.ZPlane(z0=z_bot)
    blade_z_top = openmc.ZPlane(z0=z_top)
    plate_z    = +_z_plate_bot & -_z_plate_top   # [−31, +31]

    cells = []

    # Pitch cell boundaries
    pitch_left  = openmc.XPlane(x0=-PITCH_X / 2.0)
    pitch_right = openmc.XPlane(x0= PITCH_X / 2.0)
    pitch_front = openmc.YPlane(y0=-PITCH_Y / 2.0)
    pitch_back  = openmc.YPlane(y0= PITCH_Y / 2.0)

    # Element envelope
    elem_left  = openmc.XPlane(x0=-ELEM_X / 2.0)
    elem_right = openmc.XPlane(x0= ELEM_X / 2.0)
    elem_front = openmc.YPlane(y0=-ELEM_Y / 2.0)
    elem_back  = openmc.YPlane(y0= ELEM_Y / 2.0)

    # X-band for the interior stack (between side plates)
    side_inner_left  = openmc.XPlane(x0=-CTRL_FUEL_WIDTH_X / 2.0)
    side_inner_right = openmc.XPlane(x0= CTRL_FUEL_WIDTH_X / 2.0)

    # Fuel meat x/z bounds
    meat_zbot  = _z_fuel_bot
    meat_ztop  = _z_fuel_top
    meat_left  = openmc.XPlane(x0=MEAT_LEFT_X)
    meat_right = openmc.XPlane(x0=MEAT_RIGHT_X)

    # Y-layout — fuel stack is centered, half-width fixed by the standard
    # 0.127/0.219 pitch (CTRL_FUEL_STACK_HALF, module level).
    y_fuel_start = -CTRL_FUEL_STACK_HALF   # −2.8315 cm
    y_fuel_end   =  CTRL_FUEL_STACK_HALF   # +2.8315 cm

    # Bottom end block, built outward from the fuel stack to the wall:
    #   feeder channel (0.219) | inner guide (Al) | blade water (g) |
    #   B4C blade slot | blade water (g) | outer guide (Al) | outer offset water
    bot_slider_top = openmc.YPlane(y0=y_fuel_start - CTRL_FEEDER_CHANNEL)
    bot_slider_bot = openmc.YPlane(y0=bot_slider_top.y0 - CTRL_AL_PLATE_THICK)
    bot_slot_top     = openmc.YPlane(y0=bot_slider_bot.y0 - CTRL_BLADE_WATER)
    bot_slot_bot     = openmc.YPlane(y0=bot_slot_top.y0 - ABSORBER_THICK)
    bot_guide_top  = openmc.YPlane(y0=bot_slot_bot.y0 - CTRL_BLADE_WATER)
    bot_offset_top = openmc.YPlane(y0=bot_guide_top.y0 - CTRL_AL_PLATE_THICK)
    # bot_offset_top should coincide with elem_front + CTRL_OUTER_OFFSET
    assert abs(bot_offset_top.y0 - (-ELEM_Y / 2.0 + CTRL_OUTER_OFFSET)) < 1e-9, \
        "control end-block budget does not reach the element wall (bottom)"

    # Top end block — mirror image, built outward from the fuel stack to the wall.
    top_slider_bot = openmc.YPlane(y0=y_fuel_end + CTRL_FEEDER_CHANNEL)
    top_slider_top = openmc.YPlane(y0=top_slider_bot.y0 + CTRL_AL_PLATE_THICK)
    top_slot_bot     = openmc.YPlane(y0=top_slider_top.y0 + CTRL_BLADE_WATER)
    top_slot_top     = openmc.YPlane(y0=top_slot_bot.y0 + ABSORBER_THICK)
    top_guide_bot  = openmc.YPlane(y0=top_slot_top.y0 + CTRL_BLADE_WATER)
    top_guide_top  = openmc.YPlane(y0=top_guide_bot.y0 + CTRL_AL_PLATE_THICK)
    assert abs(top_guide_top.y0 - (ELEM_Y / 2.0 - CTRL_OUTER_OFFSET)) < 1e-9, \
        "control end-block budget does not reach the element wall (top)"

    # Absorber slot x/y footprints (unbounded in z — blade cells own their z-range).
    # The SLOT is ACTIVE_STACK_X wide (6.640, the coolant channel width); the
    # BLADE inside it is ABSORBER_WIDTH wide (6.630). Keeping the slot at the
    # full channel width leaves the not_slots end-box exclusion below
    # untouched and puts the side-water film inside the slot, where it belongs.
    slot_b = +bot_slot_bot & -bot_slot_top & +side_inner_left & -side_inner_right
    slot_t = +top_slot_bot & -top_slot_top & +side_inner_left & -side_inner_right

    blade_x_left  = openmc.XPlane(x0=-ABSORBER_WIDTH / 2.0)
    blade_x_right = openmc.XPlane(x0= ABSORBER_WIDTH / 2.0)
    blade_x       = +blade_x_left & -blade_x_right

    # ── Bottom sandwich structural cells (active zone only) ─────────────────
    # Wall -> fuel: offset water | outer guide | blade water | [blade] |
    #               blade water | inner guide | feeder channel | fuel

    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_offset_water_bottom', fill=water_core,
        region=(+elem_front & -bot_offset_top &
                +side_inner_left & -side_inner_right & plate_z)))

    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_guide_bottom', fill=aluminum,
        region=(+bot_offset_top & -bot_guide_top &
                +side_inner_left & -side_inner_right & plate_z)))

    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_blade_water_outer_bottom', fill=water_core,
        region=(+bot_guide_top & -bot_slot_bot &
                +side_inner_left & -side_inner_right & plate_z)))

    # (Absorber slot cells are handled separately below — not bounded to plate_z)
    #  they own their own z-ranges, driven by the blade position.

    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_blade_water_inner_bottom', fill=water_core,
        region=(+bot_slot_top & -bot_slider_bot &
                +side_inner_left & -side_inner_right & plate_z)))

    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_slider_bottom', fill=aluminum,
        region=(+bot_slider_bot & -bot_slider_top &
                +side_inner_left & -side_inner_right & plate_z)))

    # ── Top sandwich structural cells (active zone only) ────────────────────
    # Fuel -> wall: feeder channel | inner guide | blade water | [blade] |
    #               blade water | outer guide | offset water

    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_slider_top', fill=aluminum,
        region=(+top_slider_bot & -top_slider_top &
                +side_inner_left & -side_inner_right & plate_z)))

    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_blade_water_inner_top', fill=water_core,
        region=(+top_slider_top & -top_slot_bot &
                +side_inner_left & -side_inner_right & plate_z)))

    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_blade_water_outer_top', fill=water_core,
        region=(+top_slot_top & -top_guide_bot &
                +side_inner_left & -side_inner_right & plate_z)))

    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_guide_top', fill=aluminum,
        region=(+top_guide_bot & -top_guide_top &
                +side_inner_left & -side_inner_right & plate_z)))

    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_offset_water_top', fill=water_core,
        region=(+top_guide_top & -elem_back &
                +side_inner_left & -side_inner_right & plate_z)))

    # ── Fixed-length B4C blade ───────────────────────────────────────────────
    # B4C occupies [z_bot, z_top] in the absorber-slot band, unbounded by axial
    # region (spans across active/end-box/water boundaries as one piece).
    blade_z = +blade_z_bot & -blade_z_top
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_absorber_bottom', fill=b4c,
        region=slot_b & blade_x & blade_z))
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_absorber_top', fill=b4c,
        region=slot_t & blade_x & blade_z))

    # ABSORBER_SIDE_WATER film down each side of the blade — the slot is
    # ACTIVE_STACK_X wide, the blade ABSORBER_WIDTH. Without these the two
    # 0.005 cm slivers beside each blade are undefined space, which passes an
    # overlap check and leaks particles.
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_blade_side_water_bottom', fill=water_core,
        region=slot_b & ~blade_x & blade_z))
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_blade_side_water_top', fill=water_core,
        region=slot_t & ~blade_x & blade_z))

    # Water below the blade, down to the BOTTOM OF THE PLATES (−31), not the
    # bottom of the meat. The lower end-box now stops at −31 and carries no
    # slot exclusion, so if this stopped at −30 the slot band [−31, −30] would
    # be undefined space: it passes an overlap check and leaks particles.
    # The blade never dips below −30 (asserted above), so this band is always
    # water regardless of f.
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_slot_b_water_below', fill=water_core,
        region=slot_b & +_z_plate_bot & -blade_z_bot))
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_slot_t_water_below', fill=water_core,
        region=slot_t & +_z_plate_bot & -blade_z_bot))

    # ── Moving homogenized end-box cap (A4, resolved 2026-07-31: Option B) ──
    # An ENDBOX_HEIGHT (14 cm) end_box_homog cap rides above the blade in the
    # absorber-slot x/y band, clipped at CORE_TOP; above the cap the slot is water
    # (294 K) up to CORE_TOP.
    #
    # The cap bottom is max(z_top, HALF_PLATE_Z), NOT z_top. B1 shortened the
    # cap from 15 cm to 14 cm while the plates grew to +/-31, so a cap bolted
    # rigidly to the blade top would sit at [+30,+44] at f=0 — 1 cm low, and no
    # longer coplanar with the surrounding end-boxes at [+31,+45]. The
    # 2026-07-20 decision is that the cap IS coplanar at full insertion, so the
    # cap is anchored to the fixed end-box floor instead and the 1 cm slot band
    # [z_top, +31] is core coolant water.
    #
    # This only bites for z_top < HALF_PLATE_Z, i.e. f < CLAD_EXT/ROD_TRAVEL
    # (1/60 ~ 0.0167). At and above that the cap sits on the blade top exactly
    # as it always did, and the water band is zero-measure.
    #
    # No lower-side counterpart is needed: blade_z_bot is always >= -HALF_Z
    # (asserted above), so the blade never reaches the lower end-box/water.
    #
    # ── Blade top clad + moving cap (commit 17, 2026-07-31) ────────────────
    # A BLADE_TOP_CLAD (1 cm) aluminum clad rides directly on the B4C, and the
    # ENDBOX_HEIGHT (14 cm) end_box_homog cap rides on the CLAD, not the B4C.
    # Both fill the full slot footprint and both clip at CORE_TOP:
    #   f = 0.0   B4C [-30,+30]  clad [+30,+31]  cap [+31,+45]
    #   f = 0.5   B4C [  0,+60]  clad [+60,+61]  cap [+61,+75]
    #   f = 1.0   B4C [+30,+90]  clad clipped    cap clipped
    # Clad is partially clipped for f > 59/60, gone at f = 1.0; the cap is
    # fully gone for f >= 59/60 (cap bottom z_top + 1 reaches CORE_TOP).
    #
    # The Option B max(z_top, HALF_PLATE_Z) cap-floor guard (A4) is DROPPED,
    # not kept as a dead branch: the cap bottom is now z_top + BLADE_TOP_CLAD
    # = HALF_Z + f*ROD_TRAVEL + CLAD_EXT = 31 + 60f >= 31 for all f >= 0, with
    # equality at f = 0 — the coplanarity the guard protected now falls out of
    # the arithmetic (BLADE_TOP_CLAD == CLAD_EXT is exactly the 1 cm the A4
    # water band used to fill). The assert below keeps the invariant loud if
    # BLADE_TOP_CLAD is ever decoupled from CLAD_EXT and shrunk.
    if z_top < CORE_TOP:
        blade_clad_top = openmc.ZPlane(
            z0=min(z_top + BLADE_TOP_CLAD, CORE_TOP))
        cells.append(openmc.Cell(
            name=f'ctrl{elem_id}_blade_clad_slot_b', fill=aluminum,
            region=slot_b & +blade_z_top & -blade_clad_top))
        cells.append(openmc.Cell(
            name=f'ctrl{elem_id}_blade_clad_slot_t', fill=aluminum,
            region=slot_t & +blade_z_top & -blade_clad_top))

        if blade_clad_top.z0 < CORE_TOP:
            cap_bot_z = blade_clad_top.z0   # cap rides on the clad
            assert cap_bot_z >= HALF_PLATE_Z, (
                f"ctrl{elem_id}: cap bottom {cap_bot_z:.2f} < HALF_PLATE_Z "
                f"{HALF_PLATE_Z} — BLADE_TOP_CLAD no longer reaches the "
                f"end-box floor at full insertion (decoupled from CLAD_EXT?)")
            # Cap top is always >= ENDBOX_ABOVE_TOP, so the water above the
            # cap never encroaches on the end-box band [+31,+45].
            assert cap_bot_z + ENDBOX_HEIGHT >= ENDBOX_ABOVE_TOP, (
                f"ctrl{elem_id}: cap top {cap_bot_z + ENDBOX_HEIGHT:.2f} < "
                f"ENDBOX_ABOVE_TOP {ENDBOX_ABOVE_TOP} — cap would not clear "
                f"the end-box band")
            # Reuse blade_clad_top as the cap floor — one plane, no coincident
            # duplicate surface.
            blade_cap_top = openmc.ZPlane(
                z0=min(cap_bot_z + ENDBOX_HEIGHT, CORE_TOP))
            cells.append(openmc.Cell(
                name=f'ctrl{elem_id}_blade_cap_slot_b', fill=end_box_homog,
                region=slot_b & +blade_clad_top & -blade_cap_top))
            cells.append(openmc.Cell(
                name=f'ctrl{elem_id}_blade_cap_slot_t', fill=end_box_homog,
                region=slot_t & +blade_clad_top & -blade_cap_top))
            if blade_cap_top.z0 < CORE_TOP:
                cells.append(openmc.Cell(
                    name=f'ctrl{elem_id}_water_above_cap_slot_b', fill=water,
                    region=slot_b & +blade_cap_top & -_z_model_top))
                cells.append(openmc.Cell(
                    name=f'ctrl{elem_id}_water_above_cap_slot_t', fill=water,
                    region=slot_t & +blade_cap_top & -_z_model_top))

    # ── 17-plate fuel follower (active zone only) ───────────────────────────

    plate_bot_surfs = []
    plate_top_surfs = []

    for i in range(N_CTRL_FUEL_PLATES):
        # Standard 0.127/0.219 pitch, same as the standard fuel element.
        plate_bot = y_fuel_start + i * CTRL_PLATE_PITCH
        plate_top = plate_bot + PLATE_THICK_INNER

        plate_bot_s = openmc.YPlane(y0=plate_bot)
        plate_top_s = openmc.YPlane(y0=plate_top)
        plate_bot_surfs.append(plate_bot_s)
        plate_top_surfs.append(plate_top_s)

        meat_b = openmc.YPlane(y0=plate_bot + CLAD_THICK_INNER)
        meat_t = openmc.YPlane(y0=plate_top - CLAD_THICK_INNER)
        # The meat y-band lies inside the follower stack [-2.8315, +2.8315];
        # the absorber slots sit at |y| in [3.3165, 3.6265], outside it. Meat and
        # absorber slot are disjoint in y, so the zone cuts below cannot
        # interact with the blade cells or the not_slots complement.
        # meat_y is the y band alone — see the standard-element note on why the
        # zone cells substitute their x bounds rather than intersecting them.
        meat_y  = +meat_b & -meat_t
        meat_xy = meat_y & +meat_left & -meat_right
        meat_region = meat_xy & +meat_zbot & -meat_ztop
        clad_region = (
            +plate_bot_s & -plate_top_s &
            +side_inner_left & -side_inner_right &
            plate_z &
            ~meat_region
        )
        if zoned:
            # One cell per (x, z) zone, and one MATERIAL per cell — follower
            # plate i gets its own, shared with nothing. clad_region still
            # subtracts the FULL meat_region, so the zone cells tile exactly
            # what the single meat cell occupied.
            for j in range(N_X_ZONES):
                x_lo, x_hi = zone_x_bounds(j, meat_left, meat_right)
                for k in range(N_AXIAL_ZONES):
                    z_lo, z_hi = zone_z_bounds(k)
                    cells.append(openmc.Cell(
                        name=f'ctrl{elem_id}_meat_{i}_x{j}_z{k}',
                        fill=make_zoned_fuel(element_id, i, j, k,
                                             MEAT_ZONE_VOLUME_PER_PLATE),
                        region=meat_y & +x_lo & -x_hi & +z_lo & -z_hi))
        else:
            cells.append(openmc.Cell(
                name=f'ctrl{elem_id}_meat_{i}', fill=fuel, region=meat_region))
        cells.append(openmc.Cell(
            name=f'ctrl{elem_id}_clad_{i}', fill=clad, region=clad_region))

    # Water channels (active zone only)
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_chan_bot_half', fill=water_core,
        region=(+bot_slider_top & -plate_bot_surfs[0] &
                +side_inner_left & -side_inner_right & plate_z)))
    for i in range(N_CTRL_FUEL_PLATES - 1):
        cells.append(openmc.Cell(
            name=f'ctrl{elem_id}_chan_{i}', fill=water_core,
            region=(+plate_top_surfs[i] & -plate_bot_surfs[i + 1] &
                    +side_inner_left & -side_inner_right & plate_z)))
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_chan_top_half', fill=water_core,
        region=(+plate_top_surfs[-1] & -top_slider_bot &
                +side_inner_left & -side_inner_right & plate_z)))

    # Side plates (active zone only)
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_side_left', fill=aluminum,
        region=(+elem_left & -side_inner_left &
                +elem_front & -elem_back & plate_z)))
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_side_right', fill=aluminum,
        region=(+side_inner_right & -elem_right &
                +elem_front & -elem_back & plate_z)))

    # Inter-element water gaps (active zone only)
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_gap_xleft', fill=water_core,
        region=(+pitch_left & -elem_left &
                +pitch_front & -pitch_back & plate_z)))
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_gap_xright', fill=water_core,
        region=(+elem_right & -pitch_right &
                +pitch_front & -pitch_back & plate_z)))
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_gap_yfront', fill=water_core,
        region=(+elem_left & -elem_right &
                +pitch_front & -elem_front & plate_z)))
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_gap_yback', fill=water_core,
        region=(+elem_left & -elem_right &
                +elem_back & -pitch_back & plate_z)))

    # ── Axial regions above/below active fuel ───────────────────────────────
    # Upper end-box/water still exclude the absorber-slot footprint (handled above,
    # since the blade can reach into them). Lower end-box/water need NO such
    # exclusion: the blade never enters z<-HALF_Z (asserted above), so that
    # band is uniform material straight through — no reserved gap. End-box is
    # one solid full-pitch homogenized block — no inter-element water gap
    # subdivision (the end-box material is already a homogenized Al/water
    # mixture, so a physical gap slice within it is not meaningful).
    full_pitch   = +pitch_left & -pitch_right & +pitch_front & -pitch_back
    not_slots = ~slot_b & ~slot_t   # complement of both absorber slot footprints

    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_upper_endbox', fill=end_box_homog,
        region=full_pitch & +_z_plate_top & -_z_endbox_above & not_slots))
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_upper_water', fill=water,
        region=full_pitch & +_z_endbox_above & -_z_model_top & not_slots))
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_lower_endbox', fill=end_box_homog,
        region=full_pitch & +_z_endbox_below & -_z_plate_bot))
    cells.append(openmc.Cell(
        name=f'ctrl{elem_id}_lower_water', fill=water,
        region=full_pitch & +_z_model_bot & -_z_endbox_below))

    return openmc.Universe(name=f'ctrl_fuel_elem_{elem_id}', cells=cells)


# =============================================================================
# FLUX TRAP
# =============================================================================

def make_flux_trap():
    """
    Flux trap: aluminum block with a central cylindrical water hole (water_core
    at 316.8 K, the same core coolant water used throughout the core), matching
    the reference MCNP model, which models the hole as a ZCylinder rather than the
    originally-commented square.

    The aluminum block is FT_BLOCK_X x FT_BLOCK_Y = 7.6 x 8.0 cm inside the
    7.7 x 8.1 cm pitch, leaving the same 1 mm water gap as the fuel elements.
    The axial end-box (homogenized water/Al) region above and below stays one
    solid FULL-PITCH block — no gap subdivision there.

    Cylinder: radius FT_HOLE_RADIUS = 2.820 cm (area-equivalent to the 50 mm
    square hole), centered at element origin (x=0, y=0).
    The cylinder is axially unbounded within the plate height (plate_z clips it).
    Aluminum fills the annular region between the cylinder and the pitch envelope.
    """
    pitch_left  = openmc.XPlane(x0=-PITCH_X / 2.0)
    pitch_right = openmc.XPlane(x0= PITCH_X / 2.0)
    pitch_front = openmc.YPlane(y0=-PITCH_Y / 2.0)
    pitch_back  = openmc.YPlane(y0= PITCH_Y / 2.0)

    block_left  = openmc.XPlane(x0=-FT_BLOCK_X / 2.0)
    block_right = openmc.XPlane(x0= FT_BLOCK_X / 2.0)
    block_front = openmc.YPlane(y0=-FT_BLOCK_Y / 2.0)
    block_back  = openmc.YPlane(y0= FT_BLOCK_Y / 2.0)

    hole_cyl = openmc.ZCylinder(x0=0.0, y0=0.0, r=FT_HOLE_RADIUS)

    plate_z = +_z_plate_bot & -_z_plate_top   # [−31, +31]

    cells = []

    # Cylindrical water hole — core coolant water at 316.8 K
    cells.append(openmc.Cell(
        name='flux_trap_water_hole',
        fill=water_core,
        region=-hole_cyl & plate_z
    ))
    # Aluminum block: 7.6 x 8.0 block envelope minus the cylinder, plate height
    cells.append(openmc.Cell(
        name='flux_trap_aluminum_block',
        fill=aluminum,
        region=(+block_left & -block_right & +block_front & -block_back
                & +hole_cyl & plate_z)
    ))

    # Inter-element water gaps around the block — the 1 mm pitch gap, same as
    # the fuel elements. Core coolant (316.8 K), matching both the flux-trap
    # hole and the neighbouring fuel-element gaps.
    cells.append(openmc.Cell(
        name='flux_trap_gap_xleft', fill=water_core,
        region=(+pitch_left & -block_left &
                +pitch_front & -pitch_back & plate_z)))
    cells.append(openmc.Cell(
        name='flux_trap_gap_xright', fill=water_core,
        region=(+block_right & -pitch_right &
                +pitch_front & -pitch_back & plate_z)))
    cells.append(openmc.Cell(
        name='flux_trap_gap_yfront', fill=water_core,
        region=(+block_left & -block_right &
                +pitch_front & -block_front & plate_z)))
    cells.append(openmc.Cell(
        name='flux_trap_gap_yback', fill=water_core,
        region=(+block_left & -block_right &
                +block_back & -pitch_back & plate_z)))

    # Axial regions above/below active fuel. End-box is one solid full-pitch
    # homogenized block — no inter-element water gap subdivision (the
    # end-box material is already a homogenized Al/water mixture, so a
    # physical gap slice within it is not meaningful); water-beyond stays
    # full pitch.
    full_pitch = +pitch_left & -pitch_right & +pitch_front & -pitch_back

    cells.append(openmc.Cell(
        name='flux_trap_upper_endbox',
        fill=end_box_homog,
        region=full_pitch & +_z_plate_top & -_z_endbox_above
    ))
    cells.append(openmc.Cell(
        name='flux_trap_upper_water',
        fill=water,
        region=full_pitch & +_z_endbox_above & -_z_model_top
    ))
    cells.append(openmc.Cell(
        name='flux_trap_lower_endbox',
        fill=end_box_homog,
        region=full_pitch & +_z_endbox_below & -_z_plate_bot
    ))
    cells.append(openmc.Cell(
        name='flux_trap_lower_water',
        fill=water,
        region=full_pitch & +_z_model_bot & -_z_endbox_below
    ))

    return openmc.Universe(name='flux_trap_universe', cells=cells)


# =============================================================================
# WATER AND GRAPHITE FILL UNIVERSES
# =============================================================================

# Water universe: fully unbounded — bulk water fills whatever space the parent
# lattice boundary provides (used for the outer ring and top/bottom water rows).
water_cell = openmc.Cell(name='water_fill', fill=water)
water_univ = openmc.Universe(name='water_universe', cells=[water_cell])

# Graphite reflector universe.
#
# In-plane: each reflector position holds a discrete REFL_BLOCK_X x REFL_BLOCK_Y
# = 7.6 x 8.0 cm graphite block inside the 7.7 x 8.1 cm pitch, leaving the same
# 1 mm water gap as the fuel elements. Adjacent reflector positions therefore
# form a row of separate blocks with a water channel between them, NOT the
# continuous graphite wall this used to build.
#
# Axially: graphite occupies the full block z-range [-31, +31]. Above
# and below, the end-box (homogenized water/Al) region is one solid
# full-pitch block, same as the fuel elements — no gap subdivision.
# Water-beyond stays full pitch, mirroring the fuel element end-box + water
# stack so the reflector height matches the core height.
def make_graphite_element():
    """Graphite reflector element: discrete block + water gaps, solid end-box axially."""
    pitch_left  = openmc.XPlane(x0=-PITCH_X / 2.0)
    pitch_right = openmc.XPlane(x0= PITCH_X / 2.0)
    pitch_front = openmc.YPlane(y0=-PITCH_Y / 2.0)
    pitch_back  = openmc.YPlane(y0= PITCH_Y / 2.0)

    block_left  = openmc.XPlane(x0=-REFL_BLOCK_X / 2.0)
    block_right = openmc.XPlane(x0= REFL_BLOCK_X / 2.0)
    block_front = openmc.YPlane(y0=-REFL_BLOCK_Y / 2.0)
    block_back  = openmc.YPlane(y0= REFL_BLOCK_Y / 2.0)

    plate_z    = +_z_plate_bot & -_z_plate_top   # [−31, +31]
    full_pitch = +pitch_left & -pitch_right & +pitch_front & -pitch_back

    # Inter-block gap water: water_core (316.8 K), NOT the 294 K bulk pool
    # water. MODELING CHOICE, not a measured condition — graphite generates no
    # heat, so coolant in these channels would in reality sit nearer the inlet
    # temperature than the 316.8 K core-average value. water_core is chosen for
    # consistency with the adjacent fuel-element gaps, which these channels are
    # hydraulically continuous with.
    gap_water = water_core

    # End-box is one solid full-pitch homogenized block — no inter-element
    # water gap subdivision (the end-box material is already a homogenized
    # Al/water mixture, so a physical gap slice within it is not meaningful).
    cells = [
        openmc.Cell(
            name='graphite_block',
            fill=graphite,
            region=(+block_left & -block_right &
                    +block_front & -block_back & plate_z),
        ),
        openmc.Cell(
            name='graphite_gap_xleft', fill=gap_water,
            region=(+pitch_left & -block_left &
                    +pitch_front & -pitch_back & plate_z),
        ),
        openmc.Cell(
            name='graphite_gap_xright', fill=gap_water,
            region=(+block_right & -pitch_right &
                    +pitch_front & -pitch_back & plate_z),
        ),
        openmc.Cell(
            name='graphite_gap_yfront', fill=gap_water,
            region=(+block_left & -block_right &
                    +pitch_front & -block_front & plate_z),
        ),
        openmc.Cell(
            name='graphite_gap_yback', fill=gap_water,
            region=(+block_left & -block_right &
                    +block_back & -pitch_back & plate_z),
        ),
        openmc.Cell(
            name='graphite_upper_endbox',
            fill=end_box_homog,
            region=full_pitch & +_z_plate_top & -_z_endbox_above,  # +31 → +45 cm
        ),
        openmc.Cell(
            name='graphite_upper_water',
            fill=water,
            region=full_pitch & +_z_endbox_above & -_z_model_top,  # +45 → +90 cm
        ),
        openmc.Cell(
            name='graphite_lower_endbox',
            fill=end_box_homog,
            region=full_pitch & +_z_endbox_below & -_z_plate_bot,  # −45 → −31 cm
        ),
        openmc.Cell(
            name='graphite_lower_water',
            fill=water,
            region=full_pitch & +_z_model_bot & -_z_endbox_below,  # −90 → −45 cm
        ),
    ]

    return openmc.Universe(name='graphite_universe', cells=cells)


graphite_univ = make_graphite_element()


# =============================================================================
# CORE LATTICE EXTENT AND POOL WATER BOUNDARY
#
# The lattice holds the 30 CORE positions PLUS the 12 GRAPHITE REFLECTOR
# positions — 6 (x) x 7 (y) = 42. It used to span 8 x 9 including a one-cell
# water ring; that ring is gone and the surrounding water is handled
# externally, as an explicit pool box.
#
# CORE_HALF_X / CORE_HALF_Y below are LATTICE half-extents. The name predates
# this terminology and is kept because the 38.5 cm pool arithmetic and
# tallies.py both key off it; read it as "lattice envelope", not "core".
#
# The ring could not simply be thickened: POOL_WATER_THICK is 5 * PITCH_X
# exactly in x, but 38.5 / 8.1 = 4.75 in y, so no whole number of lattice rings
# reproduces it. Hence the explicit bounding box.
# =============================================================================

N_LAT_X = 6   # lattice positions in x (core + reflector)              [MCNP]
N_LAT_Y = 7   # lattice positions in y (core + reflector)              [MCNP]

CORE_HALF_X = N_LAT_X / 2.0 * PITCH_X     # 23.100 cm                  [DERIVED]
CORE_HALF_Y = N_LAT_Y / 2.0 * PITCH_Y     # 28.350 cm                  [DERIVED]

# Pool water on all four lateral sides of the core.
POOL_WATER_THICK = 38.5   # cm                                         [MCNP]

POOL_HALF_X = CORE_HALF_X + POOL_WATER_THICK   # 61.600 cm             [DERIVED]
POOL_HALF_Y = CORE_HALF_Y + POOL_WATER_THICK   # 66.850 cm             [DERIVED]

# Lateral extent tripwires. Tolerance, not ==: CORE_HALF_Y runs through
# 3.5 * 8.1, which is not exact in binary floating point.
assert POOL_WATER_THICK > 0, "pool water thickness must be positive"
assert abs(2 * POOL_HALF_X - 123.2) < 1e-9, \
    f"model x-extent is {2 * POOL_HALF_X}, expected 123.2 " \
    f"(6 x {PITCH_X} + 2 x {POOL_WATER_THICK})"
assert abs(2 * POOL_HALF_Y - 133.7) < 1e-9, \
    f"model y-extent is {2 * POOL_HALF_Y}, expected 133.7 " \
    f"(7 x {PITCH_Y} + 2 x {POOL_WATER_THICK})"
# REMOVED 2026-08-24 — the two pool-face asserts were vacuous on BOTH axes.
# They read
#     assert abs((POOL_HALF_X - CORE_HALF_X) - POOL_WATER_THICK) < 1e-12
#     assert abs((POOL_HALF_Y - CORE_HALF_Y) - POOL_WATER_THICK) < 1e-12
# and POOL_HALF_X/Y are DEFINED as CORE_HALF_X/Y + POOL_WATER_THICK on the two
# lines immediately above, so each recomputed its own definition and subtracted
# it from itself. Unlike the lower-water assert further up — which is vacuous
# only under an enforced symmetry assumption and is therefore kept — these had
# no such standing: no input makes either one fail.
#
# The real lateral constraints are the two extent tripwires retained above,
# which compare against the independent 123.2 / 133.7 literals, and the
# POOL_WATER_THICK > 0 check. Recorded rather than silently dropped so this is
# not read as a lost check.


# =============================================================================
# CORE MAP — position labels for the 6 (x) x 7 (y) LATTICE
#
# This map covers the LATTICE, not the core. The core is 5 x 6 = 30 positions
# (23 standard + 5 control + 2 flux traps); the two all-graphite rows are
# reflector, not core, and bring the lattice to 6 x 7 = 42.
#
# Token grid mirroring lattice_universes below, one token per CORE position:
#   'S' standard fuel   'C' control fuel   'F' flux trap   'G' graphite
# There is no 'W' token: since B4 the water ring is not part of the lattice.
# A token-for-token assert in build_core_geometry() ties this grid to the
# lattice literal, so the two can never drift apart by hand.
#
# TECDOC-643 A-2 Table 1: "Grid Plate 8x9 Positions", "Active Core Geometry
# 5x6 Positions", 23 standard + 5 control elements, and two irradiation
# channels — "1 at Core Center" and "1 at Core Edge" (A-2 §1: one water-filled
# flux trap near the center of the core, another near an edge). The literal
# below places them at D4 (center) and A6 (edge), which is the benchmark
# configuration. Table 1's 8 x 9 is the GRID PLATE row, which counts the
# surrounding water ring; its 5 x 6 is the ACTIVE CORE row. The 42 positions
# modelled here are the 6 x 7 lattice = 30 core + 12 graphite reflector.
#
# Columns A-F run left to right in +x; rows 1-7 run top to bottom, so ROW 1 IS
# THE +y EDGE — matching the array order of CORE_MAP and lattice_universes, so
# a reader comparing the two never has to mentally flip anything.
#
# The letter/number convention is THIS PROJECT'S OWN — TECDOC-643 A-2 specifies
# no element labeling scheme, so the convention itself carries no [TECDOC] tag.
# Labels feed no dimension, surface, or cell region; they name depletion
# materials and nothing else.
# =============================================================================

CORE_MAP_COLS = 'ABCDEF'
CORE_MAP = [
    ['G', 'G', 'G', 'G', 'G', 'G'],
    ['S', 'S', 'C', 'S', 'S', 'S'],
    ['S', 'S', 'S', 'S', 'C', 'S'],
    ['S', 'C', 'S', 'F', 'S', 'S'],
    ['S', 'S', 'S', 'S', 'C', 'S'],
    ['F', 'S', 'C', 'S', 'S', 'S'],
    ['G', 'G', 'G', 'G', 'G', 'G'],
]


def core_map_label(row, col):
    """Label for lattice position (row, col), or None outside the 6 x 7.

    Re-indexed in B4 when the surrounding water ring left the lattice: the map
    is now lattice positions only, so row/col are 0-based over the lattice
    rather than offset by the ring. The LABELS THEMSELVES ARE UNCHANGED
    (A1..F7) — they name depletion materials, and a silent shift would
    re-point every zoned material at a different element.
    """
    if 0 <= row < N_LAT_Y and 0 <= col < N_LAT_X:
        return f'{CORE_MAP_COLS[col]}{row + 1}'
    return None


def core_map_labels(token):
    """Row-major list of position labels for a CORE_MAP token ('S', 'C', ...).

    Row-major order matches the order the lattice literal assigns std_elems[i]
    and ctrl_elems[i], so labels line up with those indices element for element.
    """
    return [core_map_label(i, j)
            for i, row in enumerate(CORE_MAP)
            for j, t in enumerate(row) if t == token]


STD_ELEMENT_IDS  = core_map_labels('S')   # 23 standard element positions
CTRL_ELEMENT_IDS = core_map_labels('C')   # 5 control element positions

assert len(CORE_MAP) == N_LAT_Y, \
    f"CORE_MAP has {len(CORE_MAP)} rows, expected N_LAT_Y={N_LAT_Y}"
assert all(len(r) == N_LAT_X for r in CORE_MAP), \
    f"every CORE_MAP row must have N_LAT_X={N_LAT_X} entries"
assert not any('W' in r for r in CORE_MAP), \
    "CORE_MAP must hold lattice positions only — the water ring left the lattice in B4"
assert len(STD_ELEMENT_IDS) == 23, \
    f"CORE_MAP has {len(STD_ELEMENT_IDS)} 'S' positions, expected 23"
assert len(CTRL_ELEMENT_IDS) == 5, \
    f"CORE_MAP has {len(CTRL_ELEMENT_IDS)} 'C' positions, expected 5"
assert sum(r.count('F') for r in CORE_MAP) == 2, \
    "CORE_MAP must have exactly 2 flux traps (A-2 Table 1: 1 center, 1 edge)"
assert sum(r.count('G') for r in CORE_MAP) == 12, \
    "CORE_MAP must have exactly 12 graphite reflector positions"
assert sum(len(r) for r in CORE_MAP) == 42 == N_LAT_X * N_LAT_Y, \
    "lattice positions must total 42 = 6 x 7 (30 core + 12 graphite reflector)"
assert None not in STD_ELEMENT_IDS + CTRL_ELEMENT_IDS, \
    "a fuelled position fell outside the labelled 6 x 7 lattice"
assert len(set(STD_ELEMENT_IDS + CTRL_ELEMENT_IDS)) == 28, \
    "duplicate core-map labels"

# Label pin. These strings name depletion materials, so the B4 re-indexing of
# core_map_label() had to leave them byte-identical. Any future re-indexing that
# shifts them silently re-points every zoned material at a different element.
assert STD_ELEMENT_IDS == [
    'A2', 'B2', 'D2', 'E2', 'F2', 'A3', 'B3', 'C3', 'D3', 'F3',
    'A4', 'C4', 'E4', 'F4', 'A5', 'B5', 'C5', 'D5', 'F5',
    'B6', 'D6', 'E6', 'F6'], \
    f"standard element labels moved: {STD_ELEMENT_IDS}"
assert CTRL_ELEMENT_IDS == ['C2', 'E3', 'B4', 'E5', 'C6'], \
    f"control element labels moved: {CTRL_ELEMENT_IDS}"
assert core_map_labels('F') == ['D4', 'A6'], \
    f"flux trap positions moved: {core_map_labels('F')}"


# =============================================================================
# CORE LATTICE — TECDOC-643 Fig. 2.1 (LEU panel)
# =============================================================================

def build_core_geometry(withdrawn_fraction=1.0, depletion_zoning=False):
    """Build the full-core openmc.Geometry for a blade WITHDRAWAL fraction f.

    f = 0.0 → blades fully INSERTED  (absorber spans z=[-30, +30])
    f = 1.0 → blades fully WITHDRAWN (absorber spans z=[+30, +90])

    depletion_zoning=True splits every fuel meat cell on an (x, z) grid into
    N_X_ZONES x N_AXIAL_ZONES cells, one depletable material PER CELL
    (614 x N_X_ZONES x N_AXIAL_ZONES materials — the same formula as the cell
    count, since the two are 1:1 — all starting from the identical base fuel
    composition). At the 2 x 10 default that is 12,280 meat cells and 12,280
    materials. Structural scaffolding for a later depletion study — it
    configures nothing about depletion itself.
    Default False: the Phase One fresh-core cross-validation baseline must not
    move. With it off the model is byte-for-byte what it has always been.

    This is the single construction path used by core.build_model() and all
    run/ drivers. The 6 x 7 lattice sits inside a water-filled pool box whose
    lateral faces are POOL_WATER_THICK = 38.5 cm outboard of the core envelope;
    the vacuum boundary is there, not at the lattice edge. Vacuum at
    CORE_BOTTOM=-90 / CORE_TOP=+90 accommodates the full axial stack
    (water/end-box/clad/meat/clad/end-box/water); the withdrawn (f=1) blade top
    coincides exactly with CORE_TOP, so there is no water cap above it.
    """
    std_elems  = [make_standard_fuel_element(
                      i, element_id=STD_ELEMENT_IDS[i], zoned=depletion_zoning)
                  for i in range(23)]
    ctrl_elems = [make_control_fuel_element(
                      100 + i, withdrawn_fraction=withdrawn_fraction,
                      element_id=CTRL_ELEMENT_IDS[i], zoned=depletion_zoning)
                  for i in range(5)]

    W = water_univ
    G = graphite_univ
    S = std_elems
    C = ctrl_elems
    F = make_flux_trap()

    lattice_universes = [
        [G,     G,     G,     G,     G,     G    ],
        [S[0],  S[1],  C[0],  S[2],  S[3],  S[4] ],
        [S[5],  S[6],  S[7],  S[8],  C[1],  S[9] ],
        [S[10], C[2],  S[11], F,     S[12], S[13]],
        [S[14], S[15], S[16], S[17], C[3],  S[18]],
        [F,     S[19], C[4],  S[20], S[21], S[22]],
        [G,     G,     G,     G,     G,     G    ],
    ]

    # CORE_MAP must mirror the lattice literal token for token. This is the
    # guard that stops the two from being reconciled by hand: the labels that
    # name depletion materials are only meaningful if the map matches what is
    # actually built.
    _token_of = {W: 'W', G: 'G', F: 'F'}
    _token_of.update({u: 'S' for u in S})
    _token_of.update({u: 'C' for u in C})
    for _i, (_map_row, _lat_row) in enumerate(zip(CORE_MAP, lattice_universes)):
        _built = [_token_of[u] for u in _lat_row]
        assert _built == _map_row, (
            f"CORE_MAP row {_i} {_map_row} disagrees with the lattice "
            f"literal {_built}")
    assert len(CORE_MAP) == len(lattice_universes), \
        "CORE_MAP and lattice_universes have different row counts"

    core_lattice = openmc.RectLattice(name='core_lattice')
    core_lattice.pitch      = (PITCH_X, PITCH_Y)
    core_lattice.lower_left = (-CORE_HALF_X, -CORE_HALF_Y)
    core_lattice.universes  = lattice_universes
    # Guard against edge-case lattice lookups just outside the universe array
    # (floating-point roundoff at the boundary planes) — fill with bulk water
    # instead of losing the particle. Kept as roundoff insurance even though the
    # lattice is now exactly coincident with the core envelope planes.
    core_lattice.outer      = water_univ

    # Core envelope — TRANSMISSIVE. These used to be the vacuum boundary; since
    # B4 the vacuum sits POOL_WATER_THICK further out, on the pool faces.
    core_left   = openmc.XPlane(x0=-CORE_HALF_X)
    core_right  = openmc.XPlane(x0= CORE_HALF_X)
    core_front  = openmc.YPlane(y0=-CORE_HALF_Y)
    core_back   = openmc.YPlane(y0= CORE_HALF_Y)

    # Pool box lateral faces — vacuum, 38.5 cm outboard of the core envelope.
    pool_left   = openmc.XPlane(x0=-POOL_HALF_X, boundary_type='vacuum')
    pool_right  = openmc.XPlane(x0= POOL_HALF_X, boundary_type='vacuum')
    pool_front  = openmc.YPlane(y0=-POOL_HALF_Y, boundary_type='vacuum')
    pool_back   = openmc.YPlane(y0= POOL_HALF_Y, boundary_type='vacuum')

    core_bottom = openmc.ZPlane(z0=CORE_BOTTOM,     boundary_type='vacuum')
    core_top    = openmc.ZPlane(z0=CORE_TOP,        boundary_type='vacuum')

    core_box_xy = +core_left & -core_right & +core_front & -core_back
    axial_span  = +core_bottom & -core_top

    core_cell = openmc.Cell(name='core_cell', fill=core_lattice,
                            region=core_box_xy & axial_span)

    # Pool water — the 294 K BULK water, not the 316.8 K core coolant.
    # Written as the pool box minus the core box rather than as four slabs:
    # the complement is watertight at the corners, which is exactly where a
    # hand-written slab decomposition leaves undefined space that passes an
    # overlap check and still leaks particles.
    pool_cell = openmc.Cell(
        name='pool_water', fill=water,
        region=(+pool_left & -pool_right & +pool_front & -pool_back &
                axial_span & ~core_box_xy))

    root_universe = openmc.Universe(name='root', cells=[core_cell, pool_cell])
    return openmc.Geometry(root_universe)


# Module-level default geometry (blades fully inserted) — kept for direct
# `python geometry.py` debug use; drivers should call build_core_geometry().
geometry = build_core_geometry(withdrawn_fraction=0.0)


def _lattice_center(row, col):
    """Global (x, y) of the centre of CORE_MAP cell (row, col).

    Mirrors the lattice lower_left used in build_core_geometry(); row 0 is the
    +y edge, matching CORE_MAP's array order.
    """
    return (-CORE_HALF_X + (col + 0.5) * PITCH_X,
            -CORE_HALF_Y + (N_LAT_Y - 1 - row + 0.5) * PITCH_Y)


def _first_position(tok):
    """(row, col) of the first CORE_MAP cell carrying `tok`, row-major.

    Used to anchor the point checks by TOKEN rather than by hardcoded index, so
    they survive a core-map edit instead of silently probing the wrong element.
    """
    for i, row in enumerate(CORE_MAP):
        for j, t in enumerate(row):
            if t == tok:
                return i, j
    raise AssertionError(f"CORE_MAP has no '{tok}' position")


def _material_at(geom, x, y, z):
    """Name of the material filling the innermost cell containing (x, y, z)."""
    for obj in reversed(geom.find((x, y, z))):
        if isinstance(obj, openmc.Cell):
            if obj.fill is None:
                return None
            return getattr(obj.fill, 'name', None)
    return None


def _run_point_checks(geom, f):
    """Point-containment assertions for the B1 axial stack.

    Probes a standard element's first-plate meat centreline up the axial stack:
    meat -> unfueled clad extension -> end-box -> water. Any of these coming
    back as the wrong material (or None) means a band is mis-clipped or has
    been left as undefined space.
    """
    row, col = _first_position('S')          # first standard element (A2)
    ex, ey = _lattice_center(row, col)

    # Centreline of plate 0's meat, derived exactly as the builder lays it out.
    meat0_y = -(STD_STACK_HEIGHT / 2.0) + CLAD_THICK_OUTER + MEAT_THICK / 2.0
    px, py = ex, ey + meat0_y

    z_mid_clad_ext = HALF_Z + CLAD_EXT / 2.0            # +30.5
    z_mid_endbox   = HALF_PLATE_Z + ENDBOX_HEIGHT / 2.0  # +38.0
    z_mid_water    = ENDBOX_ABOVE_TOP + POOL_WATER_AXIAL / 2.0  # +67.5

    # Exact material identity, not substrings: the clad extension band must be
    # cladding and nothing else, which is the entire point of B1.
    expected = [
        (( px,  py,            0.0), fuel,          'active meat'),
        (( px,  py,  z_mid_clad_ext), clad,         'upper clad extension'),
        (( px,  py, -z_mid_clad_ext), clad,         'lower clad extension'),
        (( px,  py,  z_mid_endbox), end_box_homog,  'upper end-box'),
        (( px,  py, -z_mid_endbox), end_box_homog,  'lower end-box'),
        (( px,  py,  z_mid_water),  water,          'upper water'),
        (( px,  py, -z_mid_water),  water,          'lower water'),
    ]
    for point, want, label in expected:
        got = _material_at(geom, *point)
        assert got is not None, \
            f"f={f}: {label} at {point} is UNDEFINED SPACE (no material)"
        assert got == want.name, \
            f"f={f}: {label} at {point} is '{got}', expected '{want.name}'"

    # B1 — the clad extension band is not only clad. The coolant channels and
    # the inter-element water gaps were clipped to +/-30 alongside the plates
    # and had to be extended to +/-31 with them. Pinned explicitly at z = 30.5,
    # both signs: had any of these been missed, the band would read as
    # UNDEFINED SPACE rather than water, and an overlap check would not catch it.
    chan0_y  = -(STD_STACK_HEIGHT / 2.0) + PLATE_THICK_OUTER + WATER_CHAN_THICK / 2.0
    gap_dx   = (ELEM_X + PITCH_X) / 4.0     # centre of the x gap, 3.825
    gap_dy   = (ELEM_Y + PITCH_Y) / 4.0     # centre of the y gap, 4.025
    band_probes = [
        (ex,          ey + chan0_y, 'coolant channel 0'),
        (ex - gap_dx, ey,           'inter-element gap, -x'),
        (ex + gap_dx, ey,           'inter-element gap, +x'),
        (ex,          ey - gap_dy,  'inter-element gap, -y'),
        (ex,          ey + gap_dy,  'inter-element gap, +y'),
    ]
    for bx, by, label in band_probes:
        for bz in (0.0, z_mid_clad_ext, -z_mid_clad_ext):
            got = _material_at(geom, bx, by, bz)
            assert got is not None, \
                f"f={f}: {label} at z={bz} is UNDEFINED SPACE (no material)"
            assert got == water_core.name, (
                f"f={f}: {label} at z={bz} is '{got}', expected "
                f"'{water_core.name}'")

    # B2 — the flux-trap and graphite blocks are 7.6 x 8.0 inside the pitch, so
    # a probe just inside the block edge must be solid and a probe in the pitch
    # gap must be water. Sampling at z=0 and at the top of the clad band.
    for tok, solid in (('F', aluminum), ('G', graphite)):
        row_b, col_b = _first_position(tok)
        bx, by = _lattice_center(row_b, col_b)
        blk_x = FT_BLOCK_X if tok == 'F' else REFL_BLOCK_X
        # Just inside the block edge in x (clear of the flux-trap hole).
        inside = _material_at(geom, bx + blk_x / 2.0 - 0.02, by, 0.0)
        assert inside == solid.name, \
            f"f={f}: '{tok}' block interior is '{inside}', expected '{solid.name}'"
        # Middle of the pitch gap in x.
        gap = _material_at(geom, bx + PITCH_X / 2.0 - GAP_X / 2.0, by, 0.0)
        assert gap == water_core.name, \
            f"f={f}: '{tok}' inter-block gap is '{gap}', expected '{water_core.name}'"
        # Same block, up in the clad-extension band — blocks run to +/-31 too.
        top = _material_at(geom, bx + blk_x / 2.0 - 0.02, by, z_mid_clad_ext)
        assert top == solid.name, \
            f"f={f}: '{tok}' block at z={z_mid_clad_ext} is '{top}', " \
            f"expected '{solid.name}'"
        if tok == 'F':
            # A1 — the 2.820 cm water hole: coolant at the hole centre, and
            # still aluminum in the annulus between hole edge and block edge
            # (the x margin is the thin one, 0.98 cm at 2.820).
            centre = _material_at(geom, bx, by, 0.0)
            assert centre == water_core.name, \
                f"f={f}: flux-trap hole centre is '{centre}', " \
                f"expected '{water_core.name}'"
            annulus_x = bx + (FT_HOLE_RADIUS + FT_BLOCK_X / 2.0) / 2.0
            ann = _material_at(geom, annulus_x, by, 0.0)
            assert ann == aluminum.name, \
                f"f={f}: flux-trap annulus at x-offset " \
                f"{annulus_x - bx:.3f} is '{ann}', expected '{aluminum.name}'"

    # B4 — pool water. Probed 30 cm outboard of the core edge on each lateral
    # face (inside the 38.5 cm pool), plus both diagonal corners, which is where
    # a slab decomposition of the pool would have left undefined space.
    pool_probes = [
        ( CORE_HALF_X + 30.0, 0.0,                 '+x face'),
        (-CORE_HALF_X - 30.0, 0.0,                 '−x face'),
        (0.0,                  CORE_HALF_Y + 30.0, '+y face'),
        (0.0,                 -CORE_HALF_Y - 30.0, '−y face'),
        ( CORE_HALF_X + 30.0,  CORE_HALF_Y + 30.0, '+x+y corner'),
        (-CORE_HALF_X - 30.0, -CORE_HALF_Y - 30.0, '−x−y corner'),
    ]
    for wx, wy, label in pool_probes:
        for wz in (0.0, ENDBOX_ABOVE_TOP + 1.0, CORE_BOTTOM + 1.0):
            got = _material_at(geom, wx, wy, wz)
            assert got is not None, \
                f"f={f}: pool water {label} at z={wz} is UNDEFINED SPACE"
            assert got == water.name, (
                f"f={f}: pool water {label} at z={wz} is '{got}', expected "
                f"'{water.name}' (294 K bulk water, not the core coolant)")

    print(f"  point checks (f={f}): axial stack "
          f"meat/clad-ext/end-box/water, 7.6x8.0 blocks + gaps, "
          f"and pool water all resolve")


def _run_blade_slot_checks(geom, f):
    """Point-containment assertions down a control element's absorber slot.

    Blade assembly stack (commit 17): B4C, then BLADE_TOP_CLAD (1 cm) aluminum
    riding on it, then the 14 cm cap riding on the clad, then water — clad and
    cap clipped at CORE_TOP. At full insertion the slot must read:
        blade -> 1 cm aluminum clad [+30,+31] -> 14 cm cap (coplanar at
        [+31,+45]) -> water.
    The clad band is the old A4 Option B water band — the fig3 defect — and
    must now return aluminum, not water.
    """
    row, col = _first_position('C')          # first control element (C2)
    ex, ey = _lattice_center(row, col)

    # Centreline of the lower absorber slot, built outward exactly as the
    # element builder lays the end block out.
    slot_c = -(CTRL_FUEL_STACK_HALF + CTRL_FEEDER_CHANNEL + CTRL_AL_PLATE_THICK
               + CTRL_BLADE_WATER + ABSORBER_THICK / 2.0)
    px, py = ex, ey + slot_c

    z_bot = -HALF_Z + f * ROD_TRAVEL
    z_top = z_bot + BLADE_LENGTH

    def want_at(z):
        """Material the slot should carry at height z (mirrors the builder)."""
        if z < z_bot:
            return water_core          # slot below the blade, down to −31
        if z < z_top:
            return b4c                 # the blade itself
        if z >= CORE_TOP:
            return None
        clad_top = min(z_top + BLADE_TOP_CLAD, CORE_TOP)
        if z < clad_top:
            return aluminum            # blade top clad riding on the B4C
        if z < min(clad_top + ENDBOX_HEIGHT, CORE_TOP):
            return end_box_homog       # the 14 cm cap riding on the clad
        return water                   # bulk water above the cap

    probes = [-31.0 + CLAD_EXT / 2.0, -HALF_Z / 2.0, 0.0, HALF_Z / 2.0,
              HALF_Z + CLAD_EXT / 2.0, HALF_PLATE_Z + ENDBOX_HEIGHT / 2.0,
              ENDBOX_ABOVE_TOP - 0.5, ENDBOX_ABOVE_TOP + 0.5, 67.5, 89.5]
    for z in probes:
        want = want_at(z)
        if want is None:
            continue
        got = _material_at(geom, px, py, z)
        assert got is not None, \
            f"f={f}: absorber slot at z={z} is UNDEFINED SPACE (no material)"
        assert got == want.name, \
            f"f={f}: absorber slot at z={z} is '{got}', expected '{want.name}'"

    # Commit 17 explicitly: at full insertion the band [+30,+31] must be the
    # aluminum top clad (the fig3 defect — previously water), the cap must be
    # coplanar with the surrounding end-boxes, water above.
    if f == 0.0:
        assert _material_at(geom, px, py, HALF_Z + BLADE_TOP_CLAD / 2.0) == aluminum.name, \
            "blade top clad: slot band [+30,+31] at f=0 must be aluminum (fig3 defect)"
        assert _material_at(geom, px, py, ENDBOX_ABOVE_TOP - 0.5) == end_box_homog.name, \
            "cap must reach ENDBOX_ABOVE_TOP at f=0 (coplanar with end-boxes)"
        assert _material_at(geom, px, py, ENDBOX_ABOVE_TOP + 0.5) == water.name, \
            "cap must stop at ENDBOX_ABOVE_TOP at f=0, water above"

    # At full withdrawal it is the B4C that touches the model top: absorber at
    # z=89.5 with nothing above it (clad and cap clipped away entirely).
    if f == 1.0:
        assert _material_at(geom, px, py, CORE_TOP - 0.5) == b4c.name, \
            "f=1: B4C must reach CORE_TOP (z=89.5 probe) — clad clips, B4C does not"

    # B3 — the blade is ABSORBER_WIDTH (6.630) wide inside an ACTIVE_STACK_X
    # (6.640) slot. At the blade's own mid-height the centreline must be
    # absorber and the side film must be coolant.
    z_mid_blade = (z_bot + z_top) / 2.0
    on_blade = _material_at(geom, px, py, z_mid_blade)
    assert on_blade == b4c.name, \
        f"f={f}: blade centreline at z={z_mid_blade} is '{on_blade}', " \
        f"expected '{b4c.name}'"
    film_x = ex + ACTIVE_STACK_X / 2.0 - ABSORBER_SIDE_WATER / 2.0
    film = _material_at(geom, film_x, py, z_mid_blade)
    assert film is not None, \
        f"f={f}: blade side-water film at x={film_x} is UNDEFINED SPACE"
    assert film == water_core.name, \
        f"f={f}: blade side-water film is '{film}', expected '{water_core.name}'"

    # A3 — full end-block y-stack walk, wall to fuel stack (bottom end block).
    # Each layer is probed at its own centre; a wrong material here means the
    # 0.1305 / 0.1275 layer budget is mis-built, not merely mis-stated. The
    # absorber layer is probed at the blade's mid-height (it must contain the
    # blade); the structural/water layers at z=0 (they span the plate height).
    end_block_layers = [
        (CTRL_OUTER_OFFSET,    water_core, 'offset water'),
        (CTRL_AL_PLATE_THICK,  aluminum,   'outer guide plate'),
        (CTRL_BLADE_WATER,     water_core, 'blade water, outer'),
        (ABSORBER_THICK,       b4c,        'absorber slot'),
        (CTRL_BLADE_WATER,     water_core, 'blade water, inner'),
        (CTRL_AL_PLATE_THICK,  aluminum,   'inner guide plate'),
        (CTRL_FEEDER_CHANNEL,  water_core, 'feeder channel'),
    ]
    y_walk = ey - ELEM_Y / 2.0                 # element wall (bottom)
    for thick, want, label in end_block_layers:
        y_c = y_walk + thick / 2.0
        z_probe = z_mid_blade if want is b4c else 0.0
        got = _material_at(geom, ex, y_c, z_probe)
        assert got is not None, \
            f"f={f}: end-block '{label}' at y={y_c:.4f} is UNDEFINED SPACE"
        assert got == want.name, (
            f"f={f}: end-block '{label}' at y={y_c:.4f} is '{got}', "
            f"expected '{want.name}'")
        y_walk += thick
    assert abs(y_walk - (ey - CTRL_FUEL_STACK_HALF)) < 1e-9, \
        "end-block layer walk does not land on the fuel stack edge"

    print(f"  slot checks  (f={f}): absorber slot stack resolves correctly "
          f"(blade z=[{z_bot:.1f}, {z_top:.1f}], "
          f"{ABSORBER_SIDE_WATER:.4f} cm side film present)")


if __name__ == '__main__':
    # Provenance first, so every verification result below is stamped with the
    # code that produced it. This is the H3 gap from the Phase 1 audit: a stale
    # verification read used to be invisible in the output and had to be
    # inferred afterwards. A '-dirty' suffix here means the asserts and overlap
    # checks below did NOT run against a committed tree.
    from settings import format_provenance as _fmt_prov
    print("Run provenance:")
    print(_fmt_prov())
    print()

    geometry.export_to_xml()
    print("geometry.xml written successfully.\n")
    print(f"Lattice pitch:        {PITCH_X} x {PITCH_Y} cm")
    print(f"Element envelope:     {ELEM_X} x {ELEM_Y} x {ELEM_Z} cm")
    print(f"Active fuel meat z:   [{-HALF_Z}, {+HALF_Z}] cm ({MEAT_HEIGHT} cm)")
    print(f"Plate / clad z:       [{-HALF_PLATE_Z}, {+HALF_PLATE_Z}] cm "
          f"({PLATE_HEIGHT} cm, {CLAD_EXT} cm unfueled clad each end)")
    print(f"End-box above:        [{+HALF_PLATE_Z}, {ENDBOX_ABOVE_TOP}] cm "
          f"({ENDBOX_HEIGHT} cm)")
    print(f"End-box below:        [{ENDBOX_BELOW_BOT}, {-HALF_PLATE_Z}] cm "
          f"({ENDBOX_HEIGHT} cm)")
    print(f"Core z-bounds:        [{CORE_BOTTOM}, {CORE_TOP}] cm (vacuum)")
    print(f"Core positions:       {N_LAT_X} (x) x {N_LAT_Y} (y) = "
          f"{N_LAT_X * N_LAT_Y}")
    print(f"Core envelope:        x +/-{CORE_HALF_X}, y +/-{CORE_HALF_Y} cm "
          f"(transmissive)")
    print(f"Pool water:           {POOL_WATER_THICK} cm all four sides, "
          f"294 K bulk water")
    print(f"Model extent:         x {2 * POOL_HALF_X} x y {2 * POOL_HALF_Y} x "
          f"z {CORE_TOP - CORE_BOTTOM} cm (vacuum at pool faces)")
    print(f"Axial stack sum:      2 x ({POOL_WATER_AXIAL} + {ENDBOX_HEIGHT} + "
          f"{CLAD_EXT} + {HALF_Z}) = {_AXIAL_STACK_SUM} cm")
    print(f"Active-zone gap width: GAP_X={GAP_X:.4f} cm, GAP_Y={GAP_Y:.4f} cm "
          f"(end-box regions are solid full-pitch homogenized blocks — no "
          f"gap subdivision there)")
    print(f"\nBlade model:")
    print(f"  BLADE_LENGTH = {BLADE_LENGTH} cm (fixed)")
    print(f"  ROD_TRAVEL   = {ROD_TRAVEL} cm")
    for f_chk in [0.0, 0.5, 1.0]:
        z_b = -HALF_Z + f_chk * ROD_TRAVEL
        z_t = z_b + BLADE_LENGTH
        ok = z_b >= CORE_BOTTOM and z_t <= CORE_TOP
        print(f"  f={f_chk:.1f}: blade z=[{z_b:.1f}, {z_t:.1f}]  "
              f"within [{CORE_BOTTOM},{CORE_TOP}]: {ok}")

    print(f"\nControl element layout:")
    print(f"  Fuel stack half-width (CTRL_FUEL_STACK_HALF): "
          f"{CTRL_FUEL_STACK_HALF:.6f} cm")
    print(f"  Fuel stack:       [{-CTRL_FUEL_STACK_HALF:.6f}, "
          f"{CTRL_FUEL_STACK_HALF:.6f}] cm "
          f"({2*CTRL_FUEL_STACK_HALF:.6f} cm, 17 plates @ pitch "
          f"{CTRL_PLATE_PITCH:.6f} cm)")
    print(f"  End block (each): {CTRL_END_BLOCK:.6f} cm "
          f"(feeder {CTRL_FEEDER_CHANNEL:.5f} + guide {CTRL_AL_PLATE_THICK:.5f} "
          f"+ blade-water {CTRL_BLADE_WATER:.5f} + blade {ABSORBER_THICK:.5f} "
          f"+ blade-water {CTRL_BLADE_WATER:.5f} + guide {CTRL_AL_PLATE_THICK:.5f} "
          f"+ offset {CTRL_OUTER_OFFSET:.5f})")

    end_block_layer_sum = (CTRL_FEEDER_CHANNEL + CTRL_AL_PLATE_THICK
                           + CTRL_BLADE_WATER + ABSORBER_THICK
                           + CTRL_BLADE_WATER + CTRL_AL_PLATE_THICK
                           + CTRL_OUTER_OFFSET)
    print(f"  End-block layer sum: {end_block_layer_sum:.6f} cm "
          f"(should be {CTRL_END_BLOCK:.6f})")
    print(f"  Total (2 ends + fuel stack): "
          f"{2*end_block_layer_sum + 2*CTRL_FUEL_STACK_HALF:.6f} cm (should be {ELEM_Y})")

    assert abs(end_block_layer_sum - CTRL_END_BLOCK) < 1e-9, \
        "control end-block layers do not sum to CTRL_END_BLOCK"
    assert abs(2*end_block_layer_sum + 2*CTRL_FUEL_STACK_HALF - ELEM_Y) < 1e-9, \
        "control element total height != ELEM_Y"

    # Geometry overlap check
    import tempfile
    from materials import materials as _materials
    from settings import settings as _settings

    _settings.particles = 200
    _settings.batches   = 2
    _settings.inactive  = 1

    debug_model = openmc.Model(
        geometry=geometry, materials=_materials, settings=_settings
    )
    with tempfile.TemporaryDirectory() as _debug_dir:
        debug_model.run(geometry_debug=True, cwd=_debug_dir)
    print("\nOverlap check (f=0.0) passed: no cell overlaps detected.")
    _run_point_checks(geometry, 0.0)
    _run_blade_slot_checks(geometry, 0.0)

    # f=0.5 is the ordinary mid-travel case: blade at [0,+60], cap riding
    # directly on the blade top, no A4 coolant band.
    geometry_f05 = build_core_geometry(withdrawn_fraction=0.5)
    debug_model_f05 = openmc.Model(
        geometry=geometry_f05, materials=_materials, settings=_settings
    )
    with tempfile.TemporaryDirectory() as _debug_dir_f05:
        debug_model_f05.run(geometry_debug=True, cwd=_debug_dir_f05)
    print("Overlap check (f=0.5) passed: no cell overlaps detected.")
    _run_point_checks(geometry_f05, 0.5)
    _run_blade_slot_checks(geometry_f05, 0.5)

    # f=0.99 exercises the clipping band commit 17 introduced: for
    # f > 59/60 (~0.98333) the cap is fully clipped away but the clad is only
    # PARTIALLY clipped (here [89.4, 90]). Nothing else in the test set
    # produces a partially-clipped clad with no cap above it — this is where
    # undefined space would hide.
    geometry_f099 = build_core_geometry(withdrawn_fraction=0.99)
    debug_model_f099 = openmc.Model(
        geometry=geometry_f099, materials=_materials, settings=_settings
    )
    with tempfile.TemporaryDirectory() as _debug_dir_f099:
        debug_model_f099.run(geometry_debug=True, cwd=_debug_dir_f099)
    print("Overlap check (f=0.99) passed: no cell overlaps detected "
          "(partially-clipped clad, cap fully clipped).")
    _run_point_checks(geometry_f099, 0.99)
    _run_blade_slot_checks(geometry_f099, 0.99)

    # f=1.0 exercises the degenerate case introduced by the axial resize:
    # blade_z_top == CORE_TOP exactly (three coincident ZPlane objects at the
    # withdrawn blade top / upper_water boundary / global vacuum boundary).
    geometry_f1 = build_core_geometry(withdrawn_fraction=1.0)
    debug_model_f1 = openmc.Model(
        geometry=geometry_f1, materials=_materials, settings=_settings
    )
    with tempfile.TemporaryDirectory() as _debug_dir_f1:
        debug_model_f1.run(geometry_debug=True, cwd=_debug_dir_f1)
    print("Overlap check (f=1.0) passed: no cell overlaps detected "
          "(blade top coincident with CORE_TOP vacuum boundary).")
    _run_point_checks(geometry_f1, 1.0)
    _run_blade_slot_checks(geometry_f1, 1.0)
