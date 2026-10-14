"""
materials.py
------------
Material definitions for the IAEA TECDOC-643 Appendix A-2
Generic 10 MW LEU Research Reactor Core (Argonne design).

Reference:
    IAEA-TECDOC-643, "Research Reactor Core Conversion Guidebook,
    Volume 2: Analysis (Appendices A-F)," IAEA, Vienna, 1992.
    Appendix A-2: Generic 10 MW Reactor — Argonne National Laboratory.

Units:
    All densities in g/cm^3.
    All enrichments in weight percent unless noted.
"""

import openmc

# Natural carbon (C0, ZAID 6000) vs. an explicit C12/C13 isotopic split.
#
# GATES BOTH CARBON-BEARING MATERIALS: b4c and graphite. It used to gate only
# graphite, with b4c hardcoded to the split — so the flag's name promised more
# than it delivered and the two materials could disagree about what carbon is.
#
# The reference MCNP model specifies natural carbon (card 6000) for both. This
# flag therefore selects a LIBRARY BASIS, not a physics choice:
#   False -> C12/C13. ENDF/B-VIII.0 has no natural-carbon evaluation, so the
#            split is the only way to represent card 6000 there. It is an
#            OpenMC-side method artifact with no MCNP counterpart, and the
#            cross-validation spreadsheet carries those rows as N/A.
#   True  -> C0. The exact counterpart of card 6000, available on a matched
#            ENDF/B-VII.0 library.
#
# PROJECT_BIBLE §6 makes the choice conditional on which library is active.
# ONLY THE VIII.0 BRANCH HAS BEEN IMPLEMENTED AND RUN; the C0 branch is written
# but has never been exercised, because no VII.0 library is configured here.
# Switching this to True is not a one-line change in practice — USE_AL_SAB has
# the same library dependency and must move with it.
#
# The abundance convention below (0.9893 / 0.0107) is COSMETIC for validation
# purposes: the comparison against the reference is on TOTAL carbon
# (2.005592e-02 atom/b-cm), which both branches reproduce exactly. The split
# only redistributes that total between two nuclides.
USE_NATURAL_CARBON = False

# Al metal thermal scattering S(a,b) (c_Al27). Enabled per the 2026-07-20
# meeting decision ("We will add Al-27 SaB libraries"). The active
# ENDF/B-VIII.0 library ($OPENMC_CROSS_SECTIONS) provides c_Al27; ENDF/B-VII.0
# does not, so this must revert to False if the library is switched back.
USE_AL_SAB = True

# =============================================================================
# FUEL MATERIAL
# LEU U3Si2-Al fuel, 19.75 w/o enriched uranium
# Atom densities (atom/b-cm) taken directly from the reference MCNP model.
# NO thermal scattering table on the fuel (it has no mt card for it).
# =============================================================================

fuel = openmc.Material(name='LEU_U3Si2_Al_fuel')
fuel.temperature = 332.1
fuel.add_nuclide('U235', 2.251800e-03)   # atom/b-cm
fuel.add_nuclide('U238', 9.034100e-03)
fuel.add_nuclide('Al27', 3.256300e-02)
fuel.add_nuclide('Si28', 6.938766e-03)
fuel.add_nuclide('Si29', 3.524947e-04)
fuel.add_nuclide('Si30', 2.326390e-04)
fuel.set_density('sum')

# =============================================================================
# CLADDING MATERIAL
# 6061-T6 Aluminum alloy (standard MTR fuel plate cladding)
# Density: 2.70 g/cm3
# =============================================================================

clad = openmc.Material(name='Al_6061_cladding')
clad.temperature = 330.7 
clad.set_density('g/cm3', 2.70)
clad.add_element('Al', 1.00, 'ao')
# Pure aluminum stands in for 6061-T6. [MCNP] — this is NOT an approximation we
# chose independently: the reference MCNP model does the same thing, and the
# resulting Al-27 number densities agree exactly.
#   OpenMC 6.026261E-02  vs  MCNP 6.026260E-02  atoms/b-cm  —  MATCH, 0.0000%
# Al metal S(a,b) (c_Al27) added per the 2026-07-20 meeting decision, gated on
# USE_AL_SAB (ENDF/B-VIII.0 provides c_Al27; VII.0 does not).
if USE_AL_SAB:
    clad.add_s_alpha_beta('c_Al27')

# =============================================================================
# COOLANT / MODERATOR
# Two water materials per reference MCNP model cross-section assignments:
#   - Outer pool water: 0.9975 g/cm³ at 294 K   (H1 = 6.66909e-2 atom/b-cm)
#     Only the water OUTSIDE the core lattice footprint (the reflector-side
#     boundary ring) uses this — it is not core coolant.
#   - Core water:       0.9909 g/cm³ at 316.8 K (H1 = 6.625423e-2 atom/b-cm)
#     Every water feature INSIDE the core (inter-element gaps, plate/channel
#     water, end-box water, graphite channels, flux-trap holes, etc.) uses
#     this — it is the coolant water throughout the core, not a flux-trap-
#     specific material.
# (Mass densities print ~0.02% lower than the nominal values because the
# O-16-only basis has a slightly lower molar mass than natural oxygen; the
# H1/O16 atom densities above are the authoritative quantities.)
# =============================================================================

# Water is H-1 + O-16 ONLY (no H-2/O-17/O-18), matching the reference MCNP
# model's O-16 basis.
water = openmc.Material(name='light_water_294K')
water.temperature = 294.0
water.add_nuclide('H1', 6.66909e-02)
water.add_nuclide('O16', 6.66909e-02 / 2.0)
water.set_density('sum')
water.add_s_alpha_beta('c_H_in_H2O')

water_core = openmc.Material(name='light_water_core_316K')
water_core.temperature = 316.8
water_core.add_nuclide('H1', 6.625423e-02)
water_core.add_nuclide('O16', 6.625423e-02 / 2.0)
water_core.set_density('sum')
water_core.add_s_alpha_beta('c_H_in_H2O')

# =============================================================================
# CONTROL BLADE MATERIAL
# B4C absorber, defined by atom densities from the reference MCNP model [MCNP].
# set_density('sum') -> 2.000 g/cm3 [DERIVED].
# The card is not natural stoichiometric B4C. Both departures are as given by
# the MCNP card (ratios of its atom densities), not chosen here:
#   B:C  = 4.45                (not 4.0)
#   B-10 = 21.5 at% of boron   (not natural 19.9%)
# =============================================================================

b4c = openmc.Material(name='B4C control absorber')
b4c.temperature = 294.0
b4c.add_nuclide('B10', 1.914973e-02)
b4c.add_nuclide('B11', 7.010412e-02)
# Carbon gated on USE_NATURAL_CARBON, matching graphite below. Total carbon is
# 2.005592e-02 atom/b-cm either way; the flag only chooses how it is
# represented. The False branch runs the same two calls in the same order this
# block always did, so the exported material is unchanged on VIII.0.
if USE_NATURAL_CARBON:
    b4c.add_nuclide('C0', 2.005592e-02)
else:
    b4c.add_nuclide('C12', 2.005592e-02 * 0.9893)
    b4c.add_nuclide('C13', 2.005592e-02 * 0.0107)
b4c.set_density('sum')

# 2026-08-24: an orphaned second C12/C13 pair sat here, commented out, outside
# any branch and after set_density('sum'), duplicating the live lines above. It
# was dead, but it was the only other place the abundance convention appeared,
# so it read as a live alternative rather than as residue. Deleted.
#
# HOW IT GOT HERE — b4c's carbon was switched twice, and this was left behind
# by the FIRST switch, not the second:
#   4bfd5d5  split -> C0.  The displaced C12/C13 lines were commented out
#                          rather than deleted. That is this pair.
#   290bb77  C0 -> split.  Reinstated the split as live code and commented out
#                          C0 instead. It never touched 4bfd5d5's residue, so
#                          the pair survived as a duplicate of the new live
#                          lines and stopped being an alternative to anything.
# Neither commit message mentions carbon (they read "added some bugs" and
# "more bugs"), so the history above is reconstructed from the diffs and is
# recorded here because the log does not carry it.

# NO S(a,b) on B4C carbon (the reference MCNP model has no mt card for it).


# =============================================================================
# REFLECTOR MATERIAL
# Graphite reflector blocks surrounding the core.
#
# Density: 8.724000E-02 atoms/b-cm total carbon, taken DIRECTLY from the
# reference MCNP model card m00005 [MCNP]:
#
#   m00005     $     294.0    Graphite reflector
#          6000    8.724000E-02
#
# TECDOC-643 specifies no graphite density; neither model assigns a mass
# density to this material — the atom density above is the specification on
# both sides. The 1.7400 g/cm3 previously here was this same atom density
# back-converted and rounded (8.724000E-02 -> 1.74000 g/cm3 using OpenMC's
# isotopic masses, M_eff = 12.01112 from 0.988922 x 12.000000 +
# 0.011078 x 13.00335484); it reproduced the atom density to 0.0003%, inside
# the 0.010% tolerance. Specifying in atom/b-cm removes even that round-off.
#
# Card 6000 is natural carbon, unsplit. OpenMC's add_element('C') expands to
# C-12/C-13 (no natural-carbon evaluation exists in ENDF/B-VIII.0), so the
# C-12/C-13 split below is an OpenMC-side method artifact with no MCNP
# counterpart — the spreadsheet carries those rows as N/A on the MCNP side.
# For a VII.0 matched-library run the exact counterpart is the C0 nuclide
# (USE_NATURAL_CARBON branch below).
# =============================================================================

graphite = openmc.Material(name='graphite_reflector')
graphite.temperature = 294.0                      # card m00005: $ 294.0
if USE_NATURAL_CARBON:
    graphite.add_nuclide('C0', 1.0)
else:
    graphite.add_element('C', 1.0)
graphite.set_density('atom/b-cm', 8.724000E-02)
graphite.add_s_alpha_beta('c_Graphite')

# =============================================================================
# STRUCTURAL ALUMINUM
# Pure aluminum for grid plates, side plates, core structure
# Density: 2.70 g/cm3
# =============================================================================

aluminum = openmc.Material(name='aluminum_structure')
aluminum.temperature = 330.7
aluminum.set_density('g/cm3', 2.70)
aluminum.add_element('Al', 1.0, 'ao')
# Al metal S(a,b) (c_Al27) added per the 2026-07-20 meeting decision, gated on
# USE_AL_SAB.
if USE_AL_SAB:
    aluminum.add_s_alpha_beta('c_Al27')

# =============================================================================
# END-BOX HOMOGENIZED MATERIAL
# Homogenized Al/water end box. Atom densities from the reference MCNP model,
# card m00004, at 316.8 K [MCNP]. set_density('sum') -> 1.41806 g/cm3 [DERIVED].
# The mix is exactly 25 v/o Al / 75 v/o CORE water (water_core, not the 294 K
# pool water) [DERIVED]:
#   Al27 1.506565e-02 / 6.026260e-02 (Al metal)   = 0.2500
#   H1   4.969068e-02 / 6.625423e-02 (core water) = 0.7500
# Fills the 14 cm end box above and below the fuel plates, |z| = 31 -> 45 cm
# (ENDBOX_HEIGHT in geometry.py).
# TECDOC-643 A-2 gives 15 cm; the reference MCNP model uses 14 cm [MCNP].
# =============================================================================

end_box_homog = openmc.Material(name='end_box_homogenized')
end_box_homog.temperature = 316.8                      # card m00004: $ 316.8
end_box_homog.add_nuclide('Al27', 1.506565e-02)        # card 13027
end_box_homog.add_nuclide('H1',   4.969068e-02)        # card 1001
end_box_homog.add_nuclide('O16',  2.484534e-02)        # card 8016
end_box_homog.set_density('sum')                       # -> 1.41806 g/cm3
end_box_homog.add_s_alpha_beta('c_H_in_H2O')
# Al metal S(a,b) (c_Al27) on the aluminum component, added per the 2026-07-20
# meeting decision, gated on USE_AL_SAB.
if USE_AL_SAB:
    end_box_homog.add_s_alpha_beta('c_Al27')

# Water component is H-1 + O-16 ONLY (no H-2/O-17/O-18) per the reference
# MCNP model.

# =============================================================================
# DEPLETION ZONING — per-element, per-(x,z)-zone fuel meat materials
#
# RESOLUTION: N_X_ZONES x N_AXIAL_ZONES = 2 x 10 = 20 zones per element.
# x is the meat WIDTH direction (MEAT_WIDTH, 6.3 cm); z is axial. The plate
# stacking direction y is NOT subdivided.
#
# [MCNP — Kyle confirmed 2026-08-12] The reference MCNP model subdivides EACH
# FUEL PLATE 2 x 10. This is transcribed authority, not an inference, and it
# replaces the [ASSUMED] 8 x 20 recorded below.
#
# --- SUBDIVISION AND GRANULARITY — BOTH NOW CONFIRMED ------------------------
# These are two different claims. Kyle answered both on 2026-08-12:
#
#   SUBDIVISION  per plate, 2 x 10   ->  12,280 cells  [MCNP — Kyle 2026-08-12]
#   MATERIALS    per plate, per zone ->  12,280 mats   [MCNP — Kyle 2026-08-12]
#
# CELLS AND MATERIALS ARE NOW 1:1 — one material per meat cell, each owning
# exactly one plate's zone volume (MEAT_ZONE_VOLUME_PER_PLATE, 0.9639 cm^3).
# No plate shares a material with any other. The std/ctrl volume distinction
# that existed under element-shared materials is GONE: there is no longer a
# 23-plate or 17-plate multiplier anywhere in the material path.
#
# --- THE 560 SCHEME WAS NOT A BUG. READ THIS BEFORE "FIXING" ANYTHING. -------
# Until 2026-08-12 all 23 plates of a standard element (17 of a control element)
# shared ONE material per (x, z) zone: 28 x 20 = 560 materials over the same
# 12,280 cells. That was VALID OpenMC. OpenMC does NOT require a 1:1
# cell-to-material mapping; many cells sharing one material is normal, fully
# supported, and still used elsewhere in this model (clad, water, graphite...).
#
# It was a MODELING CHOICE, and this is what it meant physically: a depletion
# solve would have seen ONE flux and reaction-rate spectrum averaged over all
# plates of an element in a given (x, z) zone, so plate-to-plate burnup
# gradients WITHIN an element could not develop. Per-plate materials let them.
#
# We changed it to MATCH THE REFERENCE MODEL, not to correct a defect. Anyone
# reading the diff should not record this as a bug fix.
#
# --- SUPERSEDED, IN ORDER — the history is the point -------------------------
# Three values have been recorded here. None of the earlier ones are deleted,
# because how we arrived at the current one is itself the record.
#
# (1) 5 axial zones, no x subdivision.  [MCNP-VISUAL, SUPERSEDED]
#     Implemented until 2026-08-12, on a visual reading of a zx slice plot of
#     the reference MCNP model. NEVER confirmed by Kyle and, as a reading,
#     REMAINS UNCONFIRMED. Retained as a record of what was inferred, NOT as a
#     live claim:
#       * 5 axial depletion zones over the active height  [MCNP-VISUAL, SUPERSEDED]
#       * uniform zone height (60/5 = 12.0 cm)   [DERIVED, MCNP-VISUAL, SUPERSEDED]
#       * all plates in an element share a zone material
#                                                [MCNP-VISUAL, inferred, RETIRED]
#       * one unique material per element per zone        [MCNP-VISUAL, SUPERSEDED]
#     Note the third: plate-sharing is the ONE inference still implemented, and
#     it is still unconfirmed. Superseding the zone COUNT does not retire it.
#     Kyle's 2026-08-12 answer superseded the COUNT twice over and STILL has not
#     retired it — see the granularity block above.
#       ^^ RETIRED 2026-08-12. The paragraph above is kept verbatim as the
#          record of what was believed and for how long. What retired it was a
#          SEPARATE answer from Kyle on the same day — per-plate materials —
#          not either of the zone-count supersessions, exactly as the paragraph
#          warned. Cells and materials are now 1:1; see the granularity block.
#
#     NO [MCNP-VISUAL] CLAIM REMAINS LIVE IN THE ZONING SCHEME. Every
#     [MCNP-VISUAL] string still present in this file sits inside a dated
#     SUPERSEDED/RETIRED record above, or is the negative statement in (2)
#     below about what 8 x 20 did NOT carry. None is an active assertion about
#     the reference model. Checked by enumeration on 2026-08-12, not assumed.
#
# (2) 8 x 20.  [ASSUMED, SUPERSEDED]
#     Implemented 2026-08-12, superseded the same day by Kyle's answer. It was
#     OUR OWN resolution choice and was NEVER a claim about the reference model:
#     not transcribed from it, not read off any plot of it, not stated in
#     TECDOC-643. It carried no [MCNP], [MCNP-VISUAL] or [TECDOC] authority and
#     was tagged [ASSUMED] throughout. Recorded so that a reader who finds
#     8 x 20 in the history knows it was a placeholder resolution, not evidence.
#
# (3) 2 x 10.  [MCNP — Kyle confirmed 2026-08-12]  <-- LIVE, implemented below.
#
# --- A FOURTH, SEPARATE NUMBER — do not conflate ----------------------------
# [TECDOC-643 App. A-2, Sec. 3.1 "Burnup Results", p. 31] states that the
# three-dimensional burnup calculations for the HEU and LEU cores were first
# performed with the REBUS-3 fuel cycle analysis code using HALF-CORE SYMMETRY
# and EIGHT AXIAL DEPLETION ZONES ABOVE THE CORE MIDPLANE, on ENDF/B-IV cross
# sections generated with EPRI-CELL in five energy groups.
#
# READ THAT CAREFULLY. It is eight zones per HALF core, under half-core
# symmetry — i.e. SIXTEEN zones over the full active height if mirrored, NOT
# eight over the active height. Every axial number on record, stated together so
# the wrong two are never compared:
#       TECDOC/REBUS-3:   8 per half  ==  16 full-height equivalent  [TECDOC]
#       superseded (1):   5 full-height              [MCNP-VISUAL, SUPERSEDED]
#       superseded (2):  20 full-height                  [ASSUMED, SUPERSEDED]
#       this model:      10 full-height        [MCNP — Kyle confirmed 2026-08-12]
# REBUS-3 is moreover a DIFFUSION-based fuel cycle code on FIVE-GROUP ENDF/B-IV
# data with a HALF-CORE model: a different method, a different library and a
# different geometry from this continuous-energy full-core Monte Carlo model.
# It is a data point, not a target, and NOT the basis for N_AXIAL_ZONES = 10.
# Note it also disagrees with the [MCNP] value Kyle confirmed — 16 vs 10 — which
# is unsurprising for a different code and model, and is not a discrepancy to
# reconcile.
#
# COINCIDENCE WARNING — now HISTORICAL, kept because it explains itself.
# This warning was written when N_X_ZONES was 8 and collided numerically with
# TECDOC's eight-per-half. That collision is gone: N_X_ZONES is now 2. The
# warning stays because the hazard it names has not gone away — TECDOC's 8 is an
# AXIAL, half-core, diffusion-model count, and it is a separate number from all
# three of ours. If a future resolution reintroduces an 8 anywhere in this file,
# it will not derive from TECDOC either.
#
# N_X_ZONES and N_AXIAL_ZONES below are the ONLY places the zone counts are
# written. Zone boundaries, volumes and cell counts are all derived from them
# in geometry.py — changing either is a one-line edit and breaks no assertion.
# They are now [MCNP]-backed, so changing them is no longer a free choice:
# it would be a departure from the confirmed reference model.
#
# COST, before you raise either count. READ ALL THREE PARAGRAPHS — the headline
# table cannot see the largest cost of the per-plate change.
#
# CELLS AND MATERIALS NOW SHARE ONE FORMULA: both are
#     614 plates x N_X_ZONES x N_AXIAL_ZONES.
# Raising the resolution therefore costs on BOTH axes simultaneously — more
# cells for transport AND more materials for the depletion solve. Under the old
# element-shared scheme materials were 28 x N_X x N_Z and grew 22x more slowly.
# Anyone proposing a resolution study needs to see this before proposing it.
#
# (1) TRANSPORT. A standard element universe holds 460 meat cells at 2 x 10
# against ~110 in the unzoned model.
#
# HOW OPENMC ACTUALLY FINDS CELLS — corrected 2026-08-12. An earlier version of
# this block said OpenMC "searches cells within a universe linearly, so that ~4x
# is paid on EVERY neutron boundary crossing". THAT WAS WRONG AS STATED, and it
# overstated the transport cost of raising the zone counts. Per OpenMC's Theory
# and Methodology documentation:
#   * Finding a cell from a point (Sec. 2.3) IS a linear loop over a universe's
#     cells, recursing into filled universes. That much was right.
#   * A SURFACE CROSSING (Sec. 2.6) — the common case in transport — searches
#     only that surface's NEIGHBOR LIST. The full linear search is the FALLBACK
#     when the neighbor-list search misses.
#   * Neighbor lists (Sec. 2.7) are CELL-BASED as of OpenMC 0.11 and are built
#     dynamically during transport. OpenMC moved off surface-based lists because
#     those degrade when one surface bounds many cells — exactly what zoning
#     creates, since each shared interior zone plane bounds two cells in all 614
#     plates.
#   * Harper, Romano, Forget, Smith, Nucl. Sci. Eng.,
#     doi 10.1080/00295639.2020.1719765
# So per-universe cell count is NOT paid in full on every boundary crossing.
#
# READ THE SCAN COLUMN ACCORDINGLY. geometry_debug locates points from scratch
# to check for overlaps, which is the unmitigated Sec. 2.3 path — precisely what
# neighbor lists bypass in real transport. The scan is therefore a WORST CASE
# and an UPPER BOUND on the geometry-search term, NOT a transport predictor.
# No transport penalty has been measured and none is estimated here; doing that
# requires an eigenvalue run, which has never been performed on this model.
#
# ALL MEASURED 2026-08-12:
#
#          zones    cells     mats   build   geometry.xml  materials.xml  scan
#  unzoned     -      614        1  0.08 s        0.47 MB      0.002 MB   3.4 s
#  2 x 10     20   12,280   12,280  0.68 s        1.86 MB       4.92 MB   8.4 s
#  8 x 20    160   98,240    4,480  1.55 s       12.16 MB       1.79 MB  47.7 s
#  (The 8 x 20 row used ELEMENT-SHARED materials, 28 x N_X x N_Z — which is why
#   it has FEWER materials than the live 2 x 10 row despite 8x the cells. Only
#   the live row is per-plate. Build times are build_model(); scan is the
#   geometry_debug overlap scan at f=0.5.)
#
# materials.xml grew 22x, 0.22 -> 4.92 MB, going from the OLD element-shared
# 2 x 10 scheme (560 materials, 0.22 MB; not a row in the table above) to the
# live per-plate one (12,280), tracking the material count exactly.
# geometry.xml barely moved (1.85 -> 1.86 MB): the cells are unchanged, only the
# material ID each one references. That split is the whole story of this change.
#
# DO NOT EXTRAPOLATE THE SCAN COLUMN LINEARLY. There is a large fixed cost
# (~3.2 s of startup at any size), so the cheap end looks flatter than it is:
# 20x the cells from unzoned to 2 x 10 cost only 1.9x the time. Fit a line
# through those two points and it predicts ~28 s at 8 x 20 — the measured value
# is 47.7 s, 1.7x worse. Cost accelerates once cells dominate the fixed overhead.
#
# (2) THE DEPLETION SOLVE — AND THE SCAN COLUMN ABOVE CANNOT SEE IT AT ALL.
# A depletion solve runs a Bateman/CRAM solve PER DEPLETABLE MATERIAL, every
# timestep. Going element-shared (560) -> per-plate (12,280) multiplied that
# work by 22x while changing the CELL count not at all. Transport is unchanged;
# the overlap scan is unchanged; nothing in the table above moves. The cost is
# real and it is invisible here. Do not read a flat scan time as "this change
# was free" — it was free for transport only, and no depletion solve or
# eigenvalue run has ever been measured on this model.
#
# Do not raise the resolution without measuring BOTH axes.
#
# This module holds NO geometry dimensions: zone height, zone width and meat
# volume are computed in geometry.py, next to the dimensions they derive from,
# and the volume is passed in. Structural scaffolding only — nothing here
# executes or configures depletion.
# =============================================================================

N_X_ZONES     = 2     # [MCNP — Kyle 2026-08-12] meat-width divisions per plate
N_AXIAL_ZONES = 10    # [MCNP — Kyle 2026-08-12] axial divisions per plate

assert isinstance(N_X_ZONES, int) and N_X_ZONES >= 1, \
    "N_X_ZONES must be a positive integer"
assert isinstance(N_AXIAL_ZONES, int) and N_AXIAL_ZONES >= 1, \
    "N_AXIAL_ZONES must be a positive integer"

# (element_id, plate_index, x_index, zone_index) -> Material, in creation order.
_zoned_fuel_registry = {}


def make_zoned_fuel(element_id, plate_index, x_index, zone_index, volume):
    """One depletion material: one (x, z) zone of ONE plate of `element_id`.

    Fills exactly one meat cell. Materials and meat cells are 1:1 — each plate
    carries its own set of N_X_ZONES x N_AXIAL_ZONES materials and shares
    nothing with its neighbours. [MCNP — Kyle confirmed 2026-08-12]

    plate_index 0 = first plate in the stack (y order), as the geometry builders
                    enumerate them.
    x_index     0 = -x edge of the meat (MEAT_LEFT_X), N_X_ZONES-1 = +x edge.
    zone_index  0 = bottom of the active fuel, N_AXIAL_ZONES-1 = top.

    `volume` is computed by the caller from the authoritative geometry
    dimensions; materials.py holds no geometry constants.

    NOTE the asymmetry in the range guards below: x_index and zone_index are
    bounded here because N_X_ZONES and N_AXIAL_ZONES live in this module, but
    plate_index is only checked for non-negativity. Its upper bound differs by
    element type (N_PLATES_STD 23 vs N_CTRL_FUEL_PLATES 17) and both are
    geometry constants, which this module deliberately does not hold — the same
    reason `volume` is passed in rather than computed. The geometry-side loop
    bounds it naturally: plate_index is the builder's own enumeration variable
    and cannot exceed the plate count it is iterating.

    Composition, temperature (332.1 K) and any S(a,b) are inherited unchanged
    from the base `fuel` material via clone(); nothing is re-specified here, so
    all zoned materials start life identical to the fresh-core fuel.

    Repeat calls with the same key return the same material, so building the
    geometry more than once in a process does not duplicate materials.
    """
    if plate_index < 0:
        raise ValueError(f"plate_index {plate_index} is negative")
    if not 0 <= x_index < N_X_ZONES:
        raise ValueError(
            f"x_index {x_index} outside [0, {N_X_ZONES})")
    if not 0 <= zone_index < N_AXIAL_ZONES:
        raise ValueError(
            f"zone_index {zone_index} outside [0, {N_AXIAL_ZONES})")

    key = (element_id, plate_index, x_index, zone_index)
    if key in _zoned_fuel_registry:
        return _zoned_fuel_registry[key]

    m = fuel.clone()
    m.name       = f'fuel_{element_id}_p{plate_index}_x{x_index}_z{zone_index}'
    m.depletable = True
    m.volume     = volume
    _zoned_fuel_registry[key] = m
    return m


def get_zoned_fuels():
    """Every zoned fuel material created so far, in creation order."""
    return list(_zoned_fuel_registry.values())


# =============================================================================
# Collect all materials into a Materials object for export
# =============================================================================

materials = openmc.Materials([
    fuel,
    clad,
    water,
    water_core,
    b4c,
    graphite,
    aluminum,
    end_box_homog,
])

if __name__ == '__main__':
    materials.export_to_xml()
    print("materials.xml written successfully.")
    print("\nMaterial summary:")
    for mat in materials:
        t = f"  T={mat.temperature} K" if mat.temperature is not None else ""
        print(f"  [{mat.id}] {mat.name}  —  {mat.density} g/cm3{t}")