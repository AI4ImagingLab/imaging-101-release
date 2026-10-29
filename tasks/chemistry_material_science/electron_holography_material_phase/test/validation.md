# Provenance and validation evidence

Validated 2026-10-18, Windows CPU. This is a newly cleaned **real-data material
electron-holography task**. It uses an experimentally recorded Fe needle with
YOx inclusions and a separately acquired vacuum hologram. No synthetic arrays
replace the observations, and no actual complex-wave ground truth is available.

## Selected scope and sources

The selected standard method is the Fourier wave-reconstruction stage of
HoloSpy v0.3: find a vacuum carrier, isolate and recenter a smooth reciprocal
sideband, inverse-transform each exposure, then divide object by vacuum.
The associated specimen article is Migunov et al., J. Appl. Phys. 117, 134301,
published version of record 2015-04-01, DOI 10.1063/1.4916609. Crossref's
published title/date and VOR PDF URL were fetched and archived. The task does
**not** claim reproduction of the paper's full multi-condition charge-density
inference. The selected wave-reconstruction mathematics is specified by the
archived HoloSpy documentation and source and derived self-contained in the
README and approach plan.

- Repository: https://github.com/hyperspy/holospy
- Used release v0.3 commit: `fe4a0dae1e34e40215e9b20e5d836c321c2b5e07`.
  GitHub's commit API resolved the remote tag during this session; it is not a
  local tracking revision. The Zenodo archive directory also identifies
  `hyperspy-holospy-fe4a0da`. This historical release is deliberately pinned;
  no claim of using latest main is made.
- Used source/data archive: https://doi.org/10.5281/zenodo.11099344,
  direct https://zenodo.org/api/records/11099344/files/hyperspy/holospy-v0.3.zip/content
  (record file `hyperspy/holospy-v0.3.zip`, published MD5
  `af5881037f878a0e1600ed4c8ec1cf5c`).
- Experimental object: `holospy/data/01_holo_Vbp_130V_0V_bin2_crop.hdf5`;
  vacuum: `holospy/data/00_ref_Vbp_130V_0V_bin2_crop.hdf5`.
- Original metadata identifies author Vadim Migunov, specimen preparation
  Andrew London, microscope FEI Titan G2 60-300 HOLO at ER-C, 300 keV,
  10-second exposure, 130-V biprism, 5-degree tilt. Vacuum General/title is
  explicitly **Vacuum reference hologram**, with distinct original filename
  `00_ref_Vbp_130V_0V.dm3`; object original filename is
  `01_holo_Vbp_130V_0V.dm3`. Both experimental dataset loaders cite the paper.
  Archived documentation explicitly instructs independent vacuum correction.
  This is positive provenance evidence beyond filename inference.
- Instrument article: DOI 10.17815/jlsrf-2-70.

Access limitation: publisher full article PDF returned HTTP 403; the Juelich
institutional record returned a JavaScript anti-bot challenge. The VOR metadata
was checked via Crossref and its institutional location via OpenAlex. These
responses are retained. Full article text and numbered charge-density equations
were not audited or represented as reproduced. The release documentation and
original executable wave-reconstruction stage are available and archived in
full. No absent PDF was invented or included in the cleaned task.

## License and data rights

The retrieved Zenodo record explicitly declares license `cc-by-4.0` for this
deposited archive; data were taken from that archive. This is an independent
record of archive licensing, rather than an inference that a code license must
cover its data. The original source files explicitly state Copyright 2007-2023
The HyperSpy developers and GPL version 3 or later. Adapted numerical source
retains these notices, the complete GPL text is retained below, and the migrated notice
separately preserves data attribution and transformation details. The
archive-level metadata is not used to relicense GPL source as CC BY.

The only data transformation is lossless HDF5 dataset-to-NPZ storage plus a
leading singleton batch axis. Original int64 samples are unchanged. No hidden
seed, truth or sample-derived calibration is supplied. The vacuum is an
independently acquired observation available at acquisition time.

## Maintainer-only archive index

All original source, downloads, capture scripts, environment inspection and
logs reside outside the task at
`../imaging-101-maintainer/batch10/task03_chemistry/` relative to repository root.

| Archive item | Role |
|---|---|
| `holospy-v0.3-zenodo.zip`, `zenodo_source/` | Exact used release; original HDF5 and unmodified oracle source/docs/license |
| `zenodo_record.json` | Download URLs, published MD5, author list, explicit archive CC-BY-4.0 metadata |
| `commit.json`, `upstream.zip`, `upstream/` | Initial main exploration, commit 9ded7553919a4c437409b0f20302b983979fa8dc; not the used release |
| `paper_crossref.json`, `paper_openalex.json`, `paper_record.html`, `paper_search.html` | Article VOR metadata, institutional location and access-limit responses |
| `h5py-3.16.0-cp311-cp311-win_amd64.whl`, `oracle_packages/` | Private HDF5 extraction only; no install in shared runtime |
| `capture_oracle.py`, `oracle_run.log`, `oracle_metadata.json`, `upstream_reruns.npz` | Original function runs, metadata extraction, independent double/single reruns |
| `backend_and_hashes.log`, `artifact_hashes.json`, `inspect_provenance.py` | NumPy numerical backend, scalar diagnostics and hashes |
| `build_notebook.py`, `execute_notebook.py`, `notebook_execution.log` | Authoring and actual nbconvert execution, explicit exit status 0 |
| `notebook_observations.png`, `notebook_comparison.png` | Private reviewed renders, never public input |
| `pytest.log`, `final_validation.log`, `archive_manifest.json` | Full tests, final audits and recursive archive file/hash index |
| `tmp/` | Private temporary execution files; not a test fixture source |

## Hashes and numerical artifact consumers

SHA-256 values:

| File | SHA-256 |
|---|---|
| Original object HDF5 | `cf702662c7157cc7218946330fdafc5eb9b7d0a8c01c5f872c38167754384c1f` |
| Original vacuum HDF5 | `38e1f53199dae306b9768b0e1761afdfbe08aa8698bcc8358e16304e30ab58cd` |
| `data/raw_data.npz` | `c614de1d647e6319969bf53addca02f0d53c48312da92cf8789df90e615a57c2` |
| `test/fixtures/upstream.npz` | `e17d982854ba203543454f7875e7b3cd0a82dcd45db15273b411da06b9bf252d` |
| `evaluation/baseline_reference.npz` | `ad6aacd3871437c3af7e6986761884eaa5f313ac9a4f4f27893637068567cac2` |
| `notebooks/reference_outputs/upstream_measurements.npz` | `ad6aacd3871437c3af7e6986761884eaa5f313ac9a4f4f27893637068567cac2` |
| `notebooks/reference_outputs/cleaned_measurements.npz` | `75beb61b180057557ad978e1451b9e5a83c50a34668b11839532a9fd47178a75` |

The original-run stage fixture is consumed by comparing_test/test_parity.py;
it contains the original function's frequency grid, carrier, aperture scalars,
object and vacuum cross-term waves. A redundant full ratio was removed: parity
tests derive the original ratio from the two frozen original-stage arrays.
The evaluator's baseline contains only reconstruction and is needed for the
standalone hidden contract. The notebook's original measurement contains the
same standard-method reconstruction for explicitly labeled comparison. Its
separate cleaned measurement contains the freshly computed reconstruction and
scalar runtime_seconds. These two independently labeled measurement files are
not ground truth and are not loaded by the solver. Their separate consumers
are prescribed evaluator/reference-output roles, not spare output backups.

## Original run, test-first extraction and environments

The oracle was run before writing local implementation. `capture_oracle.py`
loads unmodified release `holospy/reconstruct.py` by its archived file path;
that low-level module uses NumPy, Matplotlib and SciPy FFTPACK and has no
HyperSpy import. HDF5 extraction uses private h5py 3.16.0 via a private
sys.path. The cleaned code, tests, Judge and notebook never import this oracle
or HDF5 reader. No upstream package was installed into the shared environment.

Oracle command: shared environment `python capture_oracle.py`, launched from
the repository during capture. It reads the two exact experimental HDF5 arrays,
saves the NPZ observation batch, and independently runs original carrier/radius
estimation and original reconstruct on both images. Seed: not applicable;
the method is deterministic and no random data/noise are generated. Exit 0,
no numerical warnings, total capture/reconstruction/rerun time 0.7277861 s.

Environment: Windows-10-10.0.26200-SP0 x86_64, Python 3.11.11 MSC v1943,
NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1. NumPy uses pocketfft for FFT and
OpenBLAS 0.3.31.188.0 for its linked BLAS/LAPACK; the oracle uses SciPy FFTPACK
API. CPU execution only, GPU not required. Shared pytest 9.1.1 and nbconvert
7.17.1 are development tools, not task requirements. The default shell Python
3.14 is not the tested environment. Source requires only NumPy for the CLI and
Matplotlib for review plots. NumPy backend inspection warned that optional
pyyaml would improve configuration formatting; it is unrelated to execution.

Independent original double rerun relative complex L2 difference: 0. Original
float32-vs-double difference: 9.561154009425683e-7. Before local extraction the
stage tolerances and 1e-5 metric bounds were written in fixtures/README.md.
Then `python -m pytest test/comparing_test/test_parity.py -q` initially failed
collection because src did not yet exist. After local extraction, the first
full stage-parity test passed. The cleaned stages use local readable NumPy FFT,
roll, radial grids, tanh aperture and division; they call no domain solvers.

## Equation/stage traceability

The labels below name the complete equations in README/plan rather than
pretending to cite unaudited numbered equations of the charge-density article.

| Required stage/equation | Archived upstream file/function | Local implementation | Independent validation |
|---|---|---|---|
| Off-axis intensity and negative cross term | documented electron-holography Fourier stage in `doc/user_guide/electron_holography.rst` | physics_model.interference; approach expansion | known complex constant-wave interference proves phase/sign/normalization |
| Calibrated radial DFT frequency grid | reconstruct.freq_array | physics_model.radial_frequency | frozen original grid, absolute tolerance 1e-15 nm^-1 |
| Smooth tanh aperture | reconstruct.aperture_function | physics_model.aperture | analytic center/edge/limit cases; full-stage original parity |
| Vacuum carrier estimate / central stop | reconstruct.estimate_sideband_position | solvers.estimate_carrier | frozen (122,454); exact equality |
| Half-distance aperture size | reconstruct.estimate_sideband_size | solvers.estimate_carrier | radius absolute tolerance 1e-12 Fourier pixel |
| Positive roll, FFT normalization, no crop | reconstruct.reconstruct | physics_model.sideband_wave | frozen original complex waves rtol1e-11/atol1e-10; analytic interference |
| Corrected object/vacuum ratio | signals.hologram_image.HologramImage.reconstruct_phase, `wave_object / wave_reference` | solvers.reconstruct | full original ratio rtol1e-10/atol1e-11; vacuum/vacuum=1 |
| Strong/weak aperture comparison | archived docs' half-distance and 2/3-radius examples | active notebook sideband calls | both freshly computed; metric table and visual review |
| Independent evaluation | new benchmark contract, not upstream accuracy claim | evaluation/judge.py | known complex metric cases, invalid input and CLI tests |

## Fresh results and threshold justification

No historical paper NCC/NRMSE or cached notebook accuracy values are used.
Reproduced original-code results and fresh local results use the same data hash,
carrier, aperture, convention and full arrays. The fresh double cleaned wave
has relative complex L2 8.696900921068617e-16 against the independently captured
standard method. Its independently computed observation-sideband residual is
3.874332211118671e-16. Both are below the predeclared 1e-5 boundaries.

Self-comparison NCC=1/NRMSE=0 is deliberately not used to set a zero tolerance.
The nonzero 1e-5 gate is about 10.46 times the measured original float32/double
difference. No phase alignment, rescaling, crop or mask is applied. The
sideband metric tests consistency with the measured finite-bandwidth cross
term; it is not a raw full-intensity noise fit or independent accuracy proof.
The weak-aperture comparison has relative complex L2 0.32790405647 and
selected-band residual 0.248383570068, demonstrating that this bandwidth change
is not interchangeable with the declared selected output.

Full-array corrected amplitude range is approximately 0.02070 to 14.39673.
The minimum vacuum-sideband magnitude is 3.11825 counts, with maximum 370.55213;
it has no numerical zeros but locally weak modulation can amplify ratios.
The needle phase and surrounding curved field are visible. No region is
silently masked to suppress difficult measured features. Reference agreement
does not establish material-property accuracy or remove experimental stability
limitations.

## Commands and executed notebook

From the task directory, in the shared prepared environment:

```text
python -m src.main
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -v
python -m jupyter nbconvert --to notebook --execute notebooks/electron_holography_material_phase.ipynb --output electron_holography_material_phase_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=120
```

The Windows host's default temporary pytest folder had an existing permission
failure. Retrying with TEMP/TMP set to this task's external archive/tmp resolved
the environment issue; no assertions/tolerances were changed. pytest result:
**34 passed**, initial final run 2.41 seconds. Judge CLIs were additionally run
from unrelated working directories by tests, validating hidden-path semantics,
exit 0/1/2 and report-write failure. Missing files/keys, shapes, real/object
dtypes, NaN/Inf, poor waves, evaluator reference/config/raw failures and known
complex norm cases are covered.

The notebook was actually executed via nbconvert with exit code 0 and all
three code cells completed without error. It reconstructs both selected and
smaller-aperture comparisons actively, writes/scorers the fresh selected wave,
and displays upstream comparison plus uncorrected phase. Selected fresh
reconstruction time was **0.0522040999494493 seconds**. A 120-second timeout
allows startup and plotting overhead; it is not a fabricated algorithm time.
The executed outputs were retained in the required active notebook and the
temporary output notebook removed after inspection. Private rendered figures
were opened with view_image: four observation/Fourier panels and six
amplitude/phase/parity/aperture panels, all readable, correctly labeled, and
consistent with the measured arrays. The Markdown metric table was verified
to contain actual newlines, not literal escape characters.

Jupyter produced the host's Windows Proactor/selector-thread warning and an
IPKernel local-TCP warning; neither caused execution errors. They are retained
in the private log. There were no solver numerical warnings. Importing library
modules never selects a Matplotlib backend.

## Final semantic/source review and gate ledger

Reviewed the formulas, normalization and sign against source, documentation,
independent interference test and original-stage arrays. Checked signatures,
defaults, array axis order/dtypes, count versus wave-amplitude units, full-frame
FFT boundary, exact finite-bandwidth observable, acquisition table and actual
CLI/notebook/evaluator caller paths. The notebook's comparison-only methods are
explicitly distinguished from the submission path. An import/call audit across
src, Judge, tests and notebook found only local task imports plus permitted
general computation/plotting and shared development tools; no domain library,
cross-task import, solver checkpoint or upstream oracle is used. No synthetic
generation module is present because no observations were generated.

| Required gate | Status and evidence |
|---|---|
| Source/version/data hashes and stage mapping | Passed for selected Fourier wave stage; exact remote tag and archive record; full-article access limit stated above |
| Original and local run on exact observations | Passed; oracle_run.log, original arrays, fresh CLI and notebook |
| Independent parity and physical checks | Passed; frozen original intermediate arrays plus analytic known-wave sign/scale and constant rejection |
| Constants/units/precision/coordinates | Passed; original HDF5 metadata, public table, readme constant and physical tests |
| Standard CLI and independent Judge | Passed; fresh output and both numeric criteria |
| Judge invalid/bad predictions and failures | Passed; contract, evaluator-error and exit-code tests |
| Public input sufficiency | Passed; two full observations and complete physical/output/bandwidth/acquisition conventions in README |
| No truth/calibration/URL leakage | Passed; no truth exists, vacuum independent, solver-visible URL/import scan |
| Complete metadata/output; no metadata JSON | Passed; public README tables, only NPZ observation data |
| Minimal environment without upstream packages | Passed; local NumPy/Matplotlib subset; private HDF5 exception only during extraction |
| Explicit local core mathematics | Passed; audited FFT/roll/aperture/ratio/forward expression; no domain solver calls |
| Full test suite | Passed; 34 tests |
| Active scientific notebook and visual review | Passed; nbconvert exit0; 3 code cells; private rendered panels checked |
| Distinct provenance and nonzero metric tolerance | Passed; original, fresh and optional comparisons separated; independent reruns justify bounds |
| README/plans/source semantic consistency | Passed; explicit review described above |
| Inventory/archive/license | Passed; required layout, no original HDF5/PDF/source archive in task; license preserved; temporary notebook/cache removed |

No mandatory gate of the declared wave-reconstruction task is failed or
unverified. Independent true-wave/material-property accuracy and full
charge-density-paper reproduction are not claimed. Existing legacy benchmark
harness integration is not tested or implied by the NPZ/Judge contract.

## Independent coordinator review

On 2026-10-18 the coordinator independently reran the standard CLI, standalone
Judge from a different working directory, all 34 tests, and the active notebook
through nbconvert. All four commands exited 0. The coordinator reviewed the
explicit Fourier extraction, calibration division, independent Judge and
mathematical plan, and visually inspected the generated six-panel scientific
comparison. Public-data and import-role scans passed. Additional review logs,
executed notebook and figures are in private `batch10/coordinator/
electron_holography_material_phase`; the structured result is its sibling
`electron_holography_material_phase-review.json`. No physical truth is inferred
from this additional agreement check.

The notebook's fresh-measurement snapshot includes wall-clock runtime and is
rewritten on active execution. The current packaged snapshot is from this
coordinator run (0.055374600109644234 seconds); its hash above supersedes the
initial snapshot hash retained in private `artifact_hashes.json`. Numerical
reconstruction and metric results are unchanged. The primary notebook retains
the worker's separately identified initial execution outputs.


## Preserved full license and attribution record (documentation revision)

The user requested removal of task-root LICENSE/NOTICE files. Their complete original texts and attributions are retained verbatim below, and original bytes are backed up in the external revision archive. This relocates notices without changing their terms, scientific data rights or source copyright. Historical upstream source paths/hashes remain historical evidence. Current source references point to this section.

### Original LICENSE (verbatim)

Original-byte SHA-256: `bf516bf3ac154ca46f5baa204f59bc75a308d856c14d87f2f05d407953a593da`.

```text
                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
the GNU General Public License is intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.  We, the Free Software Foundation, use the
GNU General Public License for most of our software; it applies also to
any other work released this way by its authors.  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights.  Therefore, you have
certain responsibilities if you distribute copies of the software, or if
you modify it: responsibilities to respect the freedom of others.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must pass on to the recipients the same
freedoms that you received.  You must make sure that they, too, receive
or can get the source code.  And you must show them these terms so they
know their rights.

  Developers that use the GNU GPL protect your rights with two steps:
(1) assert copyright on the software, and (2) offer you this License
giving you legal permission to copy, distribute and/or modify it.

  For the developers' and authors' protection, the GPL clearly explains
that there is no warranty for this free software.  For both users' and
authors' sake, the GPL requires that modified versions be marked as
changed, so that their problems will not be attributed erroneously to
authors of previous versions.

  Some devices are designed to deny users access to install or run
modified versions of the software inside them, although the manufacturer
can do so.  This is fundamentally incompatible with the aim of
protecting users' freedom to change the software.  The systematic
pattern of such abuse occurs in the area of products for individuals to
use, which is precisely where it is most unacceptable.  Therefore, we
have designed this version of the GPL to prohibit the practice for those
products.  If such problems arise substantially in other domains, we
stand ready to extend this provision to those domains in future versions
of the GPL, as needed to protect the freedom of users.

  Finally, every program is threatened constantly by software patents.
States should not allow patents to restrict development and use of
software on general-purpose computers, but in those that do, we wish to
avoid the special danger that patents applied to a free program could
make it effectively proprietary.  To prevent this, the GPL assures that
patents cannot be used to render the program non-free.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Use with the GNU Affero General Public License.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU Affero General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the special requirements of the GNU Affero General Public License,
section 13, concerning interaction through a network will apply to the
combination as such.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/#GPL>.

Also add information on how to contact you by electronic and paper mail.

  If the program does terminal interaction, make it output a short
notice like this when it starts in an interactive mode:

    <program>  Copyright (C) <year>  <name of author>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, your program's commands
might be different; for a GUI interface, you would use an "about box".

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU GPL, see
<https://www.gnu.org/licenses/#GPL>.

  The GNU General Public License does not permit incorporating your program
into proprietary programs.  If your program is a subroutine library, you
may consider it more useful to permit linking proprietary applications with
the library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.  But first, please read
<http://www.gnu.org/philosophy/why-not-lgpl.html>.
```

### Original NOTICE (verbatim)

Original-byte SHA-256: `61a44e8d37bc780fa8ba5f0cb70e2c8043e0372c47583b63ce685b241edface7`.

```text
Adapted numerical source: HoloSpy v0.3, Copyright 2007-2023 The HyperSpy developers.
Local adaptations: Copyright 2026 imaging-101 contributors.
Source code is distributed under GNU GPL version 3 or, at your option, later.
The full text is included in LICENSE. Numerical algorithms are extracted into
readable local NumPy operations; HoloSpy is not a runtime dependency.

Experimental data: HoloSpy v0.3 archival deposit, DOI 10.5281/zenodo.11099344.
The archived record explicitly declares Creative Commons Attribution 4.0:
https://creativecommons.org/licenses/by/4.0/
Holograms: Vadim Migunov. Fe needle specimen: prepared by Andrew London.
Publication: V. Migunov et al., Journal of Applied Physics 117, 134301 (2015),
DOI 10.1063/1.4916609. Archive authors include Eric Prestat, Francisco de la Pena,
Vadim Migunov and collaborators, listed in the retained private archive record.
Changes to data: HDF5 intensity datasets were losslessly converted to NPZ and
given one leading batch dimension. No intensities were synthesized or modified.
The archive-level CC-BY-4.0 metadata and file-level GPL source notices are recorded
separately; no claim is made that the source files were relicensed under CC BY.
```


## Completed documentation revision and revalidation (2026-10-18)

This revision changes explanatory documents, notebook organization and license
placement. Acquisition arrays, inverse mathematics, evaluation assets, thresholds
and packaged scientific reference snapshots are unchanged. The original documents,
notebooks, code and license bytes are retained under the private archive
`D:/yan1/agent/AutoSkill/imaging-101-maintainer/batch10/revision_docs/task10/electron_holography_material_phase/before/`; removed root notices are additionally retained in
`removed-root-files/`. Full original license/attribution text and original hashes
are reproduced verbatim in the preceding preserved-license record. References to
the former root notice in source comments/module docstrings now identify this
validation record; executable ASTs match the pre-revision source after stripping
docstrings. Necessary README attribution references were updated accordingly.

The approach plan now defines the actual forward discretization, variables,
normalizations, units, initialization, inverse updates/options, termination,
evaluation and scientific limitations. The design plan records every shipped
runtime/Judge top-level signature, parameter/return contract, caller flow, file
consumer and side effect. Signatures were checked against source, and equations,
units, signs and caller roles were reviewed against the unchanged implementation.

The notebook has exactly six H2 sections: Problem, Data, Physics, Method, Results,
Conclusion, numbered 1 through 6. Data includes concrete original filenames,
repository/archive provenance, crop or processing status, visible array contracts
and acquisition metadata. Physics and Method are self-contained. All original
scientific comparison groups remain present, with upstream, fresh and initial or
alternative-condition results labeled explicitly. The default path actively
reconstructs observations with local modules and evaluates the fresh submission;
historical reference snapshots are never rewritten during review.

Revalidation gates: **passed**. The full `python -m pytest test/ -v
-p no:cacheprovider` run passed **34 tests**. Full `python -m jupyter nbconvert
--to notebook --execute notebooks/electron_holography_material_phase.ipynb --output review_executed.ipynb
--output-dir <private-task-archive> --ExecutePreprocessor.timeout=240` completed
with exit 0. Every one of **4 code cells** executed without error. All
**2 exported figures** were opened and visually inspected; comparison
panels, annotations, residuals and metric groups remain readable and scientifically
consistent. The shipped notebook contains these executed outputs. Exact absolute
commands, working directory, exit status and wall times are in `commands.json`;
stdout/stderr are in `pytest.log` and `notebook.log`. Fresh reconstruction runtimes
and metric streams are in `new-notebook-streams.json`; no historical runtime was
relabelled as a new measurement.

Byte-level checks passed for all public observations, evaluation files and every
file under `notebooks/reference_outputs/` against the pre-revision snapshot.
The private `before-hashes.json`, `reference-hashes-before.json` and
`reference-hashes-after.json` record this evidence. Figures are indexed by
`figures.json`; the executed review notebook and PNGs stay private. The private
`final-manifest.json` indexes the entire final task with SHA256 hashes, including
this validation record, and records the immutable-reference comparison. No
temporary executed notebook, screenshot, upstream source or capture script is
shipped in the task. No scientific completion gate is replaced by this
documentation review; the original numerical evidence remains authoritative.

Final reviewed document hashes (the validation record is hashed externally to
avoid a self-referential digest):

| File | SHA256 |
|---|---|
| `plan/approach.md` | `24175d352e034e51dd6ed45105f7379eb6757b006d66aeec16899a6f7008ba0a` |
| `plan/design.md` | `c5579b122ac5559bf1ebbac338bba014c768812db68223e1cfc79268715b7b86` |
| `notebooks/electron_holography_material_phase.ipynb` | `255de95bf39814a02a96ac62eae5c8cff7ed36f54ed659854bff041b5a1a48a6` |
| `README.md` | `5c1cd4bdb6c6248c07b5532f6d0a08af9a9b129c80d394edd350ad1e174dd890` |

### Final notebook Markdown readability review

The final prose review corrected concatenated English words and spacing while
preserving scientific meanings and all fenced mathematical formulas. Repeated
source/crop descriptions in the phase-shifting and Spitzer Data sections were
consolidated into their detailed provenance paragraphs. Every code cell was
compared in full against the successfully executed private
`review_executed.ipynb`: source, execution count, outputs and cell metadata are
identical. This Markdown-only change therefore did not rerun unchanged code or
relabel old computations. `final-markdown-proof.json` records the equality
checks and final notebook SHA256; the document-hash table above and private
`final-manifest.json` are updated to the final bytes.


## Solver-visible URL boundary review (2026-10-29)

Software documentation, standard-toolbox archives and mixed code/data archive links were removed from public README references while preserving attribution and license notices. The sources remain here for maintainer provenance; they do not add solver calibration.

- source: https://doi.org/10.5281/zenodo.11099344.; role: software documentation/source or mixed code-data archive; moved to maintainer provenance
