Query Semantics and Verification Budgets for Versioned Music Features
reproduction

Requirements
  CPython 3.14 on a POSIX system with numpy 2.5.3 (SQLite ships with the
  standard library).  The package is self-contained: every input is included and
  no network access is used.

Command
  python3.14 -B reproduce.py

  The command copies code/ and data/ into runs/<UTC timestamp>/ and runs five
  stages there: the shared root queries, the shared-build SQLite baseline, the
  targeted variable-membership witness, the joint verification register, and the
  cross-domain transfer experiment with its global-edition controls.  The
  shipped tree is never modified.  Each stage has a 120-second CPU
  limit; use --cpu-limit 7200 if the machine is slow.

Comparison
  Each result file under runs/.../results/ is compared with expected.json,
  which holds the corresponding shipped reference register.  Execution records,
  environment fields, absolute paths and wall-clock timings are dropped before
  comparison; integers and text must then agree exactly and floats exactly too,
  since this package reproduces without drift.  PASS and CHECK.json indicate a
  full match.

Modifications
  Three primary programs are shipped with their input wiring rewritten, and this
  is the only change made to them:

    code/sqlite_semantic_baseline.py
    code/sqlite_targeted_membership.py
    code/joint_verification_exhaustive.py

  Each originally located the repository's shared/ tree through
  pathlib.Path(__file__).parent and wrote into a fixed sibling directory.  In the
  package they take --data, --results, --code and --out from argv instead, which
  is what lets reproduce.py drive them from a runs/<UTC timestamp>/ copy; and
  each pins the BLAS/OpenMP thread-count variables to 1 at import so that the
  reduced-precision reduction order is fixed.  Every computation, constant,
  query and output field is unmodified.  The receipts these programs emit record
  the path-derived source hash and the input paths, so those fields necessarily
  differ from the originals; they are excluded from the comparison, as are the
  other execution-record and environment fields.

  The remaining programs are shipped verbatim: code/run_queries.py,
  code/experiment.py, code/controls.py and code/__init__.py are byte-identical to
  the manuscript's copies.

Expected counts
  Eleven compared reference registers: five primary registers (the SQLite
  baseline receipt, the independent candidate selection, the targeted membership
  witnesses, the joint verification register and its world answers) and six
  transfer registers (all cases, all budgets, the SQL world checks, the summary
  and the two global-edition control registers).  The query workload covers
  336,000 world--query answers over sixteen constructed worlds.

Data
  data/official_sample/ holds the retained AcousticBrainz feature and metadata
  values (CC0).  data/audio/ and data/raster_panel.npz hold the transfer
  experiment inputs.  Sources and terms are in SOURCE.txt and LICENSE.txt.

Files
  reproduce.py      the driver
  expected.json     expected result content, keyed by run-relative path
  code/             the analysis programs
  data/             the inputs
  RESULTS.tsv       paper item to program, line, output and input
  ENVIRONMENT.json  the recorded release environment
