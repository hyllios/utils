"""
Self-contained tools built on the library: each subpackage solves one problem
end to end and carries its own CLI.

They may use anything in `waw` (units, analysis helpers), while nothing in the
rest of `waw` imports them -- so they stay optional and cannot create import
cycles. Import explicitly, e.g. ``from waw.utils.eliashberg import
tc_linearized``.

  * `eliashberg` -- band-resolved isotropic Eliashberg solver: Tc from the
    linearized equations, given alpha^2F (single or band-resolved) and mu*.
"""

from . import runs                                             # noqa: F401
from .runs import run_dir, stamp, survey, prunable_bytes       # noqa: F401
