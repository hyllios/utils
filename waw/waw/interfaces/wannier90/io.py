"""
File I/O for Wannier90 file formats.

Readers and writers for the standard Wannier90 file formats:
  Input:   .win, .nnkp, .eig, .mmn, .amn
  Output:  _hr.dat, _centres.xyz

All arrays are returned as plain numpy arrays.  Conversion to
PyTorch tensors is handled in data.py so that this layer stays
free of framework dependencies.

Wannier90 file format reference:
  https://wannier.org/support/
  Pizzi et al., J. Phys.: Condens. Matter 32 (2020) 165902
"""

import gzip
import re
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open(path: str | Path, mode: str = "r"):
    return open(Path(path), mode, encoding="utf-8")


def resolve_input_path(path: str | Path) -> Path:
    """
    Path a reader should actually open: `path` itself if it exists, else a
    gzip'd sibling `<path>.gz` if that exists, else `path` unchanged (so a
    natural FileNotFoundError surfaces on read).

    This lets every reader transparently accept e.g. `seed.mmn.gz` in place
    of `seed.mmn`, so large overlap/eigenvalue files can be committed
    compressed and used without a manual decompression step.
    """
    path = Path(path)
    if path.exists():
        return path
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gz
    return path


def _read_text(path: str | Path) -> str:
    p = resolve_input_path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return f.read()
    return p.read_text(encoding="utf-8")


def _read_bytes(path: str | Path) -> bytes:
    p = resolve_input_path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rb") as f:
            return f.read()
    return p.read_bytes()


def _next_nonblank(lines: list[str], idx: int) -> tuple[str, int]:
    """Return the next non-blank, non-comment line and its new index."""
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if line and not line.startswith("#") and not line.startswith("!"):
            return line, idx
    raise EOFError("Unexpected end of file")


# ---------------------------------------------------------------------------
# .win reader
# ---------------------------------------------------------------------------

def _strip_win_comments(text: str) -> str:
    """
    Drop Wannier90 comments line-by-line, matching W90's own reader: a
    line is dropped entirely once left-trimmed if it starts with '!' or
    '#'; otherwise it's truncated at the first '!' or '#'. Applied to the
    whole file before any block/keyword parsing, so a commented-out
    alternative block can never be mistaken for the active one.
    """
    out_lines = []
    for line in text.splitlines():
        if line.strip().startswith(("!", "#")):
            out_lines.append("")
            continue
        cut = min((i for i in (line.find("!"), line.find("#")) if i != -1), default=None)
        out_lines.append(line if cut is None else line[:cut])
    return "\n".join(out_lines)


def read_win(path: str | Path) -> dict:
    """
    Parse a Wannier90 .win input file.

    Returns a dict with keys matching the Wannier90 parameter names
    (lower-cased).  Block sections (projections, unit_cell_cart, etc.)
    are stored as lists of strings for the caller to interpret.

    Only the parameters needed by the optimization engine are parsed
    with typed values.  Everything else is kept as raw strings.
    """
    path = Path(path)
    text = _strip_win_comments(_read_text(path))

    params = {}

    # ---- extract block sections (%block...%endblock or begin...end) -------
    # W90 accepts both syntaxes interchangeably.  W90's own reader is
    # lenient about the separator between "begin"/"end" and the block name
    # (it only checks the keyword substring appears somewhere on a line
    # starting with begin/end), e.g. accepting "end_unit_cell_cart"
    # (underscore, no space) -- so [\s_]* (not \s+) matches that too.
    block_pattern = re.compile(
        r"(?:%block|begin)[\s_]*(\w+)(.*?)(?:%endblock|end)[\s_]*\1",
        re.IGNORECASE | re.DOTALL,
    )
    for match in block_pattern.finditer(text):
        key   = match.group(1).lower()
        lines = [l.strip() for l in match.group(2).strip().splitlines()
                 if l.strip() and not l.strip().startswith("!")]
        params[key] = lines

    # ---- remove block sections before parsing scalar keywords -------------
    text_no_blocks = block_pattern.sub("", text)

    # ---- parse key = value pairs ------------------------------------------
    kv_pattern = re.compile(
        r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[=:]\s*(.+)",
        re.MULTILINE,
    )
    for match in kv_pattern.finditer(text_no_blocks):
        key   = match.group(1).lower().strip()
        value = match.group(2).split("!")[0].split("#")[0].strip()
        params[key] = _parse_win_value(value)

    # ---- select_projections: the one keyword W90 writes bare, with no
    # '='/':' separator (e.g. "select_projections 1 2 3 4"), so the
    # generic kv_pattern above never matches it.
    sel_match = re.search(
        r"^\s*select_projections\s+(.+)$", text_no_blocks,
        re.IGNORECASE | re.MULTILINE,
    )
    if sel_match:
        value = sel_match.group(1).split("!")[0].split("#")[0].strip()
        params["select_projections"] = value

    return params


def _parse_win_value(s: str):
    """Try to cast a .win value to int, float, bool, or leave as str."""
    if s.lower() in ("true", ".true.", "t"):
        return True
    if s.lower() in ("false", ".false.", "f"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# .nnkp reader
# ---------------------------------------------------------------------------

def read_nnkp(path: str | Path) -> dict:
    """
    Parse a Wannier90 .nnkp file (generated by wannier90.x -pp).

    Returns a dict with:
      kpoints  : (nk, 3)       float64  k-points in crystal coordinates
      nnkpts   : (nk, nnb, 2)  int      nearest-neighbour table
                  nnkpts[ik, ib, 0] = index of k+b in kpoints (0-based)
                  nnkpts[ik, ib, 1] = index of the G-vector needed:
                                       k+b = kpoints[ik2] + G_vectors[ig]
      g_vectors: (nk, nnb, 3)  int      G-vectors (reciprocal lattice units)
      nntot    : int            number of nearest-neighbour shells used
    """
    path = Path(path)
    lines = _read_text(path).splitlines()
    idx   = 0

    # skip header / comments until we hit a known block keyword
    def skip_to(keyword: str):
        nonlocal idx
        while idx < len(lines):
            if lines[idx].strip().lower().startswith(keyword.lower()):
                idx += 1
                return
            idx += 1
        raise ValueError(f"Keyword '{keyword}' not found in {path}")

    result = {}

    # ---- k-points ---------------------------------------------------------
    skip_to("begin kpoints")
    nk = int(lines[idx].strip()); idx += 1
    kpoints = np.zeros((nk, 3), dtype=np.float64)
    for ik in range(nk):
        kpoints[ik] = [float(x) for x in lines[idx].split()]; idx += 1
    result["kpoints"] = kpoints

    # ---- nearest-neighbour k-point table ----------------------------------
    skip_to("begin nnkpts")
    nntot = int(lines[idx].strip()); idx += 1
    result["nntot"] = nntot

    # nnkpts[ik, ib] = (ik2, ig1, ig2, ig3)
    nnkpts    = np.zeros((nk, nntot), dtype=np.int64)   # neighbour k index
    g_vectors = np.zeros((nk, nntot, 3), dtype=np.int64)

    for ik in range(nk):
        for ib in range(nntot):
            parts = lines[idx].split(); idx += 1
            # W90 uses 1-based indices; convert to 0-based
            ik_self = int(parts[0]) - 1
            ik2     = int(parts[1]) - 1
            gvec    = [int(parts[2]), int(parts[3]), int(parts[4])]
            assert ik_self == ik, (
                f"nnkpts: expected k-index {ik+1}, got {ik_self+1}"
            )
            nnkpts[ik, ib]     = ik2
            g_vectors[ik, ib]  = gvec

    result["nnkpts"]    = nnkpts
    result["g_vectors"] = g_vectors

    # ---- exclude_bands (optional) ----------------------------------------
    try:
        skip_to("begin exclude_bands")
        nexcl = int(lines[idx].strip()); idx += 1
        excl  = [int(lines[idx + i].strip()) - 1 for i in range(nexcl)]
        result["exclude_bands"] = np.array(excl, dtype=np.int64)
        idx += nexcl
    except (ValueError, IndexError):
        result["exclude_bands"] = np.array([], dtype=np.int64)

    return result


# ---------------------------------------------------------------------------
# .eig reader
# ---------------------------------------------------------------------------

def read_eig(path: str | Path) -> np.ndarray:
    """
    Parse a Wannier90 .eig file.

    The file contains one line per (band, k-point) pair:
        band_index  k_index  eigenvalue_eV

    Returns eig[ik, ib] (nk, nb) float64 array of eigenvalues in eV.
    W90 uses 1-based indices; we convert to 0-based.
    """
    path = Path(path)
    data = np.loadtxt(resolve_input_path(path), dtype=np.float64)
    # columns: band (1-based), kpoint (1-based), energy
    bands  = data[:, 0].astype(int)
    kpts   = data[:, 1].astype(int)
    energies = data[:, 2]

    nb = bands.max()
    nk = kpts.max()
    eig = np.zeros((nk, nb), dtype=np.float64)
    eig[kpts - 1, bands - 1] = energies

    return eig


# ---------------------------------------------------------------------------
# .mmn reader
# ---------------------------------------------------------------------------

def read_mmn(path: str | Path) -> tuple[np.ndarray, list[tuple]]:
    """
    Parse a Wannier90 .mmn file.

    The file encodes overlap matrices M^(k,b)_{mn} = <u_{mk}|u_{n,k+b}>.

    Returns:
      Mmn    : (nk, nnb, nb, nb) complex128 array
                 Mmn[ik, ib, m, n] = M^(k,b)_{mn}
      kpb_map: list of (ik, ik2, g1, g2, g3) for each (k, b) pair,
                 in the order they appear in the file (ik 0-based, ik2 0-based)
    """
    path  = Path(path)
    lines = _read_text(path).splitlines()
    idx   = 1   # skip comment line

    header = lines[idx].split(); idx += 1
    nb, nk, nnb = int(header[0]), int(header[1]), int(header[2])

    Mmn     = np.zeros((nk, nnb, nb, nb), dtype=np.complex128)
    kpb_map = []
    nb2     = nb * nb

    for ib_global in range(nk * nnb):
        # header line: ik  ik2  g1  g2  g3  (1-based k indices)
        parts = lines[idx].split(); idx += 1
        ik  = int(parts[0]) - 1
        ik2 = int(parts[1]) - 1
        g   = (int(parts[2]), int(parts[3]), int(parts[4]))

        ib = ib_global % nnb
        kpb_map.append((ik, ik2, *g))

        # Parse nb² Re/Im pairs in one vectorized shot instead of a Python loop.
        block = lines[idx : idx + nb2]
        idx  += nb2
        vals  = np.fromiter(
            (float(x) for line in block for x in line.split()),
            dtype=np.float64, count=nb2 * 2,
        ).reshape(nb2, 2)

        # W90 writes in Fortran column-major order (fastest index = row = m)
        Mmn[ik, ib] = (vals[:, 0] + 1j * vals[:, 1]).reshape((nb, nb), order="F")

    return Mmn, kpb_map


# ---------------------------------------------------------------------------
# .amn reader
# ---------------------------------------------------------------------------

def read_amn(path: str | Path) -> np.ndarray:
    """
    Parse a Wannier90 .amn file.

    The file encodes projection matrices A^(k)_{mn} = <psi_{mk}|g_n>
    where g_n are the trial orbitals.

    Returns Amn[ik, m, n] (nk, nb, nw) complex128 array.
    """
    path = Path(path)
    lines = _read_text(path).splitlines()
    idx   = 1   # skip comment line

    header = lines[idx].split(); idx += 1
    nb, nk, nw = int(header[0]), int(header[1]), int(header[2])

    Amn  = np.zeros((nk, nb, nw), dtype=np.complex128)
    n_entries = nb * nk * nw

    # Parse all entries at once: each line is "m n ik Re Im" (1-based indices)
    data = np.fromiter(
        (float(x) for line in lines[idx : idx + n_entries] for x in line.split()),
        dtype=np.float64, count=n_entries * 5,
    ).reshape(n_entries, 5)

    m_idx  = data[:, 0].astype(int) - 1
    n_idx  = data[:, 1].astype(int) - 1
    ik_idx = data[:, 2].astype(int) - 1
    Amn[ik_idx, m_idx, n_idx] = data[:, 3] + 1j * data[:, 4]

    return Amn


# ---------------------------------------------------------------------------
# UNK reader (periodic Bloch parts on the real-space FFT grid)
# ---------------------------------------------------------------------------

def unk_filename(ik: int, spin_channel: int = 1) -> str:
    """UNKnnnnn.s filename for a 1-based k-point index (Wannier90 convention)."""
    return f"UNK{ik:05d}.{spin_channel}"


def read_unk(path: str | Path) -> dict:
    """
    Parse a Wannier90 UNK file (written by pw2wannier90 with
    write_unk=.true.), in either the formatted (ASCII, wvfn_formatted=.true.)
    or the default unformatted (Fortran-binary) variant. The variant is
    auto-detected.

    Grid ordering matches Wannier90's plot_wannier (src/plot.F90): points
    are stored with the first grid axis fastest, then the second, then
    the third (Fortran/column-major for a (ngx, ngy, ngz) array), one
    band's full grid at a time.

    Returns a dict with:
      ngx, ngy, ngz : int                            FFT grid dimensions
      ik            : int                             1-based k-point index (as stored)
      nbnd          : int                             number of bands in the file
      u_nk          : (nbnd, ngx, ngy, ngz) complex128 periodic Bloch parts
    """
    path = Path(path)
    raw = _read_bytes(path)   # transparently decompresses a .gz sibling

    # Auto-detect: an unformatted UNK begins with the Fortran record-length
    # marker for its 5-int32 header record, i.e. a leading int32 == 20. A
    # formatted UNK begins with ASCII digits/spaces, which as an int32 is
    # some large value, never 20.
    if len(raw) >= 4 and np.frombuffer(raw[:4], dtype=np.int32)[0] == 20:
        return _read_unk_unformatted(raw, path)
    return _read_unk_formatted(raw, path)


def _read_unk_formatted(raw: bytes, path: Path) -> dict:
    lines = raw.decode("utf-8").splitlines()
    ngx, ngy, ngz, ik, nbnd = (int(x) for x in lines[0].split()[:5])
    npts = ngx * ngy * ngz

    data = np.array(
        [[float(x) for x in ln.split()] for ln in lines[1:1 + nbnd * npts]],
        dtype=np.float64,
    )
    if data.shape != (nbnd * npts, 2):
        raise ValueError(
            f"{path}: expected {nbnd * npts} data lines of 2 columns, "
            f"got shape {data.shape}"
        )
    u_flat = (data[:, 0] + 1j * data[:, 1]).reshape(nbnd, npts)
    u_nk = np.stack(
        [u_flat[ib].reshape((ngx, ngy, ngz), order="F") for ib in range(nbnd)]
    )
    return {"ngx": ngx, "ngy": ngy, "ngz": ngz, "ik": ik, "nbnd": nbnd, "u_nk": u_nk}


def _read_unk_unformatted(raw: bytes, path: Path) -> dict:
    """
    Fortran sequential-unformatted UNK: each WRITE is one record wrapped in
    a leading and trailing 4-byte int32 record-length marker (gfortran's
    default). Layout mirrors plot.F90's unformatted branch:
      record 0 : ngx, ngy, ngz, ik, nbnd            (5 int32)
      record b : (u_nk(g), g=1..ngx*ngy*ngz)         (npts complex128), per band
    """
    i32 = np.int32().itemsize   # 4

    def _record(off: int, dtype, count: int) -> tuple[np.ndarray, int]:
        n = np.frombuffer(raw[off:off + i32], dtype=np.int32)[0]
        off += i32
        arr = np.frombuffer(raw[off:off + n], dtype=dtype, count=count)
        off += n
        trailer = np.frombuffer(raw[off:off + i32], dtype=np.int32)[0]
        off += i32
        if trailer != n:
            raise ValueError(f"{path}: Fortran record markers disagree ({n} != {trailer})")
        return arr, off

    header, off = _record(0, np.int32, 5)
    ngx, ngy, ngz, ik, nbnd = (int(x) for x in header)
    npts = ngx * ngy * ngz

    bands = []
    for _ in range(nbnd):
        u_flat, off = _record(off, np.complex128, npts)
        bands.append(u_flat.reshape((ngx, ngy, ngz), order="F"))
    u_nk = np.stack(bands)
    return {"ngx": ngx, "ngy": ngy, "ngz": ngz, "ik": ik, "nbnd": nbnd, "u_nk": u_nk}


def read_spn(path: str | Path) -> dict:
    """
    Parse a Wannier90 .spn file (written by pw2wannier90 with
    write_spn=.true.): Pauli spin-operator matrix elements
    <psi_nk|sigma_i|psi_mk> between ab-initio (pre-Wannierization,
    "Bloch gauge") eigenstates, per k-point. Auto-detects formatted
    (ASCII) vs the default unformatted (Fortran-binary) variant.

    File layout: a header line/record, then (num_bands, num_kpts), then
    for each k-point the upper triangle only (m outer 1..num_bands, n
    inner 1..m) of the three Pauli matrices (sigma_x, sigma_y, sigma_z)
    -- the lower triangle is Hermitian-filled here (S_mn = conj(S_nm)).

    Returns a dict with:
      header               : str
      num_bands, num_kpts  : int
      spn                  : (num_kpts, num_bands, num_bands, 3) complex128,
                              spn[k, n, m, :] = <psi_nk|sigma_{x,y,z}|psi_mk>
                              (0-based n, m; k in the seedname.eig/.mmn order)
    """
    path = Path(path)
    raw = _read_bytes(path)

    # Unformatted starts with the Fortran record-length marker for the
    # header's fixed character(len=60) record, i.e. a leading int32 == 60
    # (same style of auto-detection as read_unk's "== 20" check).
    if len(raw) >= 4 and np.frombuffer(raw[:4], dtype=np.int32)[0] == 60:
        return _read_spn_unformatted(raw, path)
    return _read_spn_formatted(raw, path)


def _spn_from_triangle(header: str, nb: int, nk: int, packed: np.ndarray) -> dict:
    """packed: (nk, nb*(nb+1)/2, 3), in Fortran's (m outer, n<=m inner) order."""
    m_idx, n_idx = np.tril_indices(nb)   # (row=m outer, col=n inner, n<=m), 0-based
    spn = np.zeros((nk, nb, nb, 3), dtype=np.complex128)
    spn[:, n_idx, m_idx, :] = packed
    off_diag = n_idx != m_idx
    spn[:, m_idx[off_diag], n_idx[off_diag], :] = packed[:, off_diag, :].conj()
    return {"header": header, "num_bands": nb, "num_kpts": nk, "spn": spn}


def _read_spn_formatted(raw: bytes, path: Path) -> dict:
    lines = raw.decode("utf-8").splitlines()
    header = lines[0].strip()
    nb, nk = (int(x) for x in lines[1].split()[:2])
    ntri = nb * (nb + 1) // 2

    body = lines[2:2 + nk * ntri * 3]
    if len(body) != nk * ntri * 3:
        raise ValueError(f"{path}: expected {nk * ntri * 3} data lines, got {len(body)}")
    vals = np.array([[float(x) for x in ln.split()[:2]] for ln in body], dtype=np.float64)
    packed = (vals[:, 0] + 1j * vals[:, 1]).reshape(nk, ntri, 3)
    return _spn_from_triangle(header, nb, nk, packed)


def _read_spn_unformatted(raw: bytes, path: Path) -> dict:
    """
    Fortran sequential-unformatted .spn: header (60-byte character record),
    (num_bands, num_kpts) (2 int32), then one record per k-point holding
    ((spn_temp(s, m), s=1, 3), m=1, num_bands*(num_bands+1)/2) -- s (the
    Cartesian spin component) varies fastest within each m-block.
    """
    i32 = np.int32().itemsize

    def _record(off: int, dtype, count: int) -> tuple[np.ndarray, int]:
        n = np.frombuffer(raw[off:off + i32], dtype=np.int32)[0]
        off += i32
        arr = np.frombuffer(raw[off:off + n], dtype=dtype, count=count)
        off += n
        trailer = np.frombuffer(raw[off:off + i32], dtype=np.int32)[0]
        off += i32
        if trailer != n:
            raise ValueError(f"{path}: Fortran record markers disagree ({n} != {trailer})")
        return arr, off

    header_bytes, off = _record(0, np.uint8, 60)
    header = bytes(header_bytes).decode("utf-8", errors="replace").strip()
    dims, off = _record(off, np.int32, 2)
    nb, nk = int(dims[0]), int(dims[1])
    ntri = nb * (nb + 1) // 2

    packed = np.empty((nk, ntri, 3), dtype=np.complex128)
    for ik in range(nk):
        rec, off = _record(off, np.complex128, 3 * ntri)
        packed[ik] = rec.reshape(ntri, 3)   # s fastest (inner), m slower (outer)
    return _spn_from_triangle(header, nb, nk, packed)


def read_uHu(path: str | Path) -> dict:
    """
    Parse a Wannier90 .uHu file (written by pw2wannier90 with
    write_uHu=.true.): the ab-initio (pre-Wannierization) matrix elements
    <u_{k+b1}|H_k|u_{k+b2}> between the two nearest-neighbour shells of
    every k-point -- the extra overlap postw90's orbital-magnetization
    module (`berry_task = morb`) needs on top of the usual .mmn/.amn/.eig,
    to build the `CC_R` real-space quantity (`core.hamiltonian.compute_cc_r`).

    File layout: a header record, then (num_bands, num_kpts, nntot), then
    for each k-point, for nn2 in 1..nntot, for nn1 in 1..nntot: one
    num_bands x num_bands complex128 block (Fortran column-major, first
    index fastest), which pw2wannier90 writes transposed relative to
    wannier90's own convention, so wannier90 transposes every block
    immediately after reading it -- reproduced here. Auto-detects
    formatted (ASCII) vs the default unformatted (Fortran-binary) variant
    (leading record-length marker == 60, the header's
    `character(len=60)` record).

    Returns a dict with:
      header                    : str
      num_bands, num_kpts, nntot: int
      uHu : (num_kpts, nntot, nntot, num_bands, num_bands) complex128
            uHu[k, nn1, nn2, m, n] = <u_{m,k+b(nn1)}|H_k|u_{n,k+b(nn2)}>
            (0-based m, n, nn1, nn2; k in the seedname.eig/.mmn/.nnkp order;
            nn1/nn2 in the seedname.nnkp neighbour-shell order)
    """
    path = Path(path)
    raw = _read_bytes(path)

    if len(raw) >= 4 and np.frombuffer(raw[:4], dtype=np.int32)[0] == 60:
        return _read_uHu_unformatted(raw, path)
    return _read_uHu_formatted(raw, path)


def _read_uHu_formatted(raw: bytes, path: Path) -> dict:
    lines = raw.decode("utf-8").splitlines()
    header = lines[0].strip()
    nb, nk, nntot = (int(x) for x in lines[1].split()[:3])

    body = lines[2:2 + nk * nntot * nntot * nb * nb]
    if len(body) != nk * nntot * nntot * nb * nb:
        raise ValueError(
            f"{path}: expected {nk * nntot * nntot * nb * nb} data lines, got {len(body)}"
        )
    vals = np.array([[float(x) for x in ln.split()[:2]] for ln in body], dtype=np.float64)
    flat = vals[:, 0] + 1j * vals[:, 1]
    # File order is ik (slowest) -> nn2 -> nn1 (fastest among block indices);
    # each (nb, nb) block is written m-outer/n-inner then transposed by w90,
    # which nets a plain C-order reshape of that block.
    by_nn2_nn1 = flat.reshape(nk, nntot, nntot, nb, nb)   # (ik, nn2, nn1, m, n)
    uHu = by_nn2_nn1.transpose(0, 2, 1, 3, 4)             # (ik, nn1, nn2, m, n)
    return {"header": header, "num_bands": nb, "num_kpts": nk, "nntot": nntot, "uHu": uHu}


def _read_uHu_unformatted(raw: bytes, path: Path) -> dict:
    """
    Fortran sequential-unformatted .uHu: header (60-byte character record),
    (num_bands, num_kpts, nntot) (3 int32), then one record per (k, nn2, nn1)
    holding the num_bands x num_bands block ((Ho(n, m), n=1,nb), m=1,nb) --
    n (bra) fastest, i.e. Fortran column-major -- transposed immediately
    after reading, reproduced here with `.reshape(nb, nb, order='F').T`.
    """
    i32 = np.int32().itemsize

    def _record(off: int, dtype, count: int) -> tuple[np.ndarray, int]:
        n = np.frombuffer(raw[off:off + i32], dtype=np.int32)[0]
        off += i32
        arr = np.frombuffer(raw[off:off + n], dtype=dtype, count=count)
        off += n
        trailer = np.frombuffer(raw[off:off + i32], dtype=np.int32)[0]
        off += i32
        if trailer != n:
            raise ValueError(f"{path}: Fortran record markers disagree ({n} != {trailer})")
        return arr, off

    header_bytes, off = _record(0, np.uint8, 60)
    header = bytes(header_bytes).decode("utf-8", errors="replace").strip()
    dims, off = _record(off, np.int32, 3)
    nb, nk, nntot = int(dims[0]), int(dims[1]), int(dims[2])

    uHu = np.empty((nk, nntot, nntot, nb, nb), dtype=np.complex128)
    for ik in range(nk):
        for nn2 in range(nntot):
            for nn1 in range(nntot):
                rec, off = _record(off, np.complex128, nb * nb)
                block = rec.reshape(nb, nb, order="F")   # n (bra) fastest, as written
                uHu[ik, nn1, nn2] = block.T               # w90's post-read transpose
    return {"header": header, "num_bands": nb, "num_kpts": nk, "nntot": nntot, "uHu": uHu}


def _read_sX(path: str | Path, tag: str) -> dict:
    """
    Shared parser for .sHu/.sIu (written by pw2wannier90 with
    write_sHu=.true./write_sIu=.true., needed for the spin Hall
    conductivity Ryoo method, `analysis.spin_hall`): the ab-initio
    matrix elements

        sHu(k,b)[n,m,c] = <u_{m,k}| sigma_c H(k) |u_{n,k+b}>
        sIu(k,b)[n,m,c] = <u_{m,k}| sigma_c       |u_{n,k+b}>

    (`tag` selects which). Unlike `.uHu` (two nearest-neighbour indices,
    nn1/nn2), these have only ONE neighbour index `b` (matching `.mmn`'s
    own convention) plus an extra Pauli-axis index `c` (x,y,z).

    File layout: a header record, then (num_bands, num_kpts, nntot), then
    for each k-point, for ib in 1..nntot, for c in 1..3: one num_bands x
    num_bands complex128 block, `n` (the k+b index) fastest -- pw2wannier90
    writes it transposed relative to its own bra/ket convention (same
    "transpose immediately after reading" pattern as `.uHu`, see
    `pw2wannier90.f90::compute_shc`/wannier90's `get_SBB_R`/`get_SAA_R`).
    `sHu` additionally carries an Ry->eV conversion pw2wannier90 applies at
    write time (`H_evc_kb` built from QE's internal Rydberg-unit `h_psi`);
    `sIu` needs no such factor.

    Auto-detects formatted (ASCII) vs the default unformatted
    (Fortran-binary) variant (leading record-length marker == 60).

    Returns a dict with:
      header                   : str
      num_bands, num_kpts, nntot: int
      data : (num_kpts, nntot, num_bands, num_bands, 3) complex128
             data[k, ib, m, n, c] = <u_{m,k}| sigma_c [H(k)] |u_{n,k+b(ib)}>
             (0-based m, n, ib; k/ib in the seedname.eig/.mmn/.nnkp order;
             eV units for sHu, dimensionless Pauli convention for sIu)
    """
    path = Path(path)
    raw = _read_bytes(path)

    if len(raw) >= 4 and np.frombuffer(raw[:4], dtype=np.int32)[0] == 60:
        return _read_sX_unformatted(raw, path, tag)
    return _read_sX_formatted(raw, path, tag)


def _read_sX_formatted(raw: bytes, path: Path, tag: str) -> dict:
    lines = raw.decode("utf-8").splitlines()
    header = lines[0].strip()
    nb, nk, nntot = (int(x) for x in lines[1].split()[:3])

    n_blocks = nk * nntot * 3
    body = lines[2:2 + n_blocks * nb * nb]
    if len(body) != n_blocks * nb * nb:
        raise ValueError(f"{path}: expected {n_blocks * nb * nb} data lines, got {len(body)}")
    vals = np.array([[float(x) for x in ln.split()[:2]] for ln in body], dtype=np.float64)
    flat = vals[:, 0] + 1j * vals[:, 1]
    # File order: ik (slowest) -> ib -> c -> bra(m, slow) -> ket(n, fast);
    # a plain C-order reshape already lands the trailing two axes as (m, n)
    # directly (verified against a hand-simulated Fortran write loop -- no
    # extra transpose needed here, unlike the unformatted reader's
    # Fortran-order-then-.T dance).
    by_ib_c = flat.reshape(nk, nntot, 3, nb, nb)   # (ik, ib, c, m, n) already
    data = by_ib_c.transpose(0, 1, 3, 4, 2)         # -> (ik, ib, m, n, c)
    return {"header": header, "num_bands": nb, "num_kpts": nk, "nntot": nntot, "data": data}


def _read_sX_unformatted(raw: bytes, path: Path, tag: str) -> dict:
    i32 = np.int32().itemsize

    def _record(off: int, dtype, count: int) -> tuple[np.ndarray, int]:
        n = np.frombuffer(raw[off:off + i32], dtype=np.int32)[0]
        off += i32
        arr = np.frombuffer(raw[off:off + n], dtype=dtype, count=count)
        off += n
        trailer = np.frombuffer(raw[off:off + i32], dtype=np.int32)[0]
        off += i32
        if trailer != n:
            raise ValueError(f"{path}: Fortran record markers disagree ({n} != {trailer})")
        return arr, off

    header_bytes, off = _record(0, np.uint8, 60)
    header = bytes(header_bytes).decode("utf-8", errors="replace").strip()
    dims, off = _record(off, np.int32, 3)
    nb, nk, nntot = int(dims[0]), int(dims[1]), int(dims[2])

    data = np.empty((nk, nntot, nb, nb, 3), dtype=np.complex128)
    for ik in range(nk):
        for ib in range(nntot):
            for c in range(3):
                rec, off = _record(off, np.complex128, nb * nb)
                block = rec.reshape(nb, nb, order="F")   # n (ket) fastest, as written
                data[ik, ib, :, :, c] = block.T           # w90's post-read transpose -> (m,n)
    return {"header": header, "num_bands": nb, "num_kpts": nk, "nntot": nntot, "data": data}


def read_sHu(path: str | Path) -> dict:
    """
    Parse a Wannier90 .sHu file (written by pw2wannier90 with
    write_sHu=.true.): <u_{m,k}|sigma_c H(k)|u_{n,k+b}>, one of the two
    new ab-initio quantities the spin Hall conductivity Ryoo method
    (Ryoo, Park & Souza, PRB 99, 235113 (2019)) needs on top of the usual
    .mmn/.amn/.eig/.spn (`analysis.spin_hall`'s Qiao-method operators need
    only those). See `_read_sX`'s docstring for the file format.

    Returns dict with `data`: (num_kpts, nntot, num_bands, num_bands, 3)
    complex128, eV units (the Ry->eV factor pw2wannier90 applies at write
    time is already baked in).
    """
    return _read_sX(path, "sHu")


def read_sIu(path: str | Path) -> dict:
    """
    Parse a Wannier90 .sIu file (written by pw2wannier90 with
    write_sIu=.true.): <u_{m,k}|sigma_c|u_{n,k+b}>, the other new ab-initio
    quantity the spin Hall conductivity Ryoo method needs. See `_read_sX`'s
    docstring for the file format.

    Returns dict with `data`: (num_kpts, nntot, num_bands, num_bands, 3)
    complex128, dimensionless (Pauli +-1 convention).
    """
    return _read_sX(path, "sIu")


def read_dmn(path: str | Path, num_wann: int | None = None) -> dict:
    """
    Parse a Wannier90 .dmn file (written by pw2wannier90 with
    write_dmn=.true., paired with `site_symmetry = .true.` in the .win):
    the crystal point-group symmetry data wannier90's symmetry-adapted
    Wannier function (SAWF) mode needs (R. Sakuma, PRB 87, 235109 2013)
    -- representation matrices of every symmetry operation in both the
    raw ab-initio band basis and the target Wannier-function basis, plus
    the k-point mapping under each operation and the irreducible k-set.

    File layout: formatted (ASCII) Fortran list-directed I/O -- a header
    line, then (num_bands, nsymmetry, nkptirr, num_kpts), then ik2ir
    (num_kpts ints), ir2ik (nkptirr ints), kptsym (nsymmetry x nkptirr
    ints, nsymmetry fastest), d_matrix_wann (num_wann x num_wann x
    nsymmetry x nkptirr complex, first index fastest), d_matrix_band
    (num_bands x num_bands x nsymmetry x nkptirr complex, first index
    fastest). Complex numbers are Fortran's default list-directed
    "(re,im)" tuple format, not bare "re im" pairs like .mmn/.amn.
    Fortran list-directed I/O ignores newlines entirely, so this parser
    works on the whole remaining text, not line-by-line.

    `num_wann` is not stored in the file (wannier90's own reader takes it
    as a separate argument, from the .win's `num_wann`); when omitted here
    it is recovered from the total complex-value count, since
    len(d_matrix_wann) + len(d_matrix_band) and num_bands/nsymmetry/nkptirr
    are all known, leaving num_wann as the only unknown -- pass it
    explicitly only if you want an extra consistency check.

    Returns a dict with:
      num_bands, nsymmetry, nkptirr, num_kpts, num_wann : int
      ik2ir       : (num_kpts,) int, 0-based irreducible-rep index of each k
      ir2ik       : (nkptirr,) int, 0-based full-BZ k-index of each irreducible rep
      kptsym      : (nsymmetry, nkptirr) int, 0-based full-BZ k-index that
                    symmetry `isym` maps irreducible rep `ir`'s k-point to
      d_matrix_wann : (num_wann, num_wann, nsymmetry, nkptirr) complex128
      d_matrix_band : (num_bands, num_bands, nsymmetry, nkptirr) complex128
    """
    path = Path(path)
    text = _read_text(path)
    lines = text.splitlines()

    nb, nsym, nkptirr, nk = (int(x) for x in lines[1].split()[:4])

    rest = "\n".join(lines[2:])
    first_paren = rest.find("(")
    int_region = rest[:first_paren]
    complex_region = rest[first_paren:]

    ints = [int(x) for x in int_region.split()]
    n_ik2ir, n_ir2ik, n_kptsym = nk, nkptirr, nsym * nkptirr
    if len(ints) != n_ik2ir + n_ir2ik + n_kptsym:
        raise ValueError(
            f"{path}: expected {n_ik2ir + n_ir2ik + n_kptsym} integers "
            f"(ik2ir+ir2ik+kptsym), got {len(ints)}"
        )
    ik2ir = np.array(ints[:n_ik2ir], dtype=np.int64) - 1
    ir2ik = np.array(ints[n_ik2ir:n_ik2ir + n_ir2ik], dtype=np.int64) - 1
    kptsym = (np.array(ints[n_ik2ir + n_ir2ik:], dtype=np.int64) - 1
             ).reshape(nkptirr, nsym).T   # (nsym, nkptirr)

    pairs = re.findall(
        r'\(\s*([+\-0-9.eEdD]+)\s*,\s*([+\-0-9.eEdD]+)\s*\)', complex_region,
    )
    n_band_vals = nb * nb * nsym * nkptirr
    if num_wann is None:
        if (len(pairs) - n_band_vals) % (nsym * nkptirr) != 0:
            raise ValueError(f"{path}: cannot infer num_wann from {len(pairs)} complex values")
        nw2 = (len(pairs) - n_band_vals) // (nsym * nkptirr)
        nw = int(round(nw2 ** 0.5))
        if nw * nw != nw2:
            raise ValueError(f"{path}: inferred num_wann^2={nw2} is not a perfect square")
    else:
        nw = num_wann
    n_wann_vals = nw * nw * nsym * nkptirr
    if len(pairs) != n_wann_vals + n_band_vals:
        raise ValueError(
            f"{path}: expected {n_wann_vals + n_band_vals} complex values "
            f"(d_matrix_wann+d_matrix_band), got {len(pairs)}"
        )

    def _to_complex(re_im_pairs):
        re_s = np.array([p[0].replace("D", "E").replace("d", "e") for p in re_im_pairs],
                        dtype=np.float64)
        im_s = np.array([p[1].replace("D", "E").replace("d", "e") for p in re_im_pairs],
                        dtype=np.float64)
        return re_s + 1j * im_s

    flat_wann = _to_complex(pairs[:n_wann_vals])
    flat_band = _to_complex(pairs[n_wann_vals:])

    # Fortran storage order (first index fastest): reshape with dims
    # reversed, then transpose back -- same trick as read_uHu.
    d_matrix_wann = flat_wann.reshape(nkptirr, nsym, nw, nw).transpose(3, 2, 1, 0)
    d_matrix_band = flat_band.reshape(nkptirr, nsym, nb, nb).transpose(3, 2, 1, 0)

    return {
        "num_bands": nb, "nsymmetry": nsym, "nkptirr": nkptirr, "num_kpts": nk, "num_wann": nw,
        "ik2ir": ik2ir, "ir2ik": ir2ik, "kptsym": kptsym,
        "d_matrix_wann": d_matrix_wann, "d_matrix_band": d_matrix_band,
    }


# ---------------------------------------------------------------------------
# .sym writer (pw2wannier90's read_sym=.true. input)
# ---------------------------------------------------------------------------

def write_sym(
    path: str | Path,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> None:
    """
    Write a pw2wannier90 ``.sym`` file (read when ``read_sym = .true.`` in the
    ``&inputpp`` namelist): an explicit, caller-supplied list of point-group
    operations, overriding pw2wannier90's own symmetry auto-detection off the
    NSCF's crystal structure.

    Needed whenever the desired Wannier centre (an analytic projection site)
    is a fixed point of some subgroup of the crystal's full symmetry group
    but not the full group itself -- passing the reduced set here is what
    makes ``site_symmetry=.true.``/``write_dmn`` treat that reduced group as
    "the" symmetry, so the resulting `.dmn` correctly reflects only the
    operations the site actually respects.

    File format (QE's `pw2wannier90.f90::compute_dmn`, `read_sym` branch):
    a header line with the symmetry count, then per operation a bare
    list-directed placeholder record (read via a variable-less
    `read(iun,*)`, hence any single record works -- written here as a
    blank line) followed by a list-directed read of the flattened
    `sr(:,:,isym)` (Fortran array, column-major: what looks like "row i"
    in the 3-line block is `sr`'s i-th column) then `tvec(:,isym)`, 12
    values total, wrapped at 3 per line. Fortran list-directed I/O only
    cares about value order, not which "row" is textually a row vs a
    column, so the row/column labeling of `rotations` here is irrelevant
    for correctness as long as it's used consistently.

    Args:
      rotations   : (nsym, 3, 3) real, in the crystal (fractional, primitive-
                    cell) basis -- the same convention `kpoints`/`atoms_frac`
                    use elsewhere in this codebase, not Cartesian.
      translations: (nsym, 3) real, fractional.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    nsym = rotations.shape[0]
    if rotations.shape != (nsym, 3, 3) or translations.shape != (nsym, 3):
        raise ValueError(
            f"write_sym: expected rotations (nsym,3,3) and translations "
            f"(nsym,3), got {rotations.shape} and {translations.shape}"
        )

    def _row(v):
        return "".join(f"{x:23.15E}" for x in v)

    lines = [f"{nsym:5d}", ""]
    for isym in range(nsym):
        for row in rotations[isym]:
            lines.append(_row(row))
        lines.append(_row(translations[isym]))
        if isym != nsym - 1:
            lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# _hr.dat writer
# ---------------------------------------------------------------------------

def write_hr(
    path: str | Path,
    H_R: np.ndarray,
    R_vectors: np.ndarray,
    degen: np.ndarray,
    nw: int,
    seedname: str = "waw",
) -> None:
    """
    Write a Wannier90-compatible *_hr.dat file.

    Args:
      H_R      : (nR, nw, nw) complex128  Hamiltonian matrix elements H(R)
      R_vectors: (nR, 3)      int64       lattice vectors R in direct coords
      degen    : (nR,)        int64       degeneracy of each R (Wigner-Seitz)
      nw       : int          number of Wannier functions
      seedname : str          label written in the header comment
    """
    path = Path(path)
    nR   = len(R_vectors)

    with _open(path, "w") as f:
        # header
        f.write(f" written by {seedname}\n")
        f.write(f"          {nw}\n")
        f.write(f"          {nR}\n")

        # degeneracy array, 15 entries per line (W90 convention)
        for i, d in enumerate(degen):
            f.write(f"    {d:5d}")
            if (i + 1) % 15 == 0:
                f.write("\n")
        if nR % 15 != 0:
            f.write("\n")

        # matrix elements: R1 R2 R3  m  n  Re(H)  Im(H)
        for iR, R in enumerate(R_vectors):
            for m in range(nw):
                for n in range(nw):
                    h = H_R[iR, m, n]
                    f.write(
                        f"  {R[0]:5d}  {R[1]:5d}  {R[2]:5d}"
                        f"  {m+1:5d}  {n+1:5d}"
                        f"  {h.real:18.10f}  {h.imag:18.10f}\n"
                    )


def read_hr(path: str | Path) -> dict:
    """
    Read a Wannier90 *_hr.dat file (inverse of `write_hr`).

    Returns dict with H_R (nR, nw, nw) complex128 in the file's own units
    (eV for wannier90 output), R_vectors (nR, 3) int64, degen (nR,) int64,
    nw.
    """
    with _open(path) as f:
        f.readline()
        nw = int(f.readline())
        nR = int(f.readline())
        degen = []
        while len(degen) < nR:
            degen += [int(x) for x in f.readline().split()]
        degen = np.array(degen, dtype=np.int64)
        data = np.loadtxt(f)
    data = data.reshape(nR, nw * nw, 7)
    R_vectors = data[:, 0, :3].astype(np.int64)
    m = data[:, :, 3].astype(int) - 1
    n = data[:, :, 4].astype(int) - 1
    H_R = np.zeros((nR, nw, nw), dtype=np.complex128)
    H_R[np.arange(nR)[:, None], m, n] = data[:, :, 5] + 1j * data[:, :, 6]
    return {"H_R": H_R, "R_vectors": R_vectors, "degen": degen, "nw": nw}


# ---------------------------------------------------------------------------
# .chk.fmt writer / reader
#
# Wannier90's formatted checkpoint file (seedname.chk.fmt), read by
# `w90chk2chk.x -import seedname` to produce the binary seedname.chk that
# consumers such as EPW (epw.x with wannierize = .false.) expect. Units:
# all lengths/spreads in Angstrom (Angstrom^2 for spreads/omega_invariant),
# reciprocal lattice in Angstrom^-1, k-points in crystal (fractional)
# coordinates. The reader on the Fortran side uses list-directed READ
# (format '*'), which is whitespace/newline-agnostic, so only value order
# and count matter here.
# ---------------------------------------------------------------------------

def write_chk_fmt(
    path: str | Path,
    *,
    num_bands: int,
    exclude_bands: np.ndarray,
    real_lattice: np.ndarray,
    recip_lattice: np.ndarray,
    mp_grid: tuple[int, int, int],
    kpt_latt: np.ndarray,
    nntot: int,
    num_wann: int,
    have_disentangled: bool,
    omega_invariant: float,
    lwindow: np.ndarray | None,
    ndimwin: np.ndarray | None,
    u_matrix_opt: np.ndarray | None,
    u_matrix: np.ndarray,
    m_matrix: np.ndarray,
    wannier_centres: np.ndarray,
    wannier_spreads: np.ndarray,
    checkpoint: str = "postwann",
    header: str = "written by waw",
) -> None:
    """
    Write a Wannier90-compatible formatted checkpoint file (*.chk.fmt).

    Args (shapes; nk = num_kpts, derived from kpt_latt):
      num_bands       : int    total bands fed into Wannierization
      exclude_bands   : (n_excl,) int64   1-based excluded band indices (may be empty)
      real_lattice    : (3, 3) float64   rows = a1, a2, a3, in Angstrom
      recip_lattice   : (3, 3) float64   rows = b1, b2, b3, in Angstrom^-1
      mp_grid         : (3,)   int       Monkhorst-Pack grid dimensions
      kpt_latt        : (nk, 3) float64  k-points in crystal coordinates
      nntot           : int    number of b-vector neighbours per k-point
      num_wann        : int    number of Wannier functions
      have_disentangled: bool  whether a disentanglement step was performed
      omega_invariant : float  Omega_I in Angstrom^2 (ignored if not disentangled)
      lwindow         : (nk, num_bands) bool or None   band-in-outer-window mask
      ndimwin         : (nk,) int or None               bands in outer window per k
      u_matrix_opt    : (nk, num_bands, num_wann) complex128 or None  (V matrices)
      u_matrix        : (nk, num_wann, num_wann) complex128   final gauge U(k)
      m_matrix        : (nk, nntot, num_wann, num_wann) complex128   rotated overlaps
      wannier_centres : (num_wann, 3) float64   in Angstrom
      wannier_spreads : (num_wann,)   float64   in Angstrom^2
      checkpoint      : str    'postdis' or 'postwann' (default: 'postwann')
      header          : str    free-form comment line (truncated/padded to 33 chars
                                on the Fortran side by an 'A33' read; content
                                otherwise unused)

    Only the values and their order matter (see module note above); this
    writer is deliberately generous with whitespace/newlines.
    """
    nk = kpt_latt.shape[0]
    exclude_bands = np.asarray(exclude_bands, dtype=np.int64).reshape(-1)

    def _write_reals(f, values) -> None:
        f.write(" ".join(f"{v:.15e}" for v in np.asarray(values).reshape(-1)))
        f.write("\n")

    def _write_ints(f, values) -> None:
        f.write(" ".join(str(int(v)) for v in np.asarray(values).reshape(-1)))
        f.write("\n")

    def _write_complex_fortran_order(f, M: np.ndarray) -> None:
        # M: (n_i, n_j) — Fortran writes i (first axis) fastest, so
        # flatten in column-major ('F') order, not numpy's default 'C'.
        flat = M.reshape(-1, order="F")
        vals = np.empty(flat.size * 2, dtype=np.float64)
        vals[0::2] = flat.real
        vals[1::2] = flat.imag
        _write_reals(f, vals)

    with _open(path, "w") as f:
        f.write(f"{header}\n")
        _write_ints(f, [num_bands])
        _write_ints(f, [exclude_bands.size])
        if exclude_bands.size:
            _write_ints(f, exclude_bands)
        # w90's write loop is ((M(i,j), i=1,3), j=1,3) with i = vector index,
        # j = Cartesian component (component outer, vector inner) -- the
        # opposite grouping from kpt_latt/wannier_centres below, so this
        # needs an explicit transpose before the default C-order flatten.
        _write_reals(f, real_lattice.T)   # rows a1,a2,a3
        _write_reals(f, recip_lattice.T)  # rows b1,b2,b3
        _write_ints(f, [nk])
        _write_ints(f, mp_grid)
        _write_reals(f, kpt_latt)         # (nk,3) row-major == w90's per-k order
        _write_ints(f, [nntot])
        _write_ints(f, [num_wann])
        f.write(f"{checkpoint}\n")
        _write_ints(f, [1 if have_disentangled else 0])

        if have_disentangled:
            _write_reals(f, [omega_invariant])
            _write_ints(f, np.asarray(lwindow, dtype=bool).astype(np.int64))  # (nk, nb) row-major
            _write_ints(f, ndimwin)
            for ik in range(nk):
                _write_complex_fortran_order(f, u_matrix_opt[ik])

        for ik in range(nk):
            _write_complex_fortran_order(f, u_matrix[ik])

        for ik in range(nk):
            for ib in range(nntot):
                _write_complex_fortran_order(f, m_matrix[ik, ib])

        _write_reals(f, wannier_centres)  # (nw,3) row-major == w90's per-WF order
        _write_reals(f, wannier_spreads)


def read_chk_fmt(path: str | Path) -> dict:
    """
    Parse a formatted Wannier90 checkpoint file (*.chk.fmt) written by
    write_chk_fmt (or by wannier90's own w90chk2chk.x -export).

    Used mainly to round-trip-verify write_chk_fmt's output.

    Returns a dict with keys matching write_chk_fmt's parameters.
    """
    path = Path(path)
    tokens = _read_text(path).split("\n", 1)
    header, rest = tokens[0], tokens[1]
    vals = rest.split()
    pos = 0

    def next_ints(n):
        nonlocal pos
        out = [int(x) for x in vals[pos:pos + n]]
        pos += n
        return out

    def next_floats(n):
        nonlocal pos
        out = np.array([float(x) for x in vals[pos:pos + n]], dtype=np.float64)
        pos += n
        return out

    def next_complex(n):
        nonlocal pos
        raw = next_floats(2 * n)
        return raw[0::2] + 1j * raw[1::2]

    num_bands = next_ints(1)[0]
    num_excl = next_ints(1)[0]
    exclude_bands = np.array(next_ints(num_excl), dtype=np.int64)
    real_lattice = next_floats(9).reshape(3, 3).T    # see write-side transpose note above
    recip_lattice = next_floats(9).reshape(3, 3).T
    nk = next_ints(1)[0]
    mp_grid = tuple(next_ints(3))
    kpt_latt = next_floats(3 * nk).reshape(nk, 3)
    nntot = next_ints(1)[0]
    num_wann = next_ints(1)[0]
    checkpoint = vals[pos]; pos += 1
    have_disentangled = bool(next_ints(1)[0])

    omega_invariant = None
    lwindow = None
    ndimwin = None
    u_matrix_opt = None
    if have_disentangled:
        omega_invariant = next_floats(1)[0]
        lwindow = np.array(next_ints(nk * num_bands), dtype=bool).reshape(nk, num_bands)
        ndimwin = np.array(next_ints(nk), dtype=np.int64)
        u_matrix_opt = np.stack([
            next_complex(num_bands * num_wann).reshape(num_wann, num_bands).T
            for _ in range(nk)
        ])

    u_matrix = np.stack([
        next_complex(num_wann * num_wann).reshape(num_wann, num_wann).T
        for _ in range(nk)
    ])
    m_matrix = np.stack([
        np.stack([
            next_complex(num_wann * num_wann).reshape(num_wann, num_wann).T
            for _ in range(nntot)
        ])
        for _ in range(nk)
    ])
    wannier_centres = next_floats(3 * num_wann).reshape(num_wann, 3)
    wannier_spreads = next_floats(num_wann)

    return dict(
        header=header, num_bands=num_bands, exclude_bands=exclude_bands,
        real_lattice=real_lattice, recip_lattice=recip_lattice, mp_grid=mp_grid,
        kpt_latt=kpt_latt, nntot=nntot, num_wann=num_wann, checkpoint=checkpoint,
        have_disentangled=have_disentangled, omega_invariant=omega_invariant,
        lwindow=lwindow, ndimwin=ndimwin, u_matrix_opt=u_matrix_opt,
        u_matrix=u_matrix, m_matrix=m_matrix,
        wannier_centres=wannier_centres, wannier_spreads=wannier_spreads,
    )


# ---------------------------------------------------------------------------
# _centres.xyz writer
# ---------------------------------------------------------------------------

def write_centres(
    path: str | Path,
    centers: np.ndarray,
    spreads: np.ndarray,
) -> None:
    """
    Write a Wannier90-compatible *_centres.xyz file.

    Args:
      centers: (nw, 3) float64  Wannier centres in Angstrom
      spreads: (nw,)   float64  spread Omega_n of each WF in Angstrom^2
    """
    path = Path(path)
    nw   = len(spreads)

    with _open(path, "w") as f:
        f.write(f" {nw}\n")
        f.write(f" Wannier centres and spreads\n")
        for n in range(nw):
            cx, cy, cz = centers[n]
            f.write(
                f" X  {cx:15.8f}  {cy:15.8f}  {cz:15.8f}"
                f"   spread= {spreads[n]:15.8f}\n"
            )


def write_nnkp(
    path: str | Path,
    real_lattice: np.ndarray,
    recip_lattice: np.ndarray,
    kpoints: np.ndarray,
    nnkpts: np.ndarray,
    g_vectors: np.ndarray,
    *,
    num_proj: int = 0,
    projections=None,
    auto_projections: int | None = None,
    exclude_bands=(),
    spinors: bool = False,
) -> None:
    """
    Write a Wannier90 .nnkp file (the output of ``wannier90.x -pp``), the input
    ``pw2wannier90.x`` and friends read to know which overlaps to compute.

    Lets waw generate the neighbour topology itself (see
    ``core.kmesh.generate_nnkp``) instead of shelling out to ``wannier90.x -pp``.

    Args:
      real_lattice : (3, 3) rows a1,a2,a3 in **Angstrom**
      recip_lattice: (3, 3) rows b1,b2,b3 in **Angstrom^-1** (2π convention)
      kpoints      : (nk, 3)        k-points in crystal coordinates
      nnkpts       : (nk, nntot)    neighbour k-index table (0-based; written 1-based)
      g_vectors    : (nk, nntot, 3) folding G-vectors
      num_proj     : number of explicit projections to declare (0 for an SCDM
                     run, which builds A_mn from an energy window instead).
                     Ignored when `projections` is given.
      projections  : optional list of analytic projection specs; each is
                     ``(centre_frac(3), l, mr, r, zaxis(3), xaxis(3), zona)``
                     -- the two-line-per-projection Wannier90 nnkp format that
                     ``pw2wannier90.x`` reads to build A_mn from trial orbitals.
                     See `interfaces.projections.spd_projections` for the
                     common s/p/d generator.
      auto_projections : num_wann for an SCDM / automatic-projection run. When
                     given, an ``auto_projections`` block is emitted (with a 0
                     projections block) -- ``pw2wannier90.x`` REQUIRES this
                     block when ``scdm_proj = .true.``.
      exclude_bands: iterable of 1-based band indices to exclude
      spinors      : set True for a noncollinear / spin-orbit (spinor) run.
                     pw2wannier90 then requires a ``spinor_projections`` block
                     (a plain ``projections`` block makes it abort with
                     "Spinorbit without spinor=T"); with SCDM this block is
                     empty and ``auto_projections`` supplies num_wann.
    """
    path = Path(path)
    kpoints  = np.asarray(kpoints, dtype=np.float64)
    nnkpts   = np.asarray(nnkpts, dtype=np.int64)
    g_vectors = np.asarray(g_vectors, dtype=np.int64)
    nk, nntot = nnkpts.shape
    exclude_bands = list(exclude_bands)

    def _block(f, name, body_lines):
        f.write(f"begin {name}\n")
        for line in body_lines:
            f.write(line + "\n")
        f.write(f"end {name}\n\n")

    with _open(path, "w") as f:
        f.write("Written by waw\n\n")
        f.write("calc_only_A  :  F\n\n")

        _block(f, "real_lattice",
                [f"  {r[0]:12.7f}  {r[1]:12.7f}  {r[2]:12.7f}" for r in real_lattice])
        _block(f, "recip_lattice",
                [f"  {r[0]:12.7f}  {r[1]:12.7f}  {r[2]:12.7f}" for r in recip_lattice])
        _block(f, "kpoints",
                [f"    {nk}"] +
                [f"  {k[0]:14.8f}  {k[1]:14.8f}  {k[2]:14.8f}" for k in kpoints])
        # projections: explicit analytic trial orbitals, or a bare count
        # (0 for SCDM, whose A_mn comes from the energy window instead).
        # For spinor runs the block must be named `spinor_projections`, and
        # each spatial projection is duplicated into two spin-resolved
        # entries -- spin_eig=+1 then -1, quantization axis (0,0,1) -- with
        # a third `spin_eig, s_qaxis` line per entry. pw2wannier90.f90 sets
        # `n_wannier = n_proj` directly from this block's count, so without
        # doubling num_wann would silently come out as `len(projections)`
        # instead of `2*len(projections)`.
        proj_block = "spinor_projections" if spinors else "projections"
        if projections:
            n_entries = 2 * len(projections) if spinors else len(projections)
            plines = [f"    {n_entries}"]
            spins = (1, -1) if spinors else (None,)
            for (c, l, mr, r, zax, xax, zona) in projections:
                for s in spins:
                    plines.append(f"  {c[0]:11.7f} {c[1]:11.7f} {c[2]:11.7f}"
                                  f"  {int(l):2d} {int(mr):2d} {int(r):2d}")
                    plines.append(f"  {zax[0]:11.7f} {zax[1]:11.7f} {zax[2]:11.7f}"
                                  f"  {xax[0]:11.7f} {xax[1]:11.7f} {xax[2]:11.7f}"
                                  f"  {zona:8.3f}")
                    if spinors:
                        plines.append(f"  {s:2d}  {0.0:11.7f} {0.0:11.7f} {1.0:11.7f}")
            _block(f, proj_block, plines)
        else:
            _block(f, proj_block, [f"    {num_proj}"])
        # auto_projections: pw2wannier90 requires this block for scdm_proj = .true.
        if auto_projections is not None:
            _block(f, "auto_projections", [f"    {int(auto_projections)}", "    0"])

        nn_lines = [f"    {nntot}"]
        for ik in range(nk):
            for ib in range(nntot):
                g = g_vectors[ik, ib]
                nn_lines.append(
                    f"  {ik + 1:5d}  {nnkpts[ik, ib] + 1:5d}"
                    f"    {g[0]:3d} {g[1]:3d} {g[2]:3d}"
                )
        _block(f, "nnkpts", nn_lines)

        _block(f, "exclude_bands",
                [f"    {len(exclude_bands)}"] + [f"  {b}" for b in exclude_bands])


def write_win(
    path: str | Path,
    lattice: np.ndarray,
    frac_positions: np.ndarray,
    symbols,
    mp_grid,
    kpts: np.ndarray,
    *,
    num_wann: int,
    num_bands: int,
) -> Path:
    """
    Write a minimal Wannier90 .win file (the counterpart of ``read_win``).

    Unit-agnostic like the other writers here: the lattice is written verbatim
    under an ``Ang`` tag, so pass it in **Angstrom**.  No ``projections`` block
    is written -- an SCDM run builds A_mn from an energy window rather than
    atomic-orbital guesses, keeping this material-agnostic.

    Args:
      lattice       : (3, 3) rows a1,a2,a3 in **Angstrom**
      frac_positions: (n_atoms, 3) atomic positions in crystal coordinates
      symbols       : length-n_atoms iterable of chemical symbols
      mp_grid       : (N1, N2, N3) Monkhorst-Pack grid
      kpts          : (nk, 3) k-points in crystal coordinates
      num_wann      : number of Wannier functions
      num_bands     : number of bands
    """
    path    = Path(path)
    lattice = np.asarray(lattice, dtype=np.float64)
    frac    = np.asarray(frac_positions, dtype=np.float64)
    symbols = list(symbols)
    kpts    = np.asarray(kpts, dtype=np.float64)

    lines = [
        f"num_wann   = {num_wann}",
        f"num_bands  = {num_bands}",
        f"mp_grid = {mp_grid[0]} {mp_grid[1]} {mp_grid[2]}",
        "",
        "begin unit_cell_cart",
        "Ang",
        *(f" {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}" for v in lattice),
        "end unit_cell_cart",
        "",
        "begin atoms_frac",
        *(f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}" for s, p in zip(symbols, frac)),
        "end atoms_frac",
        "",
        "begin kpoints",
        *(f"  {k[0]:.8f} {k[1]:.8f} {k[2]:.8f}" for k in kpts),
        "end kpoints",
    ]
    with _open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
