import io
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis

try:
    import mdtraj as md
except Exception:
    md = None

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception:
    PCA = None
    StandardScaler = None

try:
    import py3Dmol
except Exception:
    py3Dmol = None

st.set_page_config(
    page_title="Protein Folding Research Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWVY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
AROMATIC = set("FWY")
POLAR = set("STNQCY")
SMALL = set("GASPV")

MOTIF_PATTERNS = {
    "N-glycosylation (N[^P][ST])": r"N[^P][ST]",
    "Casein kinase II": r"[ST]..[DE]",
    "Protein kinase C": r"[ST].{2}[RK]",
    "Tyrosine kinase": r"[RK].{2}[DE]",
    "Leucine zipper": r"(?:[LIVM].{6}){2,}",
    "ATP/GTP-binding P-loop": r"G....GKS",
    "Zinc finger C2H2": r"C.{2,4}C.{12}H.{3,5}H",
    "SH3 proline-rich": r"P.{2}P",
}


def clean_sequence(text: str) -> str:
    text = re.sub(r"\s+", "", str(text).upper())
    return "".join(c for c in text if c in AA20)


def read_fasta_text(text: str) -> List[SeqRecord]:
    records = list(SeqIO.parse(io.StringIO(text), "fasta"))
    for rec in records:
        rec.seq = rec.seq.__class__(clean_sequence(str(rec.seq)))
    return records


def seq_from_uploaded_file(uploaded) -> str:
    raw = uploaded.getvalue().decode("utf-8", errors="ignore")
    if raw.lstrip().startswith(">"):
        recs = read_fasta_text(raw)
        if recs:
            return str(recs[0].seq)
    return clean_sequence(raw)


@st.cache_data(show_spinner=False)
def analyze_sequence(seq: str) -> Dict[str, float]:
    if not seq:
        return {}
    pa = ProteinAnalysis(seq)
    try:
        mw = molecular_weight(seq, seq_type="protein")
    except Exception:
        mw = float("nan")
    try:
        pI = pa.isoelectric_point()
    except Exception:
        pI = float("nan")
    try:
        gravy = pa.gravy()
    except Exception:
        gravy = float("nan")
    try:
        instability = pa.instability_index()
    except Exception:
        instability = float("nan")

    aromaticity = sum(a in AROMATIC for a in seq) / len(seq)
    hydrophobic = sum(a in HYDROPHOBIC for a in seq) / len(seq)
    charged = sum(a in (POSITIVE | NEGATIVE) for a in seq) / len(seq)
    polar = sum(a in POLAR for a in seq) / len(seq)
    small = sum(a in SMALL for a in seq) / len(seq)
    cys = seq.count("C")
    gly = seq.count("G")
    pro = seq.count("P")
    aliphatic = 100 * (seq.count("A") + 2.9 * seq.count("V") + 3.9 * (seq.count("I") + seq.count("L"))) / len(seq)
    net_charge_7 = (seq.count("K") + seq.count("R") + 0.1 * seq.count("H")) - (seq.count("D") + seq.count("E"))

    return {
        "length": len(seq),
        "molecular_weight": mw,
        "isoelectric_point": pI,
        "gravy": gravy,
        "instability_index": instability,
        "aliphatic_index": aliphatic,
        "aromatic_fraction": aromaticity,
        "hydrophobic_fraction": hydrophobic,
        "charged_fraction": charged,
        "polar_fraction": polar,
        "small_fraction": small,
        "cys_count": cys,
        "gly_count": gly,
        "pro_count": pro,
        "net_charge_proxy_pH7": net_charge_7,
    }


@st.cache_data(show_spinner=False)
def residue_composition(seq: str) -> pd.DataFrame:
    counts = {aa: seq.count(aa) for aa in sorted(AA20)}
    df = pd.DataFrame({"AA": list(counts.keys()), "Count": list(counts.values())})
    df["Fraction"] = df["Count"] / max(len(seq), 1)
    return df


@st.cache_data(show_spinner=False)
def sliding_window_profile(seq: str, window: int = 7, mode: str = "hydrophobicity") -> pd.DataFrame:
    if len(seq) == 0:
        return pd.DataFrame(columns=["position", "value"])
    window = max(1, min(window, len(seq)))
    values, centers = [], []
    for i in range(len(seq) - window + 1):
        chunk = seq[i : i + window]
        if mode == "hydrophobicity":
            val = sum(a in HYDROPHOBIC for a in chunk) / window
        elif mode == "charge":
            val = (sum(a in POSITIVE for a in chunk) - sum(a in NEGATIVE for a in chunk)) / window
        elif mode == "polar":
            val = sum(a in POLAR for a in chunk) / window
        else:
            val = sum(a in SMALL for a in chunk) / window
        values.append(val)
        centers.append(i + window // 2 + 1)
    return pd.DataFrame({"position": centers, "value": values})


@st.cache_data(show_spinner=False)
def motif_scan(seq: str) -> pd.DataFrame:
    rows = []
    for name, pattern in MOTIF_PATTERNS.items():
        for m in re.finditer(pattern, seq):
            rows.append({"motif": name, "start": m.start() + 1, "end": m.end(), "match": m.group(0)})
    return pd.DataFrame(rows)


def protein_profile_plot(df: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["position"], y=df["value"], mode="lines", line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title="Residue", yaxis_title=y_label, height=350)
    return fig


def composition_plot(df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.bar(df, x="AA", y="Count", text="Count", title=title)
    fig.update_layout(height=350)
    return fig


@st.cache_data(show_spinner=False)
def read_msa_text(text: str) -> List[str]:
    recs = [clean_sequence(str(r.seq)) for r in SeqIO.parse(io.StringIO(text), "fasta")]
    return [r for r in recs if r]


@st.cache_data(show_spinner=False)
def msa_conservation_from_text(text: str) -> pd.DataFrame:
    seqs = read_msa_text(text)
    if len(seqs) < 2:
        return pd.DataFrame(columns=["position", "conservation", "entropy"])
    min_len = min(len(s) for s in seqs)
    rows = []
    for i in range(min_len):
        col = [s[i] for s in seqs if i < len(s)]
        freq = pd.Series(col).value_counts(normalize=True)
        conservation = float(freq.max())
        entropy = float(-(freq * np.log2(freq + 1e-12)).sum())
        rows.append({"position": i + 1, "conservation": conservation, "entropy": entropy})
    return pd.DataFrame(rows)


def show_structure_viewer(pdb_text: str):
    if py3Dmol is None:
        st.info("Install py3Dmol for an embedded 3D viewer.")
        st.code(pdb_text[:4000])
        return
    view = py3Dmol.view(width=900, height=600)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addStyle({"hetflag": True}, {"stick": {}})
    view.zoomTo()
    st.components.v1.html(view._make_html(), height=620, scrolling=False)


def load_traj_with_topology(traj_bytes: bytes, traj_name: str, top_path: str):
    if md is None:
        raise RuntimeError("MDTraj is not installed.")
    traj_path = f"/tmp/{traj_name}"
    with open(traj_path, "wb") as f:
        f.write(traj_bytes)
    return md.load(traj_path, top=top_path)


def residue_contact_map(traj, cutoff_nm: float = 0.8) -> pd.DataFrame:
    ca = traj.topology.select("name CA")
    if len(ca) < 2:
        return pd.DataFrame()
    pairs = np.array([(i, j) for idx, i in enumerate(ca) for j in ca[idx + 1 :]])
    if len(pairs) == 0:
        return pd.DataFrame()
    dists = md.compute_distances(traj, pairs)
    contact = (dists < cutoff_nm).mean(axis=0)
    n = len(ca)
    mat = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = mat[j, i] = contact[k]
            k += 1
    return pd.DataFrame(mat, index=np.arange(1, n + 1), columns=np.arange(1, n + 1))


def cross_correlation(traj) -> pd.DataFrame:
    ca = traj.topology.select("name CA")
    if len(ca) < 3:
        return pd.DataFrame()
    xyz = traj.xyz[:, ca, :].reshape(traj.n_frames, len(ca) * 3)
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    cov = np.cov(xyz, rowvar=False)
    std = np.sqrt(np.diag(cov) + 1e-12)
    corr = cov / np.outer(std, std)
    n = len(ca)
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            block = corr[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3]
            res[i, j] = np.nanmean(block)
    return pd.DataFrame(res, index=np.arange(1, n + 1), columns=np.arange(1, n + 1))


def trajectory_metrics(traj, reference_idx: int = 0) -> Dict[str, np.ndarray]:
    ca = traj.topology.select("name CA")
    if len(ca) == 0:
        return {}
    ref = traj.xyz[reference_idx]
    rmsd_vals = md.rmsd(traj, traj, reference_idx, atom_indices=ca)
    rg = md.compute_rg(traj)
    sasa = md.shrake_rupley(traj, mode="residue").mean(axis=1) if traj.n_residues else None
    rmsf = md.rmsf(traj, ref, atom_indices=ca)
    hbonds = md.baker_hubbard(traj, periodic=False)
    return {"rmsd": rmsd_vals, "rg": rg, "sasa": sasa, "rmsf": rmsf, "hbonds": hbonds, "ca_count": len(ca)}


def pca_projection(traj):
    ca = traj.topology.select("name CA")
    if len(ca) < 3 or PCA is None or StandardScaler is None:
        return None
    X = traj.xyz[:, ca, :].reshape(traj.n_frames, -1)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(3, X.shape[1], X.shape[0]))
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def mutation_scan(seq: str, metric_fn) -> pd.DataFrame:
    rows = []
    base = metric_fn(seq)
    for pos, wt in enumerate(seq, start=1):
        for aa in sorted(AA20):
            if aa == wt:
                continue
            mut = seq[: pos - 1] + aa + seq[pos:]
            try:
                score = metric_fn(mut)
                delta = score - base
            except Exception:
                score = np.nan
                delta = np.nan
            rows.append({"position": pos, "wt": wt, "mut": aa, "score": score, "delta": delta})
    return pd.DataFrame(rows)


def novelty_score(seq: str) -> float:
    df = residue_composition(seq)
    frac = df["Fraction"].replace(0, np.nan).dropna()
    entropy = -np.sum(frac * np.log2(frac))
    return float(entropy / math.log2(20))


def maybe_download_df(df: pd.DataFrame, name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(f"Download {name} CSV", csv, file_name=f"{name}.csv", mime="text/csv")


st.title("🧬 Protein Folding Research Lab")
st.caption("Biopython + MDTraj + Streamlit + advanced research analytics")

with st.sidebar:
    st.header("Inputs")
    seq_file = st.file_uploader("Upload protein FASTA or plain sequence", type=["fasta", "fa", "txt"])
    msa_file = st.file_uploader("Upload aligned FASTA for conservation", type=["fasta", "fa", "aln", "txt"], key="msa")
    pdb_file = st.file_uploader("Upload PDB/mmCIF structure", type=["pdb", "cif", "mmcif"], key="pdb")
    traj_file = st.file_uploader("Upload trajectory file (DCD/XTC/TRR/PDB)", type=["dcd", "xtc", "trr", "pdb"], key="traj")
    cutoff = st.slider("Contact cutoff (nm)", 0.3, 1.5, 0.8, 0.05)
    window = st.slider("Sliding window", 3, 31, 9, 2)
    st.divider()
    st.header("Analysis switches")
    show_mutation = st.checkbox("Mutation scan", value=True)
    show_embeddings = st.checkbox("Latent-space features", value=True)
    show_dynamics = st.checkbox("Trajectory analytics", value=True)

sequence = ""
seq_records: List[SeqRecord] = []
if seq_file is not None:
    sequence = seq_from_uploaded_file(seq_file)
    raw_text = seq_file.getvalue().decode("utf-8", errors="ignore")
    if raw_text.lstrip().startswith(">"):
        seq_records = read_fasta_text(raw_text)
else:
    sequence = clean_sequence(st.text_input("Enter protein sequence", value="MKWVTFISLLLLFSSAYSRGVFRR"))

if sequence:
    metrics = analyze_sequence(sequence)
    comp = residue_composition(sequence)
    hydro = sliding_window_profile(sequence, window=window, mode="hydrophobicity")
    charge = sliding_window_profile(sequence, window=window, mode="charge")
    polar = sliding_window_profile(sequence, window=window, mode="polar")
    motifs = motif_scan(sequence)
    novelty = novelty_score(sequence)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Length", metrics["length"])
    c2.metric("MW", f"{metrics['molecular_weight']:.1f}")
    c3.metric("pI", f"{metrics['isoelectric_point']:.2f}")
    c4.metric("GRAVY", f"{metrics['gravy']:.2f}")
    c5.metric("Instability", f"{metrics['instability_index']:.2f}")
    c6.metric("Novelty", f"{novelty:.3f}")

    tab_overview, tab_seq, tab_msa, tab_structure, tab_traj, tab_mut, tab_ai, tab_export = st.tabs(
        ["Overview", "Sequence", "MSA", "Structure", "Trajectory", "Mutation Lab", "AI/Embedding", "Export"]
    )

    with tab_overview:
        left, right = st.columns([1.1, 0.9])
        with left:
            st.subheader("Composition")
            st.plotly_chart(composition_plot(comp, "Amino Acid Composition"), use_container_width=True)
            st.subheader("Motif hits")
            if len(motifs):
                st.dataframe(motifs, use_container_width=True, height=220)
            else:
                st.info("No built-in motifs matched this sequence.")
        with right:
            st.subheader("Summary table")
            summary_df = pd.DataFrame([metrics | {"novelty_score": novelty}]).T.reset_index()
            summary_df.columns = ["feature", "value"]
            st.dataframe(summary_df, use_container_width=True, height=420)

    with tab_seq:
        st.subheader("Sliding-window physicochemical profiles")
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(protein_profile_plot(hydro, "Hydrophobicity trace", "Hydrophobic fraction"), use_container_width=True)
        c2.plotly_chart(protein_profile_plot(charge, "Net charge trace", "Charge proxy"), use_container_width=True)
        c3.plotly_chart(protein_profile_plot(polar, "Polarity trace", "Polar fraction"), use_container_width=True)
        st.subheader("Composition table")
        st.dataframe(comp, use_container_width=True, height=300)

    with tab_msa:
        st.subheader("Multiple sequence alignment / conservation")
        if msa_file is not None:
            msa_text = msa_file.getvalue().decode("utf-8", errors="ignore")
            cons = msa_conservation_from_text(msa_text)
            if len(cons) >= 2:
                st.plotly_chart(px.line(cons, x="position", y="conservation", title="Conservation from MSA"), use_container_width=True)
                st.plotly_chart(px.line(cons, x="position", y="entropy", title="Sequence entropy from MSA"), use_container_width=True)
                st.dataframe(cons, use_container_width=True, height=240)
            else:
                st.warning("Upload at least two aligned sequences.")
        else:
            st.info("Upload an aligned FASTA to compute conservation and entropy.")
        if seq_records:
            st.subheader("Pairwise alignment")
            rec0 = str(seq_records[0].seq)
            aligner = PairwiseAligner()
            aligner.mode = "global"
            aln = aligner.align(sequence, rec0)
            if aln:
                st.text(str(aln[0]))

    with tab_structure:
        st.subheader("Structure visualization")
        if pdb_file is not None:
            pdb_bytes = pdb_file.getvalue()
            if md is not None:
                try:
                    top_path = f"/tmp/{pdb_file.name}"
                    with open(top_path, "wb") as f:
                        f.write(pdb_bytes)
                    traj = md.load(top_path)
                    st.success(f"Loaded structure with {traj.n_atoms} atoms, {traj.n_residues} residues.")
                    pdb_text = pdb_bytes.decode("utf-8", errors="ignore")
                    show_structure_viewer(pdb_text)
                    if traj.n_residues > 0:
                        ca = traj.topology.select("name CA")
                        if len(ca) >= 2:
                            pairs = np.array([(i, j) for idx, i in enumerate(ca) for j in ca[idx + 1 :]])
                            dists = md.compute_distances(traj, pairs)[0]
                            st.plotly_chart(
                                px.histogram(
                                    pd.DataFrame({"distance_nm": dists}),
                                    x="distance_nm",
                                    nbins=40,
                                    title="Inter-residue distance histogram",
                                ),
                                use_container_width=True,
                            )
                        try:
                            ss = md.compute_dssp(traj)[0]
                            ss_counts = pd.Series(list(ss)).value_counts()
                            st.plotly_chart(px.bar(x=ss_counts.index, y=ss_counts.values, title="Secondary structure distribution"), use_container_width=True)
                        except Exception:
                            st.info("DSSP not available in this environment.")
                except Exception as e:
                    st.error(f"Could not load structure: {e}")
            else:
                st.warning("MDTraj not installed; structure analysis disabled.")
        else:
            st.info("Upload a PDB/mmCIF structure to unlock 3D and distance analytics.")

    with tab_traj:
        st.subheader("Trajectory analytics")
        if show_dynamics and traj_file is not None and pdb_file is not None and md is not None:
            try:
                top_path = f"/tmp/{pdb_file.name}"
                with open(top_path, "wb") as f:
                    f.write(pdb_file.getvalue())
                traj = load_traj_with_topology(traj_file.getvalue(), traj_file.name, top_path)
                st.success(f"Loaded trajectory: {traj.n_frames} frames, {traj.n_atoms} atoms.")
                met = trajectory_metrics(traj)
                cols = st.columns(4)
                cols[0].metric("Frames", traj.n_frames)
                cols[1].metric("CA atoms", met.get("ca_count", 0))
                cols[2].metric("HBonds", len(met.get("hbonds", [])))
                cols[3].metric("Residues", traj.n_residues)

                st.plotly_chart(
                    px.line(x=np.arange(len(met["rmsd"])), y=met["rmsd"], title="RMSD trajectory", labels={"x": "Frame", "y": "RMSD (nm)"}),
                    use_container_width=True,
                )
                st.plotly_chart(
                    px.line(x=np.arange(len(met["rg"])), y=met["rg"], title="Radius of gyration", labels={"x": "Frame", "y": "Rg (nm)"}),
                    use_container_width=True,
                )
                st.plotly_chart(
                    px.line(x=np.arange(1, len(met["rmsf"]) + 1), y=met["rmsf"], title="RMSF per residue", labels={"x": "Residue", "y": "RMSF (nm)"}),
                    use_container_width=True,
                )
                if met.get("sasa") is not None:
                    st.plotly_chart(
                        px.line(x=np.arange(1, len(met["sasa"]) + 1), y=met["sasa"], title="Mean SASA per frame", labels={"x": "Frame", "y": "SASA"}),
                        use_container_width=True,
                    )

                contact = residue_contact_map(traj, cutoff_nm=cutoff)
                if not contact.empty:
                    st.plotly_chart(px.imshow(contact, aspect="auto", title="Residue contact persistence map"), use_container_width=True)
                corr = cross_correlation(traj)
                if not corr.empty:
                    st.plotly_chart(px.imshow(corr, aspect="auto", title="Dynamic cross-correlation matrix"), use_container_width=True)

                pca_out = pca_projection(traj)
                if pca_out is not None:
                    coords, variance = pca_out
                    pca_df = pd.DataFrame(coords[:, :2], columns=["PC1", "PC2"])
                    pca_df["frame"] = np.arange(len(pca_df))
                    st.plotly_chart(
                        px.scatter(pca_df, x="PC1", y="PC2", color="frame", title=f"Conformational PCA (var={variance[:2].sum():.2%})"),
                        use_container_width=True,
                    )
                else:
                    st.info("PCA requires scikit-learn and sufficient frames/residues.")
            except Exception as e:
                st.error(f"Trajectory analysis failed: {e}")
        else:
            st.info("Upload both a topology structure and a trajectory file to analyze dynamics.")

    with tab_mut:
        st.subheader("Mutation laboratory")
        if show_mutation:
            pos = st.number_input("Mutation position", min_value=1, max_value=len(sequence), value=1, step=1)
            mut_aa = st.selectbox("Mutate to", sorted(AA20))
            mut_seq = sequence[: pos - 1] + mut_aa + sequence[pos:]
            colx, coly = st.columns(2)
            colx.code(sequence[max(0, pos - 11) : pos + 10])
            coly.code(mut_seq[max(0, pos - 11) : pos + 10])

            def score_fn(s: str) -> float:
                return analyze_sequence(s)["instability_index"]

            if st.button("Run mutation scan by instability index"):
                scan_df = mutation_scan(sequence, score_fn)
                st.dataframe(scan_df.head(500), use_container_width=True, height=400)
                maybe_download_df(scan_df, "mutation_scan_instability_index")
        else:
            st.caption("Enable mutation scan from the sidebar.")

    with tab_ai:
        st.subheader("AI and latent-space features")
        if show_embeddings and len(sequence) >= 4:
            aa_index = {a: i + 1 for i, a in enumerate(sorted(AA20))}
            vec = np.array([aa_index[a] for a in sequence], dtype=float)
            emb = pd.DataFrame(
                {
                    "position": np.arange(1, len(vec) + 1),
                    "embedding_1": np.sin(vec / 2.0),
                    "embedding_2": np.cos(vec / 3.0),
                    "embedding_3": vec / vec.max(),
                }
            )
            st.plotly_chart(
                px.scatter(emb, x="embedding_1", y="embedding_2", color="embedding_3", title="Sequence latent-space proxy"),
                use_container_width=True,
            )
            st.dataframe(emb.head(100), use_container_width=True, height=240)
            st.caption("Replace this proxy with ESM/ProtT5 embeddings when available.")
        st.subheader("Interpretability")
        importance = pd.DataFrame(
            {
                "position": np.arange(1, len(sequence) + 1),
                "importance": np.array([sequence.count(a) for a in sequence], dtype=float) / max(len(sequence), 1),
            }
        )
        st.plotly_chart(px.line(importance, x="position", y="importance", title="Residue importance proxy"), use_container_width=True)

    with tab_export:
        st.subheader("Export research package")
        report = {**metrics, "novelty_score": novelty, "motif_hits": len(motifs), "sequence": sequence}
        report_df = pd.DataFrame([report])
        st.dataframe(report_df.T.reset_index().rename(columns={"index": "feature", 0: "value"}), use_container_width=True, height=400)
        maybe_download_df(report_df, "protein_summary")
        maybe_download_df(comp, "amino_acid_composition")
        if len(motifs):
            maybe_download_df(motifs, "motif_hits")
else:
    st.warning("Please provide a valid protein sequence.")
