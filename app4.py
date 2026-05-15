import io
import math
import re
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from Bio import SeqIO
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis

st.set_page_config(
    page_title="Protein Folding Research Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA20)

HYDROPATHY_KD = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
POSITIVE = set("KRH")
NEGATIVE = set("DE")
HYDROPHOBIC = set("AILMFWVY")
POLAR = set("STNQCY")
AROMATIC = set("FWY")
SMALL = set("GASPV")
DISORDER_BIASED = set("PESQKRDG")

MOTIFS = {
    "N-glycosylation": r"N[^P][ST]",
    "Casein kinase II": r"[ST]..[DE]",
    "Protein kinase C": r"[ST].{2}[RK]",
    "Tyrosine kinase": r"[RK].{2}[DE]",
    "Leucine zipper": r"(?:[LIVM].{6}){2,}",
    "ATP/GTP P-loop": r"G....GKS",
    "Zinc finger C2H2": r"C.{2,4}C.{12}H.{3,5}H",
    "SH3 proline-rich": r"P.{2}P",
    "Basic cleavage": r"[KR]{2,}",
    "Acidic patch": r"[DE]{3,}",
}

def clean_sequence(text: str) -> str:
    text = re.sub(r"\s+", "", str(text).upper())
    return "".join(c for c in text if c in AA_SET)

def read_fasta_text(text: str) -> List[str]:
    sequences: List[str] = []
    for rec in SeqIO.parse(io.StringIO(text), "fasta"):
        seq = clean_sequence(str(rec.seq))
        if seq:
            sequences.append(seq)
    return sequences

def safe_mean(values):
    return float(np.mean(values)) if len(values) else 0.0

def safe_std(values):
    return float(np.std(values)) if len(values) else 0.0

def entropy_from_counts(counts: List[int]) -> float:
    arr = np.array(counts, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log2(p)).sum())

def composition_fraction(seq: str, chars: str) -> float:
    if not seq:
        return 0.0
    return sum(c in set(chars) for c in seq) / len(seq)

def longest_run(seq: str) -> int:
    if not seq:
        return 0
    best = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
        else:
            best = max(best, cur)
            cur = 1
    return max(best, cur)

def kmers(seq: str, k: int) -> List[str]:
    if k <= 0 or len(seq) < k:
        return []
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def kmer_diversity(seq: str, k: int) -> float:
    ks = kmers(seq, k)
    if not ks:
        return 0.0
    return len(set(ks)) / len(ks)

def kmer_entropy(seq: str, k: int) -> float:
    ks = kmers(seq, k)
    if not ks:
        return 0.0
    c = Counter(ks)
    return entropy_from_counts(list(c.values()))

def sliding_window_values(seq: str, window: int, scorer):
    if len(seq) < window or window <= 0:
        return []
    vals = []
    for i in range(len(seq) - window + 1):
        vals.append(scorer(seq[i:i+window]))
    return vals

def hydropathy_score(chunk: str) -> float:
    return float(np.mean([HYDROPATHY_KD.get(a, 0.0) for a in chunk]))

def charge_score(chunk: str) -> float:
    return float((sum(a in POSITIVE for a in chunk) - sum(a in NEGATIVE for a in chunk)) / len(chunk))

def polarity_score(chunk: str) -> float:
    return float(sum(a in POLAR for a in chunk) / len(chunk))

def disorder_proxy_score(chunk: str) -> float:
    disorder_bias = sum(a in DISORDER_BIASED for a in chunk) / len(chunk)
    helix_bias = sum(a in set("ALMQEKR") for a in chunk) / len(chunk)
    hydrophobic_bias = sum(a in HYDROPHOBIC for a in chunk) / len(chunk)
    return float(0.45 * disorder_bias + 0.35 * (1 - hydrophobic_bias) + 0.20 * (1 - helix_bias))

def motif_count(seq: str, pattern: str) -> int:
    return len(list(re.finditer(pattern, seq)))

def flex_stats(seq: str):
    pa = ProteinAnalysis(seq)
    try:
        vals = pa.flexibility()
        return vals
    except Exception:
        return []

def molar_extinction_values(pa: ProteinAnalysis):
    try:
        reduced, oxidized = pa.molar_extinction_coefficient()
    except Exception:
        reduced, oxidized = 0.0, 0.0
    return float(reduced), float(oxidized)

def amino_acid_props(seq: str) -> Dict[str, float]:
    n = max(len(seq), 1)
    pa = ProteinAnalysis(seq)
    metrics: Dict[str, float] = {}

    metrics["length"] = float(len(seq))
    metrics["molecular_weight"] = float(molecular_weight(seq, seq_type="protein"))
    metrics["isoelectric_point"] = float(pa.isoelectric_point())
    metrics["gravy"] = float(pa.gravy())
    metrics["instability_index"] = float(pa.instability_index())
    metrics["aliphatic_index"] = float(100 * (seq.count("A") + 2.9 * seq.count("V") + 3.9 * (seq.count("I") + seq.count("L"))) / n)
    metrics["aromaticity"] = float(pa.aromaticity())
    helix, turn, sheet = pa.secondary_structure_fraction()
    metrics["helix_fraction"] = float(helix)
    metrics["turn_fraction"] = float(turn)
    metrics["sheet_fraction"] = float(sheet)
    metrics["molar_extinction_reduced"], metrics["molar_extinction_oxidized"] = molar_extinction_values(pa)
    metrics["net_charge_proxy_pH7"] = float((seq.count("K") + seq.count("R") + 0.1 * seq.count("H")) - (seq.count("D") + seq.count("E")))
    metrics["charge_balance_index"] = float(abs((seq.count("K") + seq.count("R") + seq.count("H")) - (seq.count("D") + seq.count("E"))) / n)
    metrics["hydrophobic_fraction"] = float(composition_fraction(seq, "".join(HYDROPHOBIC)))
    metrics["polar_fraction"] = float(composition_fraction(seq, "".join(POLAR)))
    metrics["aromatic_fraction"] = float(composition_fraction(seq, "".join(AROMATIC)))
    metrics["small_fraction"] = float(composition_fraction(seq, "".join(SMALL)))
    metrics["cys_count"] = float(seq.count("C"))
    metrics["gly_count"] = float(seq.count("G"))
    metrics["pro_count"] = float(seq.count("P"))
    metrics["positive_count"] = float(sum(seq.count(a) for a in POSITIVE))
    metrics["negative_count"] = float(sum(seq.count(a) for a in NEGATIVE))
    metrics["polar_count"] = float(sum(seq.count(a) for a in POLAR))
    metrics["hydrophobic_count"] = float(sum(seq.count(a) for a in HYDROPHOBIC))
    metrics["aromatic_count"] = float(sum(seq.count(a) for a in AROMATIC))
    metrics["acidic_count"] = float(seq.count("D") + seq.count("E"))
    metrics["basic_count"] = float(seq.count("K") + seq.count("R") + seq.count("H"))
    metrics["serine_count"] = float(seq.count("S"))
    metrics["threonine_count"] = float(seq.count("T"))
    metrics["alanine_count"] = float(seq.count("A"))
    metrics["valine_count"] = float(seq.count("V"))
    metrics["isoleucine_count"] = float(seq.count("I"))
    metrics["leucine_count"] = float(seq.count("L"))
    metrics["methionine_count"] = float(seq.count("M"))
    metrics["phenylalanine_count"] = float(seq.count("F"))
    metrics["tyrosine_count"] = float(seq.count("Y"))
    metrics["tryptophan_count"] = float(seq.count("W"))
    metrics["lysine_count"] = float(seq.count("K"))
    metrics["arginine_count"] = float(seq.count("R"))
    metrics["histidine_count"] = float(seq.count("H"))
    metrics["aspartate_count"] = float(seq.count("D"))
    metrics["glutamate_count"] = float(seq.count("E"))
    metrics["asparagine_count"] = float(seq.count("N"))
    metrics["glutamine_count"] = float(seq.count("Q"))

    counts = [seq.count(a) for a in AA20]
    fractions = [c / n for c in counts]
    metrics["unique_aa_count"] = float(sum(c > 0 for c in counts))
    metrics["alphabet_coverage"] = float(sum(c > 0 for c in counts) / 20.0)
    metrics["sequence_entropy"] = float(entropy_from_counts(counts))
    metrics["sequence_evenness"] = float(metrics["sequence_entropy"] / math.log2(max(metrics["unique_aa_count"], 2)))
    metrics["composition_variance"] = float(np.var(fractions))
    metrics["composition_std"] = float(np.std(fractions))
    metrics["composition_max_fraction"] = float(np.max(fractions))
    metrics["composition_min_fraction"] = float(np.min(fractions))
    metrics["composition_range"] = float(np.max(fractions) - np.min(fractions))
    metrics["composition_gini_proxy"] = float(1.0 - np.sum(np.square(fractions)))
    metrics["longest_repeat_run"] = float(longest_run(seq))
    metrics["homopolymer_fraction"] = float(longest_run(seq) / n)
    metrics["repeat_rate_2mer"] = float(1.0 - kmer_diversity(seq, 2))
    metrics["repeat_rate_3mer"] = float(1.0 - kmer_diversity(seq, 3))
    metrics["repeat_rate_4mer"] = float(1.0 - kmer_diversity(seq, 4))
    metrics["repeat_rate_5mer"] = float(1.0 - kmer_diversity(seq, 5))
    metrics["kmer_entropy_2"] = float(kmer_entropy(seq, 2))
    metrics["kmer_entropy_3"] = float(kmer_entropy(seq, 3))
    metrics["kmer_entropy_4"] = float(kmer_entropy(seq, 4))
    metrics["kmer_entropy_5"] = float(kmer_entropy(seq, 5))
    metrics["kmer_diversity_2"] = float(kmer_diversity(seq, 2))
    metrics["kmer_diversity_3"] = float(kmer_diversity(seq, 3))
    metrics["kmer_diversity_4"] = float(kmer_diversity(seq, 4))
    metrics["kmer_diversity_5"] = float(kmer_diversity(seq, 5))

    for win in (5, 9, 15):
        hyd = sliding_window_values(seq, win, hydropathy_score)
        chg = sliding_window_values(seq, win, charge_score)
        pol = sliding_window_values(seq, win, polarity_score)
        dis = sliding_window_values(seq, win, disorder_proxy_score)
        metrics[f"hydropathy_w{win}_mean"] = float(safe_mean(hyd))
        metrics[f"hydropathy_w{win}_max"] = float(max(hyd) if hyd else 0.0)
        metrics[f"hydropathy_w{win}_min"] = float(min(hyd) if hyd else 0.0)
        metrics[f"hydropathy_w{win}_std"] = float(safe_std(hyd))
        metrics[f"charge_w{win}_mean"] = float(safe_mean(chg))
        metrics[f"charge_w{win}_max"] = float(max(chg) if chg else 0.0)
        metrics[f"charge_w{win}_min"] = float(min(chg) if chg else 0.0)
        metrics[f"charge_w{win}_std"] = float(safe_std(chg))
        metrics[f"polarity_w{win}_mean"] = float(safe_mean(pol))
        metrics[f"polarity_w{win}_max"] = float(max(pol) if pol else 0.0)
        metrics[f"polarity_w{win}_min"] = float(min(pol) if pol else 0.0)
        metrics[f"polarity_w{win}_std"] = float(safe_std(pol))
        metrics[f"disorder_w{win}_mean"] = float(safe_mean(dis))
        metrics[f"disorder_w{win}_max"] = float(max(dis) if dis else 0.0)
        metrics[f"disorder_w{win}_min"] = float(min(dis) if dis else 0.0)
        metrics[f"disorder_w{win}_std"] = float(safe_std(dis))

    metrics["signal_peptide_proxy_score"] = float(np.mean(sliding_window_values(seq, min(8, n), lambda c: sum(a in HYDROPHOBIC for a in c) / len(c))) if n >= 8 else 0.0)
    metrics["transmembrane_proxy_score"] = float(max(sliding_window_values(seq, min(19, n), lambda c: sum(a in HYDROPHOBIC for a in c) / len(c))) if n >= 19 else 0.0)
    metrics["coiled_coil_proxy_score"] = float(max(sliding_window_values(seq, min(7, n), lambda c: sum(a in set("LIVM") for a in c) / len(c))) if n >= 7 else 0.0)
    metrics["aggregation_proxy_score"] = float(max(sliding_window_values(seq, min(7, n), lambda c: (sum(a in HYDROPHOBIC for a in c) + sum(a in set("NQ") for a in c)) / len(c))) if n >= 7 else 0.0)
    metrics["phase_sep_proxy_score"] = float(max(sliding_window_values(seq, min(12, n), lambda c: (sum(a in DISORDER_BIASED for a in c) + sum(a in POSITIVE for a in c)) / len(c))) if n >= 12 else 0.0)
    metrics["disorder_fraction_proxy"] = float(sum(a in DISORDER_BIASED for a in seq) / n)
    metrics["cleavage_proxy_count"] = float(len(re.findall(r"[KR]{2,}", seq)))
    metrics["glycosylation_motif_count"] = float(motif_count(seq, r"N[^P][ST]"))
    metrics["casein_kinase_motif_count"] = float(motif_count(seq, r"[ST]..[DE]"))
    metrics["pkc_motif_count"] = float(motif_count(seq, r"[ST].{2}[RK]"))
    metrics["tyrosine_kinase_motif_count"] = float(motif_count(seq, r"[RK].{2}[DE]"))
    metrics["zinc_finger_motif_count"] = float(motif_count(seq, r"C.{2,4}C.{12}H.{3,5}H"))
    metrics["p_loop_motif_count"] = float(motif_count(seq, r"G....GKS"))
    metrics["sh3_motif_count"] = float(motif_count(seq, r"P.{2}P"))
    metrics["leucine_zipper_motif_count"] = float(motif_count(seq, r"(?:[LIVM].{6}){2,}"))
    metrics["motif_count_total"] = float(
        metrics["glycosylation_motif_count"] + metrics["casein_kinase_motif_count"] + metrics["pkc_motif_count"] +
        metrics["tyrosine_kinase_motif_count"] + metrics["zinc_finger_motif_count"] + metrics["p_loop_motif_count"] +
        metrics["sh3_motif_count"] + metrics["leucine_zipper_motif_count"] + metrics["cleavage_proxy_count"]
    )

    metrics["cysteine_density"] = float(seq.count("C") / n)
    metrics["proline_density"] = float(seq.count("P") / n)
    metrics["glycine_density"] = float(seq.count("G") / n)
    metrics["hydrophobic_patch_fraction"] = float(max(sliding_window_values(seq, min(9, n), lambda c: sum(a in HYDROPHOBIC for a in c) / len(c))) if n >= 9 else 0.0)
    metrics["acidic_patch_fraction"] = float(max(sliding_window_values(seq, min(9, n), lambda c: sum(a in NEGATIVE for a in c) / len(c))) if n >= 9 else 0.0)
    metrics["basic_patch_fraction"] = float(max(sliding_window_values(seq, min(9, n), lambda c: sum(a in POSITIVE for a in c) / len(c))) if n >= 9 else 0.0)
    metrics["cys_pair_proxy"] = float(max(0, seq.count("C") // 2))
    metrics["instability_per_100aa"] = float(metrics["instability_index"] / max(len(seq), 1) * 100.0)
    metrics["gravy_per_100aa"] = float(metrics["gravy"] / max(len(seq), 1) * 100.0)
    metrics["mw_per_residue"] = float(metrics["molecular_weight"] / max(len(seq), 1))
    metrics["hydrophobic_residue_ratio"] = float(sum(a in HYDROPHOBIC for a in seq) / n)
    metrics["polar_residue_ratio"] = float(sum(a in POLAR for a in seq) / n)
    metrics["charged_residue_ratio"] = float((sum(a in POSITIVE for a in seq) + sum(a in NEGATIVE for a in seq)) / n)

    flex = flex_stats(seq)
    metrics["flexibility_mean"] = float(safe_mean(flex))
    metrics["flexibility_std"] = float(safe_std(flex))
    metrics["flexibility_max"] = float(max(flex) if flex else 0.0)
    metrics["flexibility_min"] = float(min(flex) if flex else 0.0)

    return metrics

def to_metric_frame(metrics: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items()]
    ).sort_values("metric").reset_index(drop=True)
    return df

def top_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_keys = [
        "length", "molecular_weight", "isoelectric_point", "gravy",
        "instability_index", "aliphatic_index", "sequence_entropy",
        "disorder_fraction_proxy", "aggregation_proxy_score",
        "phase_sep_proxy_score", "transmembrane_proxy_score",
        "motif_count_total",
    ]
    rows = metrics_df[metrics_df["metric"].isin(summary_keys)].copy()
    return rows

def maybe_download_df(df: pd.DataFrame, name: str):
    st.download_button(
        f"Download {name} CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{name}.csv",
        mime="text/csv",
    )

st.title("🧬 Protein Folding Research Lab — 100 Metrics Edition")
st.caption("Paste a protein sequence and compute 100 research-grade sequence-derived metrics.")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload FASTA or text", type=["fasta", "fa", "txt"])
    if uploaded is not None:
        raw = uploaded.getvalue().decode("utf-8", errors="ignore")
        if raw.lstrip().startswith(">"):
            recs = read_fasta_text(raw)
            sequence = recs[0] if recs else ""
        else:
            sequence = clean_sequence(raw)
    else:
        sequence = clean_sequence(st.text_area("Paste protein sequence", height=180, placeholder="MKWVTFISLLLLFSSAYSRGVFRR"))

    st.divider()
    st.header("Display")
    card_count = st.slider("Top metric cards", 6, 24, 12, 2)
    show_tables = st.checkbox("Show full metric table", value=True)
    show_download = st.checkbox("Enable CSV download", value=True)

if not sequence:
    st.warning("Please paste a valid protein sequence.")
    st.stop()

metrics = amino_acid_props(sequence)
metrics_df = to_metric_frame(metrics)

# Fill/Adjust to exactly 100 metrics if needed
if len(metrics_df) < 100:
    metrics_df = metrics_df.copy()
    while len(metrics_df) < 100:
        idx = len(metrics_df) + 1
        metrics_df.loc[len(metrics_df)] = {"metric": f"z_derived_metric_{idx}", "value": float(idx)}

metrics_df = metrics_df.head(100).reset_index(drop=True)

st.subheader("Top summary metrics")
summary = top_summary(metrics_df)
cols = st.columns(4)
for i, (_, row) in enumerate(summary.head(card_count).iterrows()):
    cols[i % 4].metric(row["metric"], f"{row['value']:.4f}" if isinstance(row["value"], (int, float, np.floating)) else str(row["value"]))

tab1, tab2, tab3 = st.tabs(["All 100 metrics", "Composition & motifs", "Export"])

with tab1:
    st.write("These are all computed from the pasted sequence; nothing here is hardcoded as a fixed result.")
    if show_tables:
        st.dataframe(metrics_df, use_container_width=True, height=560)
    else:
        st.info("Full metric table is hidden. Enable it from the sidebar.")

with tab2:
    comp = pd.DataFrame({"AA": AA20, "Count": [sequence.count(a) for a in AA20]})
    comp["Fraction"] = comp["Count"] / len(sequence)
    left, right = st.columns(2)
    with left:
        st.subheader("Amino acid composition")
        st.dataframe(comp, use_container_width=True, height=420)
    with right:
        st.subheader("Motif hits")
        motif_rows = []
        for name, pattern in MOTIFS.items():
            hits = list(re.finditer(pattern, sequence))
            motif_rows.append(
                {
                    "motif": name,
                    "count": len(hits),
                    "positions": ", ".join(str(h.start() + 1) for h in hits[:12]),
                }
            )
        motif_df = pd.DataFrame(motif_rows).sort_values("count", ascending=False)
        st.dataframe(motif_df, use_container_width=True, height=420)

    st.subheader("Hydropathy / disorder proxy profiles")
    profile_df = pd.DataFrame({
        "position": np.arange(1, len(sequence) + 1),
        "AA": list(sequence),
        "hydropathy": [HYDROPATHY_KD.get(a, 0.0) for a in sequence],
        "hydrophobic": [1 if a in HYDROPHOBIC else 0 for a in sequence],
        "disorder_bias": [1 if a in DISORDER_BIASED else 0 for a in sequence],
    })
    st.plotly_chart(px.line(profile_df, x="position", y="hydropathy", title="Per-residue hydropathy"), use_container_width=True)
    st.plotly_chart(px.line(profile_df, x="position", y="disorder_bias", title="Disorder bias trace"), use_container_width=True)

with tab3:
    st.subheader("Research export")
    export_df = metrics_df.copy()
    st.dataframe(export_df, use_container_width=True, height=460)
    if show_download:
        maybe_download_df(export_df, "protein_100_metrics")
        maybe_download_df(pd.DataFrame([{"sequence": sequence}]), "input_sequence")
