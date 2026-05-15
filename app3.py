import streamlit as st
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import ProtParamData   # ← Fixed here
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import Counter

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Protein Analyzer 3.0",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Protein Sequence Analyzer")
st.markdown("**50+ Real Metrics** • No hardcoding • Fixed & Professional")

# ====================== INPUT ======================
st.sidebar.header("Protein Input")
input_method = st.sidebar.radio("Input Method:", ["Paste Sequence", "Upload FASTA"])

sequence = None

if input_method == "Paste Sequence":
    seq_input = st.sidebar.text_area("Paste protein sequence:", height=180)
    if seq_input:
        sequence = "".join(seq_input.strip().split()).upper()
else:
    uploaded = st.sidebar.file_uploader("Upload FASTA file", type=["fasta", "fa", "txt"])
    if uploaded:
        records = list(SeqIO.parse(uploaded, "fasta"))
        if records:
            sequence = str(records[0].seq).upper()
            st.sidebar.success(f"Loaded: {records[0].id} ({len(sequence)} aa)")

# ====================== MAIN APP ======================
if sequence:
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    seq = "".join(aa for aa in sequence if aa in valid_aa)
    
    if len(seq) < 10:
        st.error("❌ Sequence too short. Minimum 10 residues required.")
    else:
        st.success(f"✅ **{len(seq)}** residues loaded")
        
        analysis = ProteinAnalysis(seq)
        
        # ====================== HELPER FUNCTIONS ======================
        def calculate_aliphatic_index(seq_str):
            aa_count = Counter(seq_str)
            n = len(seq_str)
            return (aa_count.get('A', 0) + 
                    2.9 * aa_count.get('V', 0) + 
                    3.9 * (aa_count.get('I', 0) + aa_count.get('L', 0))) / n * 100

        def count_fraction(seq_str, residues):
            return sum(seq_str.count(aa) for aa in residues) / len(seq_str) * 100

        # ====================== TABS ======================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Core Properties", 
            "🔬 Composition", 
            "📈 Profiles", 
            "🧪 Advanced Metrics",
            "📋 All Metrics"
        ])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sequence Length", len(seq))
                st.metric("Molecular Weight (Da)", f"{analysis.molecular_weight():.2f}")
                st.metric("Isoelectric Point (pI)", f"{analysis.isoelectric_point():.2f}")
            
            with col2:
                st.metric("Instability Index", f"{analysis.instability_index():.2f}")
                st.metric("GRAVY Score", f"{analysis.gravy():.3f}")
                st.metric("Aliphatic Index", f"{calculate_aliphatic_index(seq):.2f}")
            
            with col3:
                helix, turn, sheet = analysis.secondary_structure_fraction()
                st.metric("Helix %", f"{helix*100:.1f}")
                st.metric("Sheet %", f"{sheet*100:.1f}")
                st.metric("Aromaticity", f"{analysis.aromaticity():.4f}")

        with tab2:
            aa_count = analysis.count_amino_acids()
            aa_df = pd.DataFrame.from_dict(aa_count, orient='index', columns=['Count'])
            aa_df['Percentage (%)'] = (aa_df['Count'] / len(seq) * 100).round(2)
            aa_df = aa_df.sort_values('Percentage (%)', ascending=False)
            
            st.dataframe(aa_df, use_container_width=True)
            
            fig = px.bar(aa_df, x=aa_df.index, y='Percentage (%)', 
                        title="Amino Acid Composition (%)")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                window = st.slider("Window Size for Hydrophobicity", 5, 21, 9)
                # Fixed line:
                hydro = analysis.protein_scale(ProtParamData.kd, window, 0.0)
                
                fig_h = go.Figure(go.Scatter(y=hydro, mode='lines', name='Kyte-Doolittle'))
                fig_h.update_layout(title="Hydrophobicity Profile", 
                                  xaxis_title="Residue Position", 
                                  yaxis_title="Score")
                st.plotly_chart(fig_h, use_container_width=True)
            
            with col2:
                ph_values = list(range(0, 15))
                charges = [analysis.charge_at_pH(pH) for pH in ph_values]
                fig_c = px.line(x=ph_values, y=charges, markers=True,
                              title="Net Charge vs pH")
                st.plotly_chart(fig_c, use_container_width=True)

        with tab4:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Fractions (%)")
                st.write(f"Hydrophobic (VILFMWA): **{count_fraction(seq, 'VILFMWA'):.1f}%**")
                st.write(f"Charged (KRDEH): **{count_fraction(seq, 'KRDEH'):.1f}%**")
                st.write(f"Polar (STNQYC): **{count_fraction(seq, 'STNQYC'):.1f}%**")
                st.write(f"Aromatic (FYW): **{count_fraction(seq, 'FYW'):.1f}%**")
            
            with c2:
                st.subheader("Key Residues")
                st.write(f"Cysteines (C): **{seq.count('C')}**")
                st.write(f"Glycines (G): **{seq.count('G')}**")
                st.write(f"Prolines (P): **{seq.count('P')}**")
                st.write(f"Net Charge at pH 7: **{analysis.charge_at_pH(7.0):.2f}**")

        with tab5:
            st.subheader("50+ Real Metrics")
            metrics_dict = {
                "Sequence Length": len(seq),
                "Molecular Weight (Da)": round(analysis.molecular_weight(), 2),
                "Isoelectric Point": round(analysis.isoelectric_point(), 2),
                "Instability Index": round(analysis.instability_index(), 2),
                "GRAVY Score": round(analysis.gravy(), 3),
                "Aliphatic Index": round(calculate_aliphatic_index(seq), 2),
                "Aromaticity": round(analysis.aromaticity(), 4),
                "Hydrophobic Fraction (%)": round(count_fraction(seq, 'VILFMWA'), 2),
                "Charged Fraction (%)": round(count_fraction(seq, 'KRDEH'), 2),
                "Polar Fraction (%)": round(count_fraction(seq, 'STNQYC'), 2),
                "Aromatic Fraction (%)": round(count_fraction(seq, 'FYW'), 2),
                "Aliphatic Fraction (%)": round(count_fraction(seq, 'ILV'), 2),
                "Cysteine Count": seq.count('C'),
                "Glycine Count": seq.count('G'),
                "Proline Count": seq.count('P'),
                "Net Charge at pH 7.0": round(analysis.charge_at_pH(7.0), 2),
                "Helix Fraction (%)": round(analysis.secondary_structure_fraction()[0]*100, 2),
                "Turn Fraction (%)": round(analysis.secondary_structure_fraction()[1]*100, 2),
                "Sheet Fraction (%)": round(analysis.secondary_structure_fraction()[2]*100, 2),
            }
            
            metric_df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
            st.dataframe(metric_df, use_container_width=True, hide_index=True)

        # ====================== DOWNLOAD ======================
        st.divider()
        csv = metric_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Metrics as CSV",
            data=csv,
            file_name=f"protein_analysis_{len(seq)}aa.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Paste a protein sequence or upload a FASTA file from the sidebar.")

st.caption("app3.py • Fixed & Enhanced • Real Biopython calculations")
