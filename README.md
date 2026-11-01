# When Does Conversational Context Help Emotion Recognition?

**A hardness-stratified study of adaptive context in multimodal ERC with a pre-specified decision rule.**
Sudipta Bhattacharya — Institute of Engineering and Management, Kolkata. Submitted to *IEEE Transactions on Affective Computing* (Nov 2026).

This repository holds everything needed to reproduce every number in the paper: the Colab notebook, the per-seed result files it wrote, the summary tables, the superseded runs that the paper's decision rule refers to, and the manuscript source.

## What the paper finds

* A label-free **instance-hardness score** `h(t)` (normalised label entropy of the K nearest training-fold neighbours in frozen WavLM-Large + RoBERTa-base embedding space; own utterance, dialogue and speaker excluded) stratifies ERC evaluation without reading any test label.
* **Context is worth +8.4 W-F1 on IEMOCAP and nothing on MELD**, and on IEMOCAP the gain rises with hardness: +4.8 in the easiest quartile, +11.2 in the hardest.
* A learned **per-utterance context-length policy does not beat a fixed 8-turn window** (65.5 vs 67.2) and is **indistinguishable from random assignment at the same mean context** (65.5 ± 1.7 vs 64.8 ± 2.1). An oracle of sufficient context explains why: ~40 % of utterances need no context, ~40 % are not resolved by any.
* `h` does **not** track inter-annotator disagreement (Spearman ρ = 0.04 on IEMOCAP Session 5).
* Both pre-specified success criteria for a method claim failed; the paper reports the study as an analysis with controlled negative results.

## Layout

```
notebook/hardness_aware_ERC_final_colab.ipynb   # the complete pipeline (rev. 4c); Run all in Colab
scripts/build_notebook.py                        # generates the notebook (single source of truth for the code)
scripts/posthoc_analysis.py                      # per-class F1, confusions, entropy-k k-bar (also Section 11 of the notebook)
results/iemocap/        # per-seed CSVs, IEMOCAP, final protocol (Ses01-04 / Ses05; LOSO)
results/emorynlp/       # per-seed CSVs, EmoryNLP (text only)
results/summary/        # summary_tables_II_IV, paired_bootstrap_CIs, table_III_*, table_VI, kstar_by_quartile_*
results/old_policy_28oct/  # IEMOCAP policy runs with the train/test window mismatch described in Sec. III-F (superseded)
paper/                  # main.tex, references.bib, figures/, make_figures.py, cover_letter.tex, main.pdf
```

**To be added from Google Drive before release** (they exist, see the notebook's `results/` folder): the MELD per-seed CSVs (`*_meld.csv`, `*_meld_pooled.npz`), `table_perclass_iemocap.csv`, `confusion_k0/k8_iemocap.csv`, the `*_pooled.npz` prediction files, `results/text_only_pilot/` (8 Oct text-only run), and the cached embeddings `cache/*.npy` (deposit on Zenodo; ~2 GB).

## Reproducing

1. Obtain IEMOCAP (USC licence) and MELD (official release). Place `IEMOCAP_full_release` as a zip at `MyDrive/IEMOCAP/archive.zip` and `MELD.Raw.tar.gz` at `MyDrive/MELD_final/`. EmoryNLP label CSVs are fetched from GitHub automatically.
2. Open `notebook/hardness_aware_ERC_final_colab.ipynb` in Google Colab with a GPU runtime and **Run all**. Embeddings (~1.5 GPU-h) and every experiment (~30 GPU-h on an L4) are cached to `MyDrive/hardness_erc/`; a disconnected session resumes from checkpoints on the next Run all.
3. Section 10 prints the paper's tables in Markdown; Section 11 prints the per-class analysis.
4. `python paper/make_figures.py` regenerates the three figures; `latexmk -pdf paper/main.tex` builds the manuscript.

Seeds {11, 22, 33, 44, 55}; all hyper-parameters are in the `CFG` cell. The decision rule (paper Sec. IV-D) was written into the manuscript on 23 Oct 2026, before any multimodal run; the dated draft is in `paper/history/` (to be exported from the working document).

## Citation

```bibtex
@article{bhattacharya2026context,
  title   = {When Does Conversational Context Help Emotion Recognition? A Hardness-Stratified Study of Adaptive Context in Multimodal {ERC} with a Pre-Specified Decision Rule},
  author  = {Bhattacharya, Sudipta},
  journal = {submitted to IEEE Transactions on Affective Computing},
  year    = {2026}
}
```

## Licence

Code: MIT. Result files, tables and figures: CC BY 4.0. IEMOCAP and MELD are **not** redistributed; obtain them from their owners.
