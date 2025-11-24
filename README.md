# OMol25Metals: Metal-Complex Subset of OMol25 for TopoBench

This repository provides a preprocessed subset of the **Open Molecules 2025 (OMol25)** dataset
containing **metal complexes**, prepared as a **hypergraph dataset** for the
[TAG-DS Topological Deep Learning Challenge 2025](https://geometric-intelligence.github.io/topobench/tdl-challenge/).

The subset is designed to be used with **TopoBench** as a natively higher-order dataset:
atoms are nodes, bonds are edges, and rings (and other domain-defined groups) are represented
as hyperedges.

Team name for the challenge: **NREL-Insightcenter**  
Dataset name inside TopoBench: **`omol25_metals`**  
Data domain: **`hypergraph`**

---

## 1. Data origin and license

This dataset is derived from the **OMol25** collection released by Meta FAIR:

> D. S. Levine et al., “The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models,” 2025.
> arXiv:2505.08762.

The original dataset and documentation are hosted on HuggingFace:
<https://huggingface.co/facebook/OMol25>

According to the HuggingFace page, the **OMol25 dataset is provided under a CC-BY-4.0 license**.
Please make sure to read and accept the license terms on HuggingFace before downloading
the full data.

This repository only contains:
1. A **metal-complex subset** extracted from OMol25.
2. **Preprocessed PyTorch Geometric (PyG) files** ready to be used with TopoBench.
3. Scripts that explain how to regenerate the subset from the original OMol25 release.

When using this dataset, please **cite OMol25** as well as any work that uses this subset.

---

## 2. What is in this subset?

We select **metal complexes** from OMol25 as described in the OMol25 documentation
(see `DATASET.md` on the HuggingFace page) and then convert them into PyG-compatible
hypergraph objects.

High-level structure (you can adapt to your local setup):

```text
data/
  omol25_metals/
    subset/
      processed/
        data.pt
```
---

## 3. How to regenerate the subset

To fully reproduce the subset, you need to:
1. Log into HuggingFace and accept the OMol25 dataset license: <https://huggingface.co/facebook/OMol25>
2. Download the part of OMol25 that contains metal complexes (see OMol25 `DATASET.md` for the revelant split and file groups)
3. Run the conversion script in this repo to generate the PyG data:
```# From the root of this repo
python script/ase2pyg.py \
    --omol25-root /path/to/OMol25_raw \
    --output-root ./data/omol25_metals \
    --split metal_complexes
```
