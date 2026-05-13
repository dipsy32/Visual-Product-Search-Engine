# Visual Product Search Engine

This repository contains the final project for our Visual Recognition course. We built a **Query-by-Image** product search system that allows users to upload an image of a person, crop a specific clothing item (Upper, Lower, or Full Body), and instantly retrieve visually and semantically similar products from a catalog.

##  Team Members
* **Shrujan Teja** (IMT2023599)
* **Sathvik Jakkampudi** (IMT2023124)
* **Rajdeep Saha** (IMT2023600)
* **Nachiappan N** (IMT2023605)

---

##  System Architecture
* **Localization (YOLOv8):** Detects and crops the primary clothing item based on user selection.
* **Semantic Captioning (BLIP-2):** Generates a rich, consistent text description of the cropped item.
* **Multimodal Embedding (CLIP):** Fuses the visual features of the crop and the semantic features of the caption into a single vector.
* **Vector Search (FAISS):** Rapidly retrieves the top-K similar items from the indexed catalog.

### Ablation Studies & Configurations
We tested three configurations to optimize retrieval metrics (Recall@K, NDCG@K, mAP@K):
* `Config-A`: Vision-only baseline (CLIP image embeddings).
* `Config-B`: Frozen CLIP + Frozen BLIP-2 (Late fusion of text and image modalities). **(Best Performing Model)**
* `Config-C`: Fine-tuned CLIP + Frozen BLIP-2.

---

##  Repository Structure

```text
├── Ablation_Studies/
│   ├── Config-A.ipynb  # Vision-Only Baseline
│   ├── Config-B.ipynb  # Frozen Multimodal Fusion (Best Model)
│   └── Config-C.ipynb  # Fine-Tuned Multimodal
├── app/
│   ├── app.py                 # Streamlit UI
│   ├── requirements.txt       # Dependencies
│   ├── best_yolo.pt           # Trained YOLO weights
│   ├── product_index.index    # FAISS vector index
│   └── product_metadata.csv   # Catalog metadata mapping
└── README.md
