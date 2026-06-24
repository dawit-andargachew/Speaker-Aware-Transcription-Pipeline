# Speaker-Aware Meeting Transcription Pipeline

A modular Jupyter notebook pipeline that produces speaker-labeled meeting transcripts by combining voice activity detection (VAD), speaker diarization, and automatic speech recognition (ASR). Evaluated on the AMI Meeting Corpus.

## Notebooks

- `01_vad_segmentation.ipynb`: Run Silero VAD on a raw AMI audio file (16kHz mono), merge short silence gaps, and save speech segments as `(start_sec, end_sec)` pairs.

- `02_embeddings.ipynb`: Extract 192-dim ECAPA-TDNN and 512-dim x-vector speaker embeddings (both pretrained on VoxCeleb via SpeechBrain) for each VAD segment. Visualize embedding spaces with UMAP and save results to `embeddings_ecapa.json` and `embeddings_xvector.json`.

- `03_clustering.ipynb`: Two-phase comparison to select the best embedding and clustering algorithm.
    - Phase 1: Compare ECAPA vs. x-vector using ward clustering at fixed k. ECAPA wins.
    - Phase 2: Compare ward, average-cosine, and spectral clustering on ECAPA embeddings. Pick by silhouette score.
    - Estimate k via silhouette sweep and dendrogram; compare against oracle k.
    - Save speaker-labeled segments to `segments_labeled.json`.

- `04_asr.ipynb`: Run Whisper on each VAD segment with timestamp offsets and save transcripts as `(seg_id, start, end, text)`.

- `05_integration.ipynb`: Join `segments_labeled.json` and `transcripts.json` on `seg_id`, handle missing labels or transcripts, and save the merged result to `transcript_labeled.json`.

- `06_evaluation.ipynb`: Evaluate the full pipeline against AMI ground truth.
    - DER: broken down into missed speech, false alarm, and confusion using a frame-based approach.
    - WER: computed with jiwer against AMI reference transcripts.
    - Compares oracle k vs. estimated k to isolate how much speaker count estimation affects DER.
    - Saves all metrics to `evaluation_results.json`.

- `07_embedding_finetune.ipynb`: Domain-adapt ECAPA-TDNN to the AMI corpus using a lightweight adaptation layer (frozen ECAPA + trainable MLP, ~130k params). Trained on 4 meetings, evaluated on EN2001a. Reports before/after embedding quality metrics.

- `08_clustering-on-finetunned_embeding..ipynb`: Repeat the notebook 03 clustering comparison using the fine-tuned embeddings to verify that the selected algorithm holds after adaptation.