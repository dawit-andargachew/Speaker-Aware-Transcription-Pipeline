## Meeting Diarization and Transcription Pipeline

# Speaker-Aware Meeting Transcription Pipeline

This repository implements a **speaker-aware meeting transcription pipeline** that transforms meeting recordings into transcripts with speaker labels. The system identifies **who spoke and what was said** by combining voice activity detection (VAD), speaker diarization, and automatic speech recognition (ASR).

The pipeline is evaluated on the AMI Meeting Corpus and is structured as a set of modular Jupyter notebooks. Each notebook corresponds to a specific stage of the workflow, including voice activity detection (VAD), speaker embedding extraction, clustering for diarization, automatic speech recognition (ASR), and evaluation against ground truth annotations.

This modular design enables easy experimentation with different models and algorithms at each stage of the pipeline, supporting research and iterative improvement of speaker-aware transcription systems.

## Structure [in progress]

- `01_vad_segmentation.ipynb` — VAD (Silero is the right pick per your professor), outputs speech segments with timestamps
    - Load a raw AMI audio file (headset-mix, mono, 16kHz)
    - Run Silero VAD to detect speech vs. silence regions
    - Merge short gaps between speech segments (tunable threshold)
    - Visualize the waveform with speech/silence regions overlaid
    - Save output: list of `(start_sec, end_sec)` segments to disk

- `02_embeddings.ipynb` — For each speech segment detected in notebook 01, extract a fixed-size vector that encodes speaker identity. Do this with two different model architectures and compare the quality of the resulting embedding spaces.

    - Load inputs — read segments.json from notebook 01 (segment IDs, timestamps, source file) and load the corresponding audio files so you can slice out each segment by timestamp
    - Extract embeddings with two pretrained models — for each segment, pass the audio through each model to get a fixed-size speaker embedding vector. The two models are:

        - ECAPA-TDNN — loaded from the speechbrain/spkrec-ecapa-voxceleb checkpoint via the SpeechBrain library. Produces 192-dimensional embeddings. More recent architecture, designed to handle short segments and overlapping speech well.
        - x-vector — loaded from the speechbrain/spkrec-xvect-voxceleb checkpoint, also via SpeechBrain. Produces 512-dimensional embeddings. The established baseline architecture that most prior diarization work is compared against.
        - Both checkpoints were pretrained on VoxCeleb, so pretraining data is not a variable in your comparison — only the architecture differs.


    - Visualize the embedding spaces — project the high-dimensional embeddings down to 2D using UMAP (preferred) or t-SNE, and plot them colored by ground truth speaker label from the AMI reference annotations. This is a sanity check: if the model is working correctly, same-speaker segments should cluster together visually. Do this separately for ECAPA and x-vector so you can qualitatively compare how well each model separates speakers.
    - Save outputs — write two files to disk, one per model: embeddings_ecapa.json and embeddings_xvector.json. Each entry contains {seg_id, file, start_sec, end_sec, embedding} where embedding is the vector serialized as a list of floats. Notebook 03 loads these directly.



- **`03_clustering.ipynb`** — Two-phase comparison: first identify the best embedding by holding clustering fixed, then identify the best clustering algorithm by holding the embedding fixed. Output is speaker-labeled segments from the winning combination.
    - **Load embeddings** — read both `embeddings_ecapa.json` and `embeddings_xvect.json` from notebook 02. Structure the code so swapping embedding source is a single variable change.

    - **Phase 1: find the best embedding** — run agglomerative clustering (ward linkage) on both embedding sets. Keep everything else identical — same linkage, same number of clusters (4, the known AMI value). Compute silhouette score for each. The winner is the embedding that produces better-separated clusters. Expected outcome: ECAPA wins. Drop the loser after this phase.

    - **Phase 2: find the best clustering algorithm** — using only the winning embedding (ECAPA), compare three configurations:
        - Agglomerative + ward linkage
        - Agglomerative + average linkage
        - Spectral clustering
        - Compute silhouette score for each. Pick the winner.

    - **Experiment with number of clusters** — for the winning combination, test known (n=4) vs dendrogram-estimated number of speakers. Visualize the dendrogram cutoff to justify the estimate.

    - **Visualize** — UMAP plots of cluster assignments vs ground truth speaker labels for each phase. Dendrograms for agglomerative runs.

    - **Select winner** — document the chosen embedding + clustering combination with justification. Everything from notebook 04 onward uses only this.

    - **Save output** — write `segments_labeled.json`: `{seg_id, file, start_sec, end_sec, speaker_label}`. This is your RTTM and feeds into notebook 05.
    

- `04_asr.ipynb` — Whisper on each segment with timestamp-aligned input, output raw transcripts
    - Load audio + segments (with timestamps) from notebook 01
    - Run Whisper on each segment individually, passing the timestamp offset so output is time-aligned
    - Save output: `(segment_id, start, end, transcript_text)`

- `05_integration.ipynb` — Join speaker labels and transcripts into a coherent speaker-labeled dialogue, handle edge cases, and sanity check the result before evaluation.

    - Load inputs — read `segments_labeled.json` from notebook 03 and `transcripts.json` from notebook 04, join on `seg_id`
    - Handle edge cases — flag and document segments with missing transcript or missing speaker label, decide and justify how to handle them (drop vs placeholder)
    - Sanity check — verify no overlapping segments, print per-speaker turn count and average duration
    - Format output — produce a readable timestamped dialogue transcript
    - Eyeball check — display a 2-3 minute window from the middle of the meeting for quick quality assessment
    - Save output — write `transcript_labeled.json`: `{seg_id, file, start_sec, end_sec, speaker_label, transcript_text}`, feeds directly into notebook 06.
    

- `06_evaluation.ipynb` — DER, WER, analysis, comparison tables
    - Load your RTTM output (notebook 03) + AMI reference RTTM
    - Compute **DER** using pyannote.metrics or dscore
    - Load your transcripts (notebook 04) + AMI reference transcripts
    - Compute **WER** using jiwer
    - Build a comparison table across your embedding and clustering variants
    - Plot **DER/WER** tradeoffs, error breakdown (missed speech, false alarm, confusion)



- `06_evaluation.ipynb` — Evaluate the full pipeline against AMI ground truth using DER for speaker attribution and WER for transcription quality.
    - Convert to RTTM — convert segments_labeled.json from notebook 03 to proper RTTM format for DER computation. AMI reference RTTM is already in the right format.
    - Compute DER — using pyannote.metrics, compute DER against AMI reference RTTM. Break down into:
        - Missed speech — VAD failed to detect speech (notebook 01 error)
        - False alarm — silence labeled as speech (notebook 01 error)
        - Confusion — right speech, wrong speaker (notebook 03 error)
        - This tells you which stage is responsible for errors, not just how bad the total is.
    - Oracle vs estimated k — compute DER twice:
        - Estimated k from notebook 03 (silhouette sweep)
        - Oracle k from AMI ground truth (the known speaker count)
        - The gap between the two quantifies how much speaker count estimation is hurting you vs the clustering itself.
    - Compute WER — load transcript_labeled.json from notebook 05, load AMI reference transcripts, compute WER using jiwer.
    - Summary table:
            Metric                        Value
            DER (estimated k)             X%
            DER (oracle k)                Y%
            Missed speech               X%
            False alarm                 X%
            Confusion                   X%
            WER                           X%
    - Error analysis — answer these two questions explicitly:
        - Is DER dominated by confusion or missed speech? Confusion = clustering problem. Missed speech = VAD problem.
        - Is the gap between oracle k and estimated k large? If yes, speaker count estimation is a significant weakness worth discussing.

    - Save output — write evaluation_results.json with all metrics for the report.

