# Speaker-Aware Meeting Transcription Pipeline

A modular Jupyter notebook pipeline that produces speaker-labeled meeting transcripts by combining voice activity detection (VAD), speaker diarization, and automatic speech recognition (ASR). Evaluated on the AMI Meeting Corpus.

```
Audio → VAD → Speaker Embeddings → Clustering → ASR → Transcript → Evaluation
```

## Dataset

The pipeline uses a 20-meeting subset of the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), a public dataset of recorded meetings with speaker annotations. EN2001a is held out for evaluation; the remaining meetings are used for development and fine-tuning.

Run `download-dataset.ipynb` to fetch the data from HuggingFace into the `data/` directory (requires a HuggingFace token in `.env`).


## Notebooks [still in progress]

- `01_vad_segmentation.ipynb`: Runs voice activity detection on the meeting audio to identify when someone is speaking and splits the recording into timestamped speech segments.
- `02_embeddings.ipynb`: Converts each speech segment into a fixed-size vector that captures speaker identity, using two pretrained models (ECAPA-TDNN and x-vector). Compares how well each model separates speakers.
- `03_clustering.ipynb`: Groups the embedding vectors by speaker identity. Tries multiple clustering algorithms and picks the one with the best separation score. Outputs a speaker label for each segment.
- `04_asr.ipynb`: Transcribes each speech segment using Whisper, producing time-aligned text for every segment.
- `05_integration.ipynb`: Combines the speaker labels from notebook 03 and the transcripts from notebook 04 into a single readable, timestamped dialogue.
- `06_evaluation.ipynb`: Measures pipeline quality against AMI ground truth. Reports DER (how often the wrong speaker is assigned) and WER (how accurate the transcription is).
- `07_embedding_finetune.ipynb`: Adapts the speaker embeddings to the AMI domain by training a small layer on top of the pretrained model. Compares embedding quality before and after.
- `08_clustering-on-finetunned_embeding..ipynb`: Repeats the clustering comparison from notebook 03 using the fine-tuned embeddings to confirm whether the same algorithm still wins.



## Results

Evaluated on EN2001a using Spectral (cosine) clustering and Whisper base.

| Metric | Value | Note |
|---|---|---|
| DER | 36.6% | Mostly false alarms (13.5%) and confusion (12.9%), not missed speech |
| WER | 29.8% | Expected for Whisper base on multi-speaker meeting audio |
| Silhouette (pretrained) | 0.39 | k mis-estimated as 4 |
| Silhouette (fine-tuned) | 0.92 | k correctly estimated as 5; same algorithm wins |

## How to run
`uv sync` to sync the repo to your local machine. Then, run the notebooks in order. Each notebook saves its outputs to JSON files for use in subsequent notebooks.
> Also make sure to set your HuggingFace token in the `.env` file (see `.env.example` for reference).