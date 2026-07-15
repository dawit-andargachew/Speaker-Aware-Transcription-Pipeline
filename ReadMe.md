# Speaker-Aware Meeting Transcription Pipeline


An end-to-end speaker diarization and automatic speech recognition (ASR) pipeline specifically evaluated on the **AMI Meeting Corpus**. The primary goal is to accurately identify "who spoke when" (Diarization Error Rate - DER) and transcribe the dialogue (Word Error Rate - WER) in challenging, real-world acoustic environments.

```text
Audio → VAD → Speaker Embeddings (ECAPA-TDNN) → Spectral Clustering → ASR (Whisper) → Transcript
```

## Dataset & Architecture Scope

The experiments were conducted using a representative subset of the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), a public dataset of recorded meetings featuring synchronized headset and distant microphones.

### Dataset Acquisition via Google Drive
Due to the substantial size of the AMI corpus, pulling the data directly from Hugging Face during every session proved highly inefficient, frequently resulting in network timeouts and DNS blocks. To ensure a stable and rapid initialization process, the dataset was pre-processed locally, extracted into raw `.wav` files, and hosted on Google Drive. 

### Scope and Resource Constraints
This project uses a targeted subset of 15 meeting series rather than the entire corpus. The full corpus is large (100+ hours), and processing full sessions repeatedly caused Google Colab to crash from memory exhaustion. Thus, 12 series were dedicated to training/validation (using a dynamic 4-Fold Cross Validation loop) and 3 series were strictly held out for testing.

## Notebook Structure

The pipeline has been consolidated into a single, comprehensive Colab environment for streamlined execution:

- `meeting-transcription-colab_version.ipynb`: A multi-stage pipeline containing:
  - **Stage 1 (VAD):** Silero VAD segmentation and silence bridging.
  - **Stage 2 (Embeddings):** Feature extraction via pretrained ECAPA-TDNN.
  - **Stage 3 (Clustering):** Spectral clustering to group embeddings by speaker identity.
  - **Stage 4 & 5 (ASR & Integration):** Whisper transcription mapped to speaker clusters.
  - **Stage 6 (Evaluation):** Automated calculation of DER (via Hungarian matching) and WER.
  - **Stage 7 (Fine-Tuning):** Adapts the ECAPA-TDNN embeddings to the AMI domain by training a custom classification head via 4-Fold CV, dramatically improving $k$-estimation.

## Results & Complexity Challenges

Evaluated on the held-out `ES2006a` meeting using the fine-tuned ECAPA head and **Whisper Medium**:

- **WER Improvement:** Upgrading to Whisper Medium reduced the Word Error Rate by **~6%**, demonstrating strong resilience to accents and technical jargon.
- **DER Bottlenecks:** The Diarization Error Rate slightly regressed during testing. This highlights the extreme complexity of the AMI dataset.

### The Complexity of the AMI Dataset
The dataset represents one of the most difficult challenges in speech processing. Several factors cap the current performance of traditional clustering algorithms:
1. **Severe Speaker Overlap:** There is roughly an **18% rate of simultaneous overlapping speech**. Traditional clustering inherently assigns only *one* speaker label to a segment, meaning it mathematically cannot resolve moments where two people are talking over each other.
2. **Spontaneous Speech:** Participants trail off, use broken grammar, and speak at wildly varying speeds.
3. **Filler Words:** Short filler words (*"emm"*, *"ehhh"*) and laughter lack enough phonetic information to generate reliable embeddings.
4. **Acoustic Bleed:** Loud speakers "bleed" into the microphones of quieter speakers sitting next to them, corrupting the clean audio signal.

## Future Directions
1. **End-to-End Neural Diarization:** Moving toward an overlap-aware framework like **Pyannote** would allow the pipeline to natively assign multiple speaker labels to a single timestamp, completely resolving the 18% overlap constraint.
2. **Scaled Infrastructure:** Securing access to High-RAM GPUs (such as an A100) would prevent Colab OOM crashes, allowing the pipeline to train on the entire corpus and successfully run Whisper `large-v3`.