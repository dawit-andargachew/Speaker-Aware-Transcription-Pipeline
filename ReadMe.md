The initial idea was:
Meeting Audio → Speaker Diarization (Pyannote) → Automatic Speech Recognition (Whisper) → Speaker-Labeled Transcript → Evaluation

However, my understanding is that, rather than relying on end-to-end tools such as Pyannote, I should implement the main stages of the pipeline myself, building on the concepts covered in our class. The pipeline would therefore include components such as Voice Activity Detection (VAD), speech segmentation, speaker embedding extraction, speaker clustering, timeline reconstruction, and automatic speech recognition. The system will be developed and evaluated on a subset of the AMI Meeting Corpus, using metrics such as DER for speaker attribution and WER for transcription quality.

I also plan to compare alternative approaches for some of these components (e.g., speaker embeddings, clustering methods, segmentation strategies, and potentially ASR models) and analyze how these choices affect overall performance. The goal is not only to obtain speaker-labeled transcripts, but also to understand the contribution of each stage of the pipeline and justify the design decisions made throughout the project. I will also ensure that the implementation remains clean, readable, and well documented.

> then my professor said this as a comment:
Hi,
yes ok. 
VAD -> Embedding -> Clustering. Then ASR with time-boundaries and identity. You can use a simple hierarchical clustering (you find the tool in sklearn I think) and a state-of-the-art VAD (there are functions in librosa for example, or you can use something more complex like Silero). Focus on different embeddings, consider a couple of different pre-trained models.

<br/>

# About the structure of the code

It may be better to adopt a modular approach rather than putting everything together at once and ending up with spaghetti code. Start by identifying the main sections or components that will be needed, and implement them individually. A functional approach is likely sufficient here; using classes may be unnecessary and could introduce additional complexity and overhead.


### A good starting point would be this
- Meeting Audio
- Speaker Diarization (implementing speaker diarization from scartch, Pyannote is ready-made so we can't use it)
- Automatic Speech Recognition (try out Whisper and other ASR models)
- Speaker-Labeled Transcript
- Evaluation

Once each section is developed separately, they can be integrated into a single notebook or project with minimal effort. This makes debugging and testing much easier, as each component can be verified independently. It also improves code readability and helps make the overall pipeline and data flow easier to understand.

Finally, a well-structured modular design will make it much easier to write the report and present the project, since the logic and progression of the work will already be clearly organized.

I'm thinking of having separate notebooks for each of the main sections, and then a final notebook that integrates everything together. This way, I can focus on one component at a time, and then easily combine them once they are all working correctly. Each notebook can also include detailed explanations and comments to ensure that the code is clear and understandable.

## My Proposed Structure

- `01_vad_segmentation.ipynb` — VAD (Silero is the right pick per your professor), outputs speech segments with timestamps
    - Load a raw AMI audio file (headset-mix, mono, 16kHz)
    - Run Silero VAD to detect speech vs. silence regions
    - Merge short gaps between speech segments (tunable threshold)
    - Visualize the waveform with speech/silence regions overlaid
    - Save output: list of `(start_sec, end_sec)` segments to disk

- `02_embeddings.ipynb` — extract speaker embeddings per segment, compare 2+ models (e.g., ECAPA-TDNN vs. x-vectors)
    - Load audio + segments from notebook 01
    - For each segment, extract a fixed-size speaker embedding
    - Implement this for **2 models** (e.g., ECAPA-TDNN via SpeechBrain, and x-vector via a pretrained checkpoint)
    - Visualize embedding space with UMAP or t-SNE (colored by ground truth speaker if you want a sanity check)
    - Save output: `(segment_id, start, end, embedding_vector)` per model

- `03_clustering.ipynb` — agglomerative clustering, compare linkage strategies, output speaker labels per segment
    - Load embeddings from notebook 02
    - Run agglomerative clustering (sklearn), compare at least 2 linkage strategies (ward, average)
    - Experiment with number of clusters (known vs. estimated via dendrogram)
    - Visualize dendrograms and cluster assignments
    - Save output: `(segment_id, start, end, speaker_label)` — effectively your RTTM
- `04_asr.ipynb` — Whisper on each segment with timestamp-aligned input, output raw transcripts
    - Load audio + segments (with timestamps) from notebook 01
    - Run Whisper on each segment individually, passing the timestamp offset so output is time-aligned
    - Optionally compare a second ASR model (e.g., wav2vec2 fine-tuned on English)
    - Save output: `(segment_id, start, end, transcript_text)`
- `05_integration.ipynb` — stitch speaker labels + transcripts into speaker-labeled output
    - Load speaker labels (notebook 03) + transcripts (notebook 04)
    - Join on segment ID → `(start, end, speaker_label, transcript_text)`
    - Format into a readable speaker-labeled transcript
    - Print/display a clean example output so you can eyeball quality
- `06_evaluation.ipynb` — DER, WER, analysis, comparison tables
    - Load your RTTM output (notebook 03) + AMI reference RTTM
    - Compute **DER** using pyannote.metrics or dscore
    - Load your transcripts (notebook 04) + AMI reference transcripts
    - Compute **WER** using jiwer
    - Build a comparison table across your embedding and clustering variants
    - Plot **DER/WER** tradeoffs, error breakdown (missed speech, false alarm, confusion)
