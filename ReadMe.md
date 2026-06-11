The initial idea was:
Meeting Audio → Speaker Diarization (Pyannote) → Automatic Speech Recognition (Whisper) → Speaker-Labeled Transcript → Evaluation

However, my understanding is that, rather than relying on end-to-end tools such as Pyannote, I should implement the main stages of the pipeline myself, building on the concepts covered in our class. The pipeline would therefore include components such as Voice Activity Detection (VAD), speech segmentation, speaker embedding extraction, speaker clustering, timeline reconstruction, and automatic speech recognition. The system will be developed and evaluated on a subset of the AMI Meeting Corpus, using metrics such as DER for speaker attribution and WER for transcription quality.

I also plan to compare alternative approaches for some of these components (e.g., speaker embeddings, clustering methods, segmentation strategies, and potentially ASR models) and analyze how these choices affect overall performance. The goal is not only to obtain speaker-labeled transcripts, but also to understand the contribution of each stage of the pipeline and justify the design decisions made throughout the project. I will also ensure that the implementation remains clean, readable, and well documented.

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
