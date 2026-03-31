## translator assist

Build a local, self-hosted program with the following pipeline:

1. **Input**

   * Accept audio input as either:

     * a file path (wav/mp3), or
     * live microphone stream (optional, if feasible)
   * Normalize audio to 16kHz mono

2. **Transcription**

   * Use Whisper (local inference)
   * Model size should be configurable (default: `small`)
   * Input language is explicitly provided: `en` or `es`
   * Disable auto-detection and force the provided language

3. **Translation**

   * Translate transcribed text into the *other* language:

     * `en → es`
     * `es → en`
   * Support two backends:

     * NLLB (default, higher quality)
     * Marian/OPUS (optional, faster/lighter)
   * Backend selection should be configurable

4. **Output**

   * Return structured JSON:
     {
     "transcription": "...",
     "translation": "...",
     "source_lang": "...",
     "target_lang": "..."
     }

5. **Interface**

   * Provide a CLI:

     * `--input <file>`
     * `--source-lang en|es`
     * `--backend nllb|opus`
   * Optional: REST API (FastAPI)

6. **Constraints**

   * Fully offline (no external APIs)
   * Must run on CPU; use GPU if available
   * Optimize for low latency (<2s for short audio if possible)

7. **Dependencies**

   * Whisper (local)
   * Transformers (for NLLB) OR Marian models

Ask clarifying questions if:

* microphone streaming is required
* GPU availability is unknown
* batching or real-time processing is needed
