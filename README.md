# German Meeting Transcriber

Local, batch-friendly German meeting transcription using pyannote Community-1 for anonymous speaker diarization and interchangeable Parakeet, Whisper, or Granite ASR. It intentionally contains no summarization, LLM post-processing, or API service.

## Architecture

```text
recording -> ffmpeg normalization -> 16 kHz mono WAV -> pyannote (whole meeting)
                                                        |
                                                 exclusive timeline
                                                        |
                                             release diarization model
                                                        |
normalized WAV -> overlapping AudioChunker -> selected ASR backend -> ChunkMerger
                                                                    |
                                          SpeakerAligner <- global words
                                                                    |
                                             TurnBuilder -> JSON / text exporters
```

The orchestrator depends on small protocols (`Diarizer`, `Transcriber`, `SpeakerAligner`, and `TranscriptExporter`) and constructor injection. Future `PostProcessor`, `Summarizer`, `SpeakerIdentifier`, `ObjectStorage`, and `JobStatusReporter` contracts are declared in `extensions.py`, without implementations.

## ASR Backends

- Parakeet: `nvidia/parakeet-tdt-0.6b-v3`. Multilingual (including German), native punctuation/capitalization, and start/end word timestamps. Loaded with native Transformers `AutoModelForTDT`.
- Whisper: `openai/whisper-large-v3`. German transcription is explicitly requested, with punctuation/capitalization and Transformers word timestamps.
- Granite: `ibm-granite/granite-speech-4.1-2b-plus`. Uses generated word-end tags and the project timestamp parser. It intentionally does not add ordinary punctuation/capitalization.
- Diarization: `pyannote/speaker-diarization-community-1`, run locally through `pyannote.audio` over the full normalized meeting. The adapter uses `output.exclusive_speaker_diarization`.

All adapters use deterministic inference, `model.eval()`, and `torch.inference_mode()`. CUDA selects BF16 on capable GPUs and FP16 otherwise; CPU uses FP32. ASR speaker labels are never used: pyannote owns canonical speaker identity.

## Prerequisites

- Python 3.11
- `ffmpeg` and `ffprobe` on `PATH`
- sufficient RAM, or an NVIDIA GPU for practical production inference
- a Hugging Face account and token for the gated pyannote model

Before first pyannote download, accept the user conditions at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), then create a Hugging Face token. Export it only at runtime:

```bash
export HF_TOKEN=hf_...
export HF_HOME=$PWD/cache/huggingface
```

## Python Setup

The exact runtime pins are:

| Package | Version |
| --- | --- |
| Python | `>=3.11,<3.14` |
| torch | `2.8.0` |
| torchaudio | `2.8.0` |
| torchcodec | `0.7.0` |
| transformers | `5.9.0` |
| accelerate | `1.12.0` |
| pyannote.audio | `4.0.7` |
| numpy | `2.2.6` |

Install the appropriate PyTorch CPU/CUDA wheel for the target platform first if your package index requires it, then install the project:

```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[dev]'
```

CPU example:

```bash
HF_TOKEN=hf_... python -m meeting_transcriber transcribe /data/meeting.m4a --asr parakeet \
  --output /data/result --working-directory /data/work --device cpu
```

NVIDIA GPU example:

```bash
HF_TOKEN=hf_... python -m meeting_transcriber transcribe /data/meeting.m4a --asr parakeet \
  --output /data/result --working-directory /data/work --device cuda
```

## CLI

```bash
python -m meeting_transcriber transcribe INPUT --output OUTPUT [options]
meeting-transcriber transcribe INPUT --output OUTPUT [options]
```

Options include `--asr parakeet|whisper|granite`, `--asr-model`, `--device auto|cuda|cpu`, `--granite-model` (legacy Granite alias), `--pyannote-model`, `--chunk-duration 180`, `--chunk-overlap 15`, `--num-speakers`, `--min-speakers`, `--max-speakers`, `--working-directory`, `--keep-intermediate`, and `--log-level`.

Local runs use `./work` by default. The container sets `WORKING_DIRECTORY=/work`; supply `--working-directory` when using another mounted work volume.

Precedence is CLI arguments, environment variables, then defaults. `ASR_BACKEND` defaults to `parakeet`; `ASR_MODEL` overrides its backend model default. `GRANITE_MODEL` remains a Granite-only compatibility alias. Other variables are `PYANNOTE_MODEL`, `DEVICE`, `WORKING_DIRECTORY`, `CHUNK_DURATION`, `CHUNK_OVERLAP`, `NUM_SPEAKERS`, `MIN_SPEAKERS`, `MAX_SPEAKERS`, `KEEP_INTERMEDIATE_FILES`, `LOG_LEVEL`, `HF_HOME`, and `HF_TOKEN`.

Input formats are decoded by ffmpeg and include WAV, MP3, M4A, MP4, WebM, and OGG. Every source becomes a temporary PCM WAV at 16 kHz, mono, 16-bit. The original is never modified. Temporary files are removed after success unless `--keep-intermediate` is supplied.

## Output Schema

`OUTPUT/transcript.json` is canonical and versioned (`"version": "1.0"`). It contains source metadata, backend/model metadata, anonymous `SPEAKER_XX` IDs, derived turns, and raw speaker-attributed words. `start_is_inferred: true` documents approximate Granite starts; Parakeet and Whisper retain native word starts. `OUTPUT/transcript.txt` is optimized for review with a timestamp/speaker header and unmodified turn text.

With `--keep-intermediate`, `OUTPUT/intermediate/diarization.json`, `asr_words.json`, and `attributed_words.json` are also written.

Chunks default to 180 seconds with 15 seconds overlap. The `ChunkMerger` owns the first half of an overlap from the earlier chunk and the second half from the later chunk, avoiding duplicated words deterministically.

## Offline Models

Populate the Hugging Face cache with:

```bash
HF_TOKEN=hf_... meeting-transcriber prefetch-models --asr parakeet
```

For explicit local paths, obtain Granite with Hugging Face tooling and clone the accepted pyannote repository with Git LFS:

```bash
git lfs install
git clone https://huggingface.co/pyannote/speaker-diarization-community-1 /models/pyannote
```

Then run entirely from mounted local models:

```bash
ASR_BACKEND=whisper ASR_MODEL=/models/whisper PYANNOTE_MODEL=/models/pyannote \
  meeting-transcriber transcribe /data/meeting.m4a --output /data/result
```

The pipeline does not require internet after models are available locally. The pyannote repository must include its model artifacts as provided by the official offline distribution.

## Podman

Build:

```bash
podman build -t meeting-transcriber:local -f Containerfile .
```

CPU run with mounted input, output, working space, and persistent model/cache volume:

```bash
podman run --rm \
  -v $PWD/input:/input:ro,Z -v $PWD/result:/output:Z \
  -v $PWD/cache:/cache:Z -v $PWD/work:/work:Z \
  -e HF_TOKEN -e HF_HOME=/cache/huggingface \
  meeting-transcriber:local transcribe /input/meeting.m4a --asr parakeet --output /output --device cpu
```

NVIDIA GPU run (with NVIDIA Container Toolkit configured for Podman):

```bash
podman run --rm --device nvidia.com/gpu=all \
  -v $PWD/input:/input:ro,Z -v $PWD/result:/output:Z \
  -v $PWD/cache:/cache:Z -v $PWD/work:/work:Z \
  -e HF_TOKEN -e HF_HOME=/cache/huggingface \
  meeting-transcriber:local transcribe /input/meeting.m4a --asr whisper --output /output --device cuda
```

The image runs non-root, writes only to `/work`, `/cache`, mounted output, or mounted models, accepts an arbitrary OpenShift UID through root-group permissions, contains no token, and forwards SIGTERM to the Python process through its exec-form entrypoint.

## OpenShift

`deploy/openshift/job.example.yaml` is a minimal batch Job example with `restartPolicy: Never`, no privileged context, an arbitrary non-root UID-compatible image, mounted input/output/model volumes, secret-backed token injection, and an `nvidia.com/gpu` request. It deliberately leaves object storage and asynchronous orchestration outside this application.

## Manual Comparison

Run normalization and diarization once, then execute requested ASR models sequentially:

```bash
meeting-transcriber compare meeting01.m4a \
  --models parakeet,whisper,granite \
  --output ./comparison/meeting01 \
  --device cuda
```

The result contains `diarization.json`, operational `metadata.json`, and one directory per backend with `transcript.json`, review-friendly `transcript.txt`, and `asr_words.json`. The metadata records model load/transcription duration, RTF, selected dtype, and CUDA peak memory. These are operational measurements, not quality scores.

Inspect several representative meetings: multiple and overlapping speakers, technical terms/names, room and video-conference microphones, background noise, and mixed German/English terminology. Do not treat any backend as objectively best without representative recordings.

## Limitations

- Granite Plus does not add normal punctuation or capitalization; model text is preserved apart from safe whitespace normalization.
- Granite timestamps are generated word-end timestamps, not measured word starts.
- Parakeet native Transformers output is timestamped tokenizer pieces; the adapter groups them into words.
- Whisper word timestamps are model-generated and can drift around rapid or overlapping speech.
- ASR timestamps may contain timing error.
- pyannote speaker IDs are anonymous voices, not real names.
- Overlapping speech is inherently difficult.
- Speaker assignment around boundaries may be imperfect.
- German is the intended language for this application.

## Verification

Normal tests use no model downloads or GPU. Run:

```bash
pytest
ruff check .
mypy src/meeting_transcriber
```

An optional real-model smoke test needs predownloaded models, a German WAV, and `RUN_MODEL_TESTS=1 MODEL_TEST_BACKEND=parakeet MODEL_TEST_AUDIO=/path/to/audio.wav pytest tests/integration`.
