from speech_transcriber.transcription.base import TranscriberCapabilities
from speech_transcriber.transcription.cohere.transcriber import CohereTranscriber


class Recognizer:
    model_reference = "/models/cohere"
    device = "cpu"
    dtype_name = "float32"

    def load(self) -> None:
        pass

    def recognize(self, segment: object) -> str:
        return ""

    def release(self) -> None:
        pass


class Aligner:
    model_reference = "/models/aligner"
    max_segment_duration = 300.0

    def load(self) -> None:
        pass

    def align(self, segment: object, transcript: str) -> list[object]:
        return []

    def release(self) -> None:
        pass


def test_cohere_transcriber_describes_forced_alignment_configuration() -> None:
    transcriber = CohereTranscriber(Recognizer(), Aligner(), language="en-US")
    assert transcriber.capabilities == TranscriberCapabilities(
        True, True, True, True, requires_forced_alignment=True
    )
    assert transcriber.backend_models == {"forced_aligner_model": "/models/aligner"}
    assert transcriber.backend_configuration == {
        "segment_duration_seconds": 30.0,
        "segment_overlap_seconds": 5.0,
        "forced_aligner_max_segment_seconds": 300.0,
        "language": "en",
        "punctuation": True,
        "max_new_tokens": 256,
    }
