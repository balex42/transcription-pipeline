"""Future-stage contracts; no LLM or external-service implementation is included."""

from __future__ import annotations

from typing import Protocol

from meeting_transcriber.models import Transcript


class PostProcessor(Protocol):
    """Optional transcript transformation, such as punctuation restoration."""

    def process(self, transcript: Transcript) -> Transcript: ...


class Summarizer(Protocol):
    """Optional future meeting summary producer."""

    def summarize(self, transcript: Transcript) -> object: ...


class SpeakerIdentifier(Protocol):
    """Optional future mapping from anonymous speaker IDs to names."""

    def identify(self, transcript: Transcript) -> Transcript: ...


class ObjectStorage(Protocol):
    """Optional future blob-storage boundary."""

    def put(self, key: str, data: bytes) -> None: ...


class JobStatusReporter(Protocol):
    """Optional future asynchronous-job status boundary."""

    def report(self, status: str) -> None: ...
