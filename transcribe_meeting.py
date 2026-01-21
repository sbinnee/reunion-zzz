#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import logging
import time
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from openai import APIStatusError, OpenAI

logger = logging.getLogger("meeting_transcribe")


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _format_ts(seconds: float | int | None) -> str:
    if seconds is None:
        return "??:??"
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _to_data_url(path: Path) -> str:
    with path.open("rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    return "data:audio/wav;base64," + encoded


def transcribe_audio(
    client: OpenAI,
    audio_path: Path,
    language: str,
    diarize: bool,
    known_speaker_name: str | None,
    known_speaker_ref: Path | None,
    model_override: str | None,
    response_format_override: str | None,
    chunking_strategy: str | None,
    stream: bool,
) -> Any:
    extra_body: dict[str, Any] | None = None
    response_format = "verbose_json"
    model = "gpt-4o-mini-transcribe"
    if diarize:
        response_format = "diarized_json"
        model = "gpt-4o-transcribe-diarize"
        if known_speaker_name and known_speaker_ref:
            extra_body = {
                "known_speaker_names": [known_speaker_name],
                "known_speaker_references": [_to_data_url(known_speaker_ref)],
            }
    if model_override:
        model = model_override
    if response_format_override:
        response_format = response_format_override
    if stream:
        response_format = "text"
    with audio_path.open("rb") as audio_file:
        logger.info("Uploading audio for transcription: %s", audio_path.name)
        params: dict[str, Any] = {
            "model": model,
            "file": audio_file,
            "language": language,
            "response_format": response_format,
            "timestamp_granularities": ["segment"],
        }
        if diarize and chunking_strategy:
            params["chunking_strategy"] = chunking_strategy
        if extra_body:
            params["extra_body"] = extra_body
        if stream:
            params["stream"] = True
            events = client.audio.transcriptions.create(**params)
            return _collect_stream(events)
        return client.audio.transcriptions.create(**params)


def transcribe_with_retry(
    client: OpenAI,
    audio_path: Path,
    language: str,
    diarize: bool,
    known_speaker_name: str | None,
    known_speaker_ref: Path | None,
    model_override: str | None,
    response_format_override: str | None,
    chunking_strategy: str | None,
    max_retries: int,
    stream: bool,
) -> Any:
    attempt = 0
    while True:
        try:
            return transcribe_audio(
                client,
                audio_path,
                language,
                diarize,
                known_speaker_name,
                known_speaker_ref,
                model_override,
                response_format_override,
                chunking_strategy,
                stream,
            )
        except APIStatusError as exc:
            attempt += 1
            status = exc.status_code
            if status and status >= 500 and attempt <= max_retries:
                delay = min(2 ** attempt, 8)
                logger.info(
                    "Transcription failed with %s; retrying in %ss (attempt %s/%s)",
                    status,
                    delay,
                    attempt,
                    max_retries,
                )
                time.sleep(delay)
                continue
            raise


def _collect_stream(events: Any) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    texts: list[str] = []
    for event in events:
        event_type = _get_attr(event, "type", None)
        text = _get_attr(event, "text", None)
        if text:
            texts.append(text)
        if event_type == "transcript.text.delta":
            delta = _get_attr(event, "delta", None)
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
        segs = _get_attr(event, "segments", None)
        if segs:
            for segment in segs:
                segments.append(segment)
    return {"text": "".join(texts).strip(), "segments": segments}


def merge_transcriptions(items: list[tuple[float, Any]]) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    texts: list[str] = []
    for offset, transcription in items:
        text = _get_attr(transcription, "text", "") or ""
        if text.strip():
            texts.append(text.strip())
        for segment in _get_attr(transcription, "segments", []) or []:
            start = _get_attr(segment, "start", None)
            end = _get_attr(segment, "end", None)
            merged = {
                "start": start + offset if start is not None else None,
                "end": end + offset if end is not None else None,
                "text": _get_attr(segment, "text", "") or "",
            }
            speaker = _get_attr(segment, "speaker", None)
            if speaker is not None:
                merged["speaker"] = speaker
            speaker_id = _get_attr(segment, "speaker_id", None)
            if speaker_id is not None:
                merged["speaker_id"] = speaker_id
            channel = _get_attr(segment, "channel", None)
            if channel is not None:
                merged["channel"] = channel
            segments.append(merged)
    return {"text": "\n".join(texts).strip(), "segments": segments}


def split_audio(audio_path: Path, chunk_seconds: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "chunk_%03d.m4a"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        "-c",
        "copy",
        str(pattern),
    ]
    logger.info("Splitting audio into %ss chunks", chunk_seconds)
    subprocess.run(cmd, check=True)
    chunks = sorted(out_dir.glob("chunk_*.m4a"))
    if not chunks:
        raise RuntimeError("No audio chunks were created.")
    return chunks


def get_duration_seconds(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def render_transcript(transcription: Any) -> str:
    segments = _get_attr(transcription, "segments", None) or []
    if not segments:
        text = _get_attr(transcription, "text", "").strip()
        return text

    lines = []
    for segment in segments:
        start = _format_ts(_get_attr(segment, "start", None))
        end = _format_ts(_get_attr(segment, "end", None))
        speaker = (
            _get_attr(segment, "speaker", None)
            or _get_attr(segment, "speaker_id", None)
            or _get_attr(segment, "channel", None)
            or "Speaker"
        )
        text = (_get_attr(segment, "text", "") or "").strip()
        if text:
            lines.append(f"[{start} - {end}] {speaker}: {text}")
    return "\n".join(lines)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Transcribe and summarize a meeting audio file.")
    parser.add_argument("--audio", required=True, help="Path to the meeting audio file.")
    parser.add_argument("--language", default="ko", help="Language code for transcription.")
    parser.add_argument("--diarization", action="store_true", help="Enable diarization.")
    parser.add_argument(
        "--known-speaker-name",
        help="Speaker name to anchor diarization (requires --known-speaker-ref).",
    )
    parser.add_argument(
        "--known-speaker-ref",
        help="Path to a WAV reference for --known-speaker-name.",
    )
    parser.add_argument("--out-dir", default="outputs", help="Output directory.")
    parser.add_argument("--model", help="Override transcription model.")
    parser.add_argument("--response-format", help="Override response format.")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream transcription events (non-diarization only).",
    )
    parser.add_argument(
        "--chunking-strategy",
        default="auto",
        help="Chunking strategy for diarization (e.g., auto).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries for 5xx transcription errors.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=1200,
        help="Max chunk length in seconds when splitting audio.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    audio_path = Path(args.audio)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting transcription (diarization=%s, language=%s)",
        args.diarization,
        args.language,
    )
    client = OpenAI()
    known_speaker_ref = Path(args.known_speaker_ref) if args.known_speaker_ref else None
    if args.diarization:
        with tempfile.TemporaryDirectory(prefix="meeting_chunks_") as tmpdir:
            chunk_dir = Path(tmpdir)
            chunks = split_audio(audio_path, args.chunk_seconds, chunk_dir)
            transcribed: list[tuple[float, Any]] = []
            offset = 0.0
            if args.stream:
                logger.info("Streaming is disabled for diarization; continuing without stream.")
            for idx, chunk in enumerate(chunks, start=1):
                logger.info("Transcribing chunk %s/%s: %s", idx, len(chunks), chunk.name)
                result = transcribe_with_retry(
                    client,
                    chunk,
                    args.language,
                    args.diarization,
                    args.known_speaker_name,
                    known_speaker_ref,
                    args.model,
                    args.response_format,
                    args.chunking_strategy,
                    args.max_retries,
                    False,
                )
                transcribed.append((offset, result))
                offset += get_duration_seconds(chunk)
            transcription = merge_transcriptions(transcribed)
    elif args.stream:
        with tempfile.TemporaryDirectory(prefix="meeting_chunks_") as tmpdir:
            chunk_dir = Path(tmpdir)
            chunks = split_audio(audio_path, args.chunk_seconds, chunk_dir)
            transcribed = []
            offset = 0.0
            logger.info("Streaming transcription per chunk")
            for idx, chunk in enumerate(chunks, start=1):
                logger.info("Streaming chunk %s/%s: %s", idx, len(chunks), chunk.name)
                result = transcribe_with_retry(
                    client,
                    chunk,
                    args.language,
                    args.diarization,
                    args.known_speaker_name,
                    known_speaker_ref,
                    args.model,
                    args.response_format,
                    None,
                    args.max_retries,
                    True,
                )
                transcribed.append((offset, result))
                offset += get_duration_seconds(chunk)
            transcription = merge_transcriptions(transcribed)
    else:
        transcription = transcribe_with_retry(
            client,
            audio_path,
            args.language,
            args.diarization,
            args.known_speaker_name,
            known_speaker_ref,
            args.model,
            args.response_format,
            args.chunking_strategy if args.diarization else None,
            args.max_retries,
            args.stream,
        )
    logger.info("Transcription complete")

    transcript_json_path = out_dir / "transcript.json"
    transcript_text_path = out_dir / "transcript.txt"

    with transcript_json_path.open("w", encoding="utf-8") as f:
        if hasattr(transcription, "model_dump"):
            json.dump(transcription.model_dump(), f, ensure_ascii=False, indent=2)
        else:
            json.dump(transcription, f, ensure_ascii=False, indent=2)

    transcript_text = render_transcript(transcription)
    transcript_text_path.write_text(transcript_text, encoding="utf-8")

    logger.info("Saved transcript JSON: %s", transcript_json_path)
    logger.info("Saved transcript text: %s", transcript_text_path)


if __name__ == "__main__":
    main()
