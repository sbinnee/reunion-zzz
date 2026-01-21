#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from openai import OpenAI

logger = logging.getLogger("generate_report")


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


def generate_meeting_notes(client: OpenAI, transcript_text: str) -> str:
    logger.info("Generating meeting notes and summary")
    system_prompt = (
        "You are a meeting assistant. Generate concise meeting notes and a short summary. "
        "Respond in Korean. Use clear headings and bullet points. "
        "Include action items and decisions if present."
    )
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript_text},
        ],
    )
    return response.output_text.strip()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate meeting notes from a transcript file."
    )
    parser.add_argument(
        "--transcript",
        required=True,
        help="Path to transcript file (JSON or TXT).",
    )
    parser.add_argument(
        "--output",
        help="Output path for meeting notes (default: meeting_notes.md in same dir).",
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

    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    if transcript_path.suffix == ".json":
        with transcript_path.open("r", encoding="utf-8") as f:
            transcription = json.load(f)
        transcript_text = render_transcript(transcription)
    else:
        transcript_text = transcript_path.read_text(encoding="utf-8")

    if not transcript_text.strip():
        raise ValueError("Transcript is empty")

    logger.info("Loaded transcript from: %s", transcript_path)

    client = OpenAI()
    notes = generate_meeting_notes(client, transcript_text)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = transcript_path.parent / "meeting_notes.md"

    output_path.write_text(notes + "\n", encoding="utf-8")
    logger.info("Saved meeting notes: %s", output_path)


if __name__ == "__main__":
    main()
