#!/usr/bin/env python3
"""Pipeline de mini clip (15-20s): música -> trecho alto -> mídia contextual -> legenda."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass
class WordTimestamp:
    text: str
    start: float
    end: float


@dataclass
class SegmentPlan:
    start: float
    end: float
    words: List[WordTimestamp]


STOPWORDS_PT = {
    "a", "o", "e", "de", "do", "da", "em", "um", "uma", "com", "que", "pra", "para",
    "na", "no", "nos", "nas", "por", "eu", "tu", "ele", "ela", "nós", "você", "vocês",
}

MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".mov", ".webm", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}


def _tokenize(text: str) -> List[str]:
    import re

    tokens = re.findall(r"[\wÀ-ÿ]+", (text or "").lower())
    return [token for token in tokens if token not in STOPWORDS_PT and len(token) > 2]


def _assert_ffmpeg_available() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg não encontrado no PATH. Instale ffmpeg antes de executar.")


def detect_highlight_window(audio_path: str, target_seconds: int = 18) -> Tuple[float, float]:
    import librosa
    import numpy as np

    target_seconds = max(15, min(20, int(target_seconds)))
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = len(y) / sr if len(y) else 0.0
    if duration <= target_seconds:
        return 0.0, max(duration, 0.1)

    hop = 512
    frame = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    min_len = min(len(rms), len(onset_env))
    rms, onset_env = rms[:min_len], onset_env[:min_len]

    rms_n = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)
    flux_n = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min() + 1e-9)
    salience = 0.7 * rms_n + 0.3 * flux_n

    frames_per_win = max(1, int((target_seconds * sr) / hop))
    summed = np.convolve(salience, np.ones(frames_per_win, dtype=np.float32), mode="valid")
    best_frame = int(np.argmax(summed)) if len(summed) else 0

    start = best_frame * hop / sr
    end = min(duration, start + target_seconds)
    return float(start), float(end)


def transcribe_with_timestamps(
    audio_path: str,
    start: float,
    end: float,
    language: str = "pt",
    model_name: str = "small",
) -> List[WordTimestamp]:
    try:
        from faster_whisper import WhisperModel
        from pydub import AudioSegment
    except Exception as exc:
        raise RuntimeError("Dependências de transcrição ausentes. Rode pip install -r requirements.txt") from exc

    clip = AudioSegment.from_file(audio_path)[int(start * 1000): int(end * 1000)]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_wav = Path(tmp.name)
    clip.export(str(temp_wav), format="wav")

    try:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(temp_wav), language=language, word_timestamps=True, vad_filter=True)
        words: List[WordTimestamp] = []
        for segment in segments:
            for word in segment.words or []:
                if word.word and word.word.strip():
                    words.append(
                        WordTimestamp(
                            text=word.word.strip(),
                            start=float(start + word.start),
                            end=float(start + word.end),
                        )
                    )
    finally:
        temp_wav.unlink(missing_ok=True)

    return words or [WordTimestamp(text="[instrumental]", start=start, end=end)]


def split_words_into_caption_chunks(words: Sequence[WordTimestamp], max_chunk_seconds: float = 2.8) -> List[Tuple[str, float, float]]:
    if not words:
        return []

    chunks: List[Tuple[str, float, float]] = []
    current = [words[0].text]
    chunk_start = words[0].start
    chunk_end = words[0].end

    for word in words[1:]:
        if (word.end - chunk_start) <= max_chunk_seconds:
            current.append(word.text)
            chunk_end = word.end
        else:
            chunks.append((" ".join(current), chunk_start, chunk_end))
            current = [word.text]
            chunk_start, chunk_end = word.start, word.end

    chunks.append((" ".join(current), chunk_start, chunk_end))
    return chunks


def _infer_tags_from_filename(path: Path) -> List[str]:
    import re

    base = path.stem.replace("_", " ").replace("-", " ")
    tokens = [t for t in re.split(r"\s+", base.strip()) if t]
    return _tokenize(" ".join(tokens))


def load_media_bank(media_dir: str, metadata_json: str | None = None) -> List[dict]:
    directory = Path(media_dir)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Diretório de mídia inválido: {media_dir}")

    metadata = {}
    if metadata_json and Path(metadata_json).exists():
        metadata = json.loads(Path(metadata_json).read_text(encoding="utf-8"))

    items: List[dict] = []
    for file in directory.iterdir():
        ext = file.suffix.lower()
        if ext not in MEDIA_EXTENSIONS:
            continue
        info = metadata.get(file.name, {})
        tags = info.get("tags") or _infer_tags_from_filename(file)
        caption = info.get("caption", "")
        media_type = "video" if ext in VIDEO_EXTENSIONS else "image"
        searchable = " ".join([file.stem, caption, *tags])
        items.append({"path": str(file), "type": media_type, "searchable": searchable, "tags": tags})

    if not items:
        raise ValueError("Banco de mídia vazio. Adicione imagens/gifs/vídeos.")
    return items


def score_media_for_text(text: str, media_searchable: str) -> float:
    t1 = set(_tokenize(text))
    t2 = set(_tokenize(media_searchable))
    if not t1 or not t2:
        return 0.0
    overlap = len(t1 & t2)
    jaccard = overlap / max(1, len(t1 | t2))
    return overlap + jaccard


def choose_media_for_chunks(chunks: Sequence[Tuple[str, float, float]], media_bank: Sequence[dict]) -> List[dict]:
    chosen: List[dict] = []
    last_path = None
    for text, _, _ in chunks:
        ranked = sorted(media_bank, key=lambda item: score_media_for_text(text, item["searchable"]), reverse=True)
        pick = ranked[0]
        if pick["path"] == last_path and len(ranked) > 1:
            pick = ranked[1]
        chosen.append(pick)
        last_path = pick["path"]
    return chosen


def _build_visual_clip(media_item: dict, duration: float, width: int, height: int):
    from moviepy.editor import ImageClip, VideoFileClip

    path = media_item["path"]
    ext = Path(path).suffix.lower()

    if ext in VIDEO_EXTENSIONS or ext == ".gif":
        clip = VideoFileClip(path).without_audio().subclip(0, duration)
        return clip.resize(height=height).on_color(size=(width, height), color=(0, 0, 0), pos=("center", "center")).set_duration(duration)

    return (
        ImageClip(path)
        .set_duration(duration)
        .resize(height=height)
        .on_color(size=(width, height), color=(0, 0, 0), pos=("center", "center"))
    )


def render_video(
    song_path: str,
    plan: SegmentPlan,
    chunks: Sequence[Tuple[str, float, float]],
    chosen_media: Sequence[dict],
    output_path: str,
) -> None:
    from moviepy.editor import AudioFileClip, CompositeVideoClip, TextClip, concatenate_videoclips

    width, height = 1080, 1920
    clips = []

    for idx, (text, c_start, c_end) in enumerate(chunks):
        duration = max(0.2, c_end - c_start)
        visual = _build_visual_clip(chosen_media[idx], duration, width, height)
        subtitle = (
            TextClip(
                text,
                fontsize=56,
                color="white",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(int(width * 0.9), None),
                align="center",
                font="DejaVu-Sans",
            )
            .set_position(("center", int(height * 0.78)))
            .set_duration(duration)
        )
        clips.append(CompositeVideoClip([visual, subtitle]).set_duration(duration))

    video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(song_path).subclip(plan.start, plan.end)
    video.set_audio(audio).write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", preset="medium")


def process_clip(
    song_path: str,
    media_dir: str,
    output_path: str,
    metadata_json: str | None = None,
    language: str = "pt",
    target_seconds: int = 18,
    whisper_model: str = "small",
) -> dict:
    _assert_ffmpeg_available()

    start, end = detect_highlight_window(song_path, target_seconds=target_seconds)
    words = transcribe_with_timestamps(song_path, start, end, language=language, model_name=whisper_model)
    chunks = split_words_into_caption_chunks(words)

    media_bank = load_media_bank(media_dir, metadata_json)
    chosen_media = choose_media_for_chunks(chunks, media_bank)

    plan = SegmentPlan(start=start, end=end, words=list(words))
    render_video(song_path, plan, chunks, chosen_media, output_path)

    return {
        "output": output_path,
        "start": start,
        "end": end,
        "duration": end - start,
        "chunks": chunks,
        "selected_media": chosen_media,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera mini clip com música + banco de mídia + legenda sincronizada")
    parser.add_argument("--song", required=True, help="Arquivo de música/vídeo")
    parser.add_argument("--media", required=True, help="Diretório de mídia (imagem/gif/vídeo)")
    parser.add_argument("--output", default="mini_clip.mp4", help="Arquivo de saída")
    parser.add_argument("--metadata", default=None, help="JSON opcional com tags/caption")
    parser.add_argument("--language", default="pt", help="Idioma da transcrição")
    parser.add_argument("--target-seconds", type=int, default=18, help="Duração alvo (forçada para 15..20)")
    parser.add_argument("--whisper-model", default="small", help="tiny/base/small/medium")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    info = process_clip(
        song_path=args.song,
        media_dir=args.media,
        output_path=args.output,
        metadata_json=args.metadata,
        language=args.language,
        target_seconds=args.target_seconds,
        whisper_model=args.whisper_model,
    )
    print(f"✅ Mini clip gerado: {info['output']}")
    print(f"Janela escolhida: {info['start']:.2f}s -> {info['end']:.2f}s ({info['duration']:.2f}s)")
