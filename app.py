#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from clip_prototype import MEDIA_EXTENSIONS, _infer_tags_from_filename, process_clip

st.set_page_config(page_title="Mini Clip Builder", layout="wide")
st.title("🎬 Mini Clip Builder (música + mídia + legenda)")

st.write("Faça upload da música, monte o banco de mídia (imagem/gif/vídeo), revise tags e gere o mini clip.")

song = st.file_uploader("1) Música", type=["mp3", "wav", "m4a", "mp4", "ogg"])
media_files = st.file_uploader(
    "2) Banco de mídia (imagens, gifs, vídeos)",
    type=["jpg", "jpeg", "png", "webp", "gif", "mp4", "mov", "webm", "mkv"],
    accept_multiple_files=True,
)

col1, col2, col3 = st.columns(3)
language = col1.selectbox("Idioma", ["pt", "en", "es"], index=0)
whisper_model = col2.selectbox("Modelo Whisper", ["tiny", "base", "small", "medium"], index=2)
target_seconds = col3.slider("Duração alvo", min_value=15, max_value=20, value=18)

if media_files:
    st.subheader("3) Tags da mídia")
    st.caption("As tags iniciais são inferidas do nome do arquivo. Você pode editar livremente.")

    for idx, mf in enumerate(media_files):
        ext = Path(mf.name).suffix.lower()
        if ext not in MEDIA_EXTENSIONS:
            continue
        default_tags = ", ".join(_infer_tags_from_filename(Path(mf.name)))
        st.text_input(f"Tags para {mf.name}", value=default_tags, key=f"tags_{idx}")

if st.button("🚀 Processar e gerar mini clip", type="primary"):
    if not song:
        st.error("Envie uma música primeiro.")
        st.stop()
    if not media_files:
        st.error("Envie ao menos 1 item de mídia.")
        st.stop()

    with st.spinner("Processando... isso pode levar alguns minutos"):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            song_path = base / song.name
            song_path.write_bytes(song.read())

            media_dir = base / "media"
            media_dir.mkdir(parents=True, exist_ok=True)

            metadata = {}
            for idx, mf in enumerate(media_files):
                file_path = media_dir / mf.name
                file_path.write_bytes(mf.read())
                raw_tags = st.session_state.get(f"tags_{idx}", "")
                tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
                metadata[mf.name] = {"caption": "", "tags": tags}

            metadata_path = base / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

            output_path = base / "mini_clip.mp4"
            result = process_clip(
                song_path=str(song_path),
                media_dir=str(media_dir),
                output_path=str(output_path),
                metadata_json=str(metadata_path),
                language=language,
                target_seconds=target_seconds,
                whisper_model=whisper_model,
            )

            video_bytes = output_path.read_bytes()

    st.success("Mini clip gerado com sucesso!")
    st.video(video_bytes)
    st.download_button("⬇️ Baixar mini clip", data=video_bytes, file_name="mini_clip.mp4", mime="video/mp4")

    st.subheader("Resumo da seleção")
    st.write(f"Trecho detectado: {result['start']:.2f}s → {result['end']:.2f}s ({result['duration']:.2f}s)")
    for i, (chunk, _, _) in enumerate(result["chunks"]):
        st.write(f"**Legenda {i+1}:** {chunk}")
        st.caption(f"Mídia escolhida: {Path(result['selected_media'][i]['path']).name}")
