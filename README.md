# Mini Clip Builder (CLI + Frontend Web)

Agora o projeto tem:
- **Pipeline backend** para gerar mini clip de 15–20s.
- **Frontend Streamlit** para upload via browser e visualização do resultado.

## O que faz

1. Recebe uma música.
2. Detecta trecho de destaque (energia RMS + fluxo espectral).
3. Transcreve com timestamps (Whisper).
4. Faz matching do contexto da letra com banco de mídia (**imagem/gif/vídeo**).
5. Sobrepõe legenda sincronizada e gera mini clip vertical.

## Requisitos

- Python 3.10+
- ffmpeg no sistema

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Frontend (browser)

```bash
streamlit run app.py
```

No app web você pode:
- enviar a música,
- enviar banco de mídia (jpg/png/webp/gif/mp4/mov/webm/mkv),
- editar tags por arquivo,
- processar,
- ver preview e baixar o resultado.

## Uso via CLI

```bash
python clip_prototype.py \
  --song ./musica.mp3 \
  --media ./assets/media \
  --metadata ./metadata.json \
  --output ./mini_clip.mp4 \
  --language pt \
  --target-seconds 18 \
  --whisper-model small
```

### Exemplo metadata.json

```json
{
  "praia_sunset.jpg": {
    "caption": "praia no fim de tarde",
    "tags": ["praia", "sol", "calma"]
  },
  "cidade_noite.mp4": {
    "caption": "cidade molhada de chuva",
    "tags": ["cidade", "noite", "chuva"]
  }
}
```
