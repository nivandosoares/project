# Protótipo funcional: mini clip automático (15–20s)

Este projeto gera um mini clip curto a partir de uma música, com seleção automática de trecho de destaque, legenda sincronizada e imagens escolhidas por contexto da letra.

## O que o pipeline faz

1. **Identifica o trecho “alto”** da faixa (energia RMS + fluxo espectral).
2. **Recorta 15–20s** do áudio nesse trecho.
3. **Transcreve** o trecho com timestamps por palavra (`faster-whisper`).
4. **Quebra as legendas** em blocos curtos sincronizados.
5. **Seleciona imagens** por similaridade textual entre letra e metadados do banco de imagens.
6. **Renderiza um mini clip** vertical `1080x1920` com áudio + legenda.

## Requisitos

- Python 3.10+
- `ffmpeg` no sistema

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Estrutura de imagens

```text
assets/images/
  praia_por_do_sol.jpg
  cidade_chuva.png
```

`metadata.json` (opcional):

```json
{
  "praia_por_do_sol.jpg": {
    "caption": "praia no fim de tarde",
    "tags": ["sol", "mar", "calma"]
  },
  "cidade_chuva.png": {
    "caption": "cidade com chuva à noite",
    "tags": ["noite", "triste", "rua"]
  }
}
```

## Uso

```bash
python clip_prototype.py \
  --song ./musica.mp3 \
  --images ./assets/images \
  --metadata ./metadata.json \
  --output ./mini_clip.mp4 \
  --language pt \
  --target-seconds 18 \
  --whisper-model small
```

## Notas

- O alvo de duração é sempre limitado para **15..20 segundos**.
- Sem metadados, o matching usa apenas o nome dos arquivos.
- Para maior velocidade, use `--whisper-model tiny` ou `base`.
