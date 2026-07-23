# ROAD project-page asset map

All paths are relative, so the page works at `https://luomayao.github.io/ROAD3D/` when GitHub Pages publishes `ghpage /docs`.

## Paper figures

- `assets/figures/teaser.webp` — hero/teaser image
- `assets/figures/pipeline.webp` — method pipeline
- `assets/figures/efficiency.webp` — efficiency comparison
- `assets/figures/representation-gap.webp` — representation analysis
- `assets/figures/convergence.webp` — convergence plot
- `assets/figures/classification.webp` — feature analysis
- `assets/figures/geometry.webp`, `quality.webp`, `demo.webp` — qualitative results
- `assets/paper/ROAD.pdf` — paper PDF

## Latest public-code release assets

- `assets/release/overview.png` — source-release overview image copied from `assets/000.png`
- `assets/release/turntables/001.gif` … `006.gif` — six turntable GIFs from the latest source snapshot

## Interactive showcase

- `assets/showcase/<ID>/model.glb` — interactive 3D model
- `assets/showcase/<ID>/condition.webp` — conditioning image
- `assets/showcase/<ID>/preview.webp` — model-viewer poster and thumbnail

Example `001` is the updated ROAD027 asset. Keep filenames unchanged to replace assets without editing HTML. To add more interactive examples, add the files and append an entry to `showcaseItems` in `script.js`.
