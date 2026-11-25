# Cache-transfer architecture graphics

- [Interactive motion player](architecture.html): self-contained HTML with inline SVG; play/pause, replay, scrubbing, five stage explanations and a native-control toggle.
- [Animated SVG](cache-transfer-animated.svg): standalone 1600x1050 SVG, with a 12-second SMIL loop. No JavaScript or external resources in the SVG.
- [Static SVG](cache-transfer-static.svg): motion-free vector poster.
- [PNG poster](cache-transfer-poster.png): 2000-pixel-wide raster export, visually reviewed.
- [WebP poster](cache-transfer-poster.webp): lossless diagram export.

The graphic distinguishes the native comparator from the proposed cold-transfer path. It shows the complete timing boundary, an offline pair-specific mapper and the proposed S3/SageMaker execution route. Motion illustrates sequence only: it does not encode benchmark timings, a measured speedup or established quality parity.

The currently prepared code supports dense full-attention Qwen3. The hybrid Qwen3.5/3.8 pair needs recurrent/convolution-state support. No GPU job or smoke suite is launched when opening these graphics.

Rebuild source: `python3 visuals/build_architecture.py` from the parent project. PNG/WebP exports were rendered with CairoSVG and Pillow in an isolated uv environment, using the existing Homebrew Cairo library. Opening the player requires no dependencies, server or API key.
