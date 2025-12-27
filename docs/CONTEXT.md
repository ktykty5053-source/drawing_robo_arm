# Project Context

This repository combines a simple Pencil Recorder web client with the behavior-cloning training code for the robotic arm pencil recorder model.

## Layout
- `web/`: Static assets for the web interface. Deployed via GitHub Pages.
- `ml/pencil_recorder_bc/`: Training and inference code for the Pencil Recorder model.
- `README.md`: Quickstart and usage instructions.

## Notes
- Large artifacts such as datasets, training runs, and model checkpoints are ignored by Git. Place them in `ml/pencil_recorder_bc/data/`, `ml/pencil_recorder_bc/runs/`, or similar subdirectories as needed.
- The web client is currently a placeholder HTML file to be expanded in future iterations.
