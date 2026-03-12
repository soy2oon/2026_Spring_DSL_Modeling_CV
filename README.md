## Multimodal Coach (Webcam + Speech)

### Structure
- `src/multimodal_coach`: core package
- `apps/run_multimodal_coach.py`: unified webcam runner
- `assets/`: reference video/audio/subtitles/derived pose data
- `experiments/`: legacy pilots and gesture FSL project
- `tests/`: unit tests

### Run
```bash
PYTHONPATH=src python apps/run_multimodal_coach.py
```

### API (speech feedback)
```bash
PYTHONPATH=src uvicorn multimodal_coach.api.feedback_server:app --reload
```
