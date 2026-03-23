# Architecture

## Pipeline
1. `Transcriber` uses Whisper (`audio.transcriptions.create`) to convert audio to plain text.
2. `Processor` sends transcript to GPT-4 and requests strict JSON output.
3. `MarkdownFormatter` converts structured fields into standard markdown sections.
4. `CLI` orchestrates the pipeline and writes output files.

## Modules
- `src/transcriber.py`: validates file + invokes Whisper.
- `src/processor.py`: prompts GPT-4, validates JSON, normalizes categories/priority.
- `src/formatter.py`: renders markdown metadata/summary/action items/source transcript.
- `src/cli.py`: command routing (`transcribe`, `process-text`).

## Error Handling
- Missing API key: fail fast in CLI.
- Missing/unsupported audio: raised by `Transcriber`.
- Invalid JSON from model: converted to `ValueError` in `Processor`.
- All CLI commands return exit code `0` or `1` for easy automation.
