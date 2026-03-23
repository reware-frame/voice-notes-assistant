# AI Voice Notes Structured Assistant

将语音笔记自动转换为结构化 Markdown，核心流程：
1. Whisper 转文字
2. GPT-4 智能分类与结构化
3. 标题/标签/优先级生成
4. Markdown 导出

## Tech Stack
- Python 3.10+
- OpenAI Whisper API (`whisper-1`)
- GPT-4 系列模型（默认 `gpt-4o`）

## Project Structure
```text
voice-notes-assistant/
├─ src/
│  ├─ __init__.py
│  ├─ transcriber.py
│  ├─ processor.py
│  ├─ formatter.py
│  └─ cli.py
├─ tests/
│  ├─ test_transcriber.py
│  ├─ test_processor.py
│  ├─ test_formatter.py
│  └─ test_cli.py
├─ docs/
│  └─ architecture.md
├─ requirements.txt
├─ .env.example
└─ README.md
```

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，至少配置：

```bash
OPENAI_API_KEY=your_openai_api_key
```

## CLI Usage

从音频到 Markdown（完整链路）：

```bash
python -m src.cli transcribe ./samples/voice-note.m4a -o ./output/note.md
```

可选参数示例：

```bash
python -m src.cli transcribe ./audio.wav \
  --language zh \
  --whisper-model whisper-1 \
  --gpt-model gpt-4o \
  --hint "这是一次产品会议记录"
```

直接处理文本：

```bash
python -m src.cli process-text "明天上午 10 点和设计团队开会，确认首页改版优先级" -o note.md
```

## Run Tests
```bash
pytest -q
```
