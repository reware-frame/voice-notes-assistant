# AI Voice Notes Assistant

一个将语音笔记自动转换为结构化笔记的 AI 工具。

## 核心功能

- 🎤 **语音转文字**: 使用 Whisper API 高精度转录
- 🧠 **智能分类**: 自动识别内容类型（想法/待办/会议/灵感）
- 📝 **结构化输出**: 生成带标题、标签、优先级的格式化笔记
- 🔗 **知识关联**: 语义相似度检测，自动建立双向链接
- 💭 **情感分析**: 识别语音中的情绪线索
- ⏰ **智能提醒**: 待办自动同步到日历

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 添加 OpenAI API Key

# 运行 CLI
python -m src.cli transcribe audio.mp3
```

## 项目结构

```
voice-notes-assistant/
├── src/
│   ├── __init__.py
│   ├── transcriber.py    # Whisper 转录
│   ├── processor.py      # GPT-4 结构化处理
│   ├── formatter.py      # Markdown 格式化
│   ├── knowledge_graph.py # 知识关联
│   └── cli.py            # 命令行接口
├── tests/
│   ├── test_transcriber.py
│   ├── test_processor.py
│   └── test_formatter.py
├── docs/
│   └── architecture.md
├── config/
│   └── settings.py
├── scripts/
│   └── setup.sh
├── requirements.txt
├── .env.example
└── README.md
```

## 技术栈

- **ASR**: OpenAI Whisper
- **LLM**: GPT-4 / Claude
- **向量数据库**: Pinecone (可选)
- **集成**: Notion API / Obsidian

## License

MIT
