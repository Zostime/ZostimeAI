# Airis
**This is an AI project created by Zostime.**

---

<p>
  [<a href="./docs/README.zh-CN.md">简体中文</a>]
</p>

## File structure
```
ZostimeAI/
├─Airis/
│   ├─src/
│   │   ├─__init__.py
│   │   ├─gateway.py                # WebSocket protocol gateway
│   │   ├─state.py                  # Runtime state
│   │   ├─event_bus.py              # Priority Scheduling Event Bus
│   │   ├─output.py                 # Output channels
│   │   └─core/                     # Core
│   │       ├─__init__.py
│   │       ├─common/               # Public module
│   │       │   ├─__init__.py
│   │       │   ├─config.py
│   │       │   └─logger.py
│   │       ├─llm/                  # LLM module
│   │       │   ├─__init__.py
│   │       │   └─client.py
│   │       ├─prompts/              # PROMPTS module
│   │       │   ├─__init__.py
│   │       │   ├─manager.py
│   │       │   ├─system.md         # system prompt
│   │       │   ├─personality.md    # personality prompt
│   │       │   ├─memory.md         # memory prompt
│   │       │   ├─runtime_state.md  # runtime state prompt
│   │       │   └─environmental_assessment.md
│   │       ├─tts/                  # TTS module
│   │       │   ├─__init__.py
│   │       │   ├─client.py
│   │       │   └─providers/        # TTS providers
│   │       │       ├─base.py
│   │       │       └─factory.py
│   │       ├─stt/                  # STT module
│   │       │   ├─__init__.py
│   │       │   └─client.py
│   │       └─memory/               # Memory system
│   │           ├─__init__.py
│   │           ├─manager.py
│   │           ├─stm.py
│   │           ├─ltm.py
│   │           └─note.py
│   ├─data/                         # Data and configuration
│   │   └─config/
│   │       ├─config.json
│   │       └─.env.example
│   ├─scripts/                      # Scripts
│   │   ├─system/
│   │   │   └─reset_system.py       # Reset System
│   │   └─backup/
│   │       ├─restore.py            # Restore
│   │       └─backup.py             # Backup
│   ├─main.py                       # Program entry point
│   └─requirements.txt              # Project dependencies
├─airis-sdk/                        # SDK
│   ├─python/
│   │   ├─airis_sdk/                # Python package
│   │   │   ├─__init__.py
│   │   │   ├─websocket.py
│   │   │   └─devtools/
│   │   │       └─websocket_proxy.py
│   │   └─pyproject.toml
│   └─typescript/                   # Typescript package
│       ├─src/
│       │   └─index.ts
│       ├─dist/
│       ├─package.json
│       └─tsconfig.json
├─docs/                             # Documentation
│   ├─airis-api/
│   │   ├─zh-Hans/
│   │   └─en/
│   ├─README.md
│   └─README.zh-CN.md
└─services/                         # External services
    └─...
```
## Quick Start
- **Copy [.env.example](Airis/data/config/.env.example) as .env and fill in the actual values**
- **Open your [Airis](/Airis) folder with PowerShell**
- **Create virtual environment (first time)**
- `python -m venv .venv`
- **Activate virtual environment (Windows)**
- `.\.venv\Scripts\activate`
- **Install dependencies**
- `pip install -r requirements.txt`
- **Run**
- `python main.py`