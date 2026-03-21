# Airis
**This is an AI project created by Zostime.**

------------

## File structure
```
Airis/
├─core/                 # Core
│   ├─common/           # Public module
│   │  ├─__init__.py    
│   │  ├─config.py      
│   │  └─logger.py      
│   ├─llm/              # LLM module
│   │   ├─__init__.py
│   │   └─client.py  
│   ├─tts/              # TTS module
│   │   ├─__init__.py
│   │   └─client.py  
│   ├─stt/              # STT module
│   │   ├─__init__.py
│   │   └─client.py  
│   └─memory/           # Memory System
│       ├─__init__.py
│       ├─manager.py    
│       ├─stm.py
│       └─ltm.py
├─Files/                # Data and config
│   ├─models/           # Models
│   ├─cache/            # Cache
│   ├─config/           # Config
│   └─logs/             # Logs
├─.venv/                # Python virtual environment
├─main.py               # Program entry point
└─requirements.txt      # Project dependencies
```
## Start
- Copy .env.example as .env and fill in the actual values
- Open your Airis folder with PowerShell
- Create virtual environment (run only the first time)
- `python -m venv .venv`
- Activate virtual environment
- `.\.venv\Scripts\activate`
- Install dependencies
- `pip install -r requirements.txt`
- Run
- `python main.py`