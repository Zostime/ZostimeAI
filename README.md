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
│   ├─core/                     # Core
│   │   ├─__init__.py      
│   │   ├─common/               # Public module
│   │   │  ├─__init__.py        
│   │   │  ├─config.py          
│   │   │  └─logger.py          
│   │   ├─llm/                  # LLM module
│   │   │   ├─__init__.py   
│   │   │   └─client.py     
│   │   ├─prompts/              # PROMPTS module
│   │   │   ├─init.py   
│   │   │   ├─system.md         # system prompt
│   │   │   ├─personality.md    # personality prompt
│   │   │   ├─memory.md         # memory prompt
│   │   │   ├─runtime_state.md  # runtime state prompt
│   │   │   └─builder.py        # prompts builder
│   │   ├─tts/                  # TTS module
│   │   │   ├─__init__.py    
│   │   │   └─client.py      
│   │   ├─stt/                  # STT module
│   │   │   ├─__init__.py    
│   │   │   └─client.py      
│   │   └─memory/               # Memory systrm
│   │       ├─__init__.py   
│   │       ├─manager.py        
│   │       ├─stm.py   
│   │       └─ltm.py   
│   ├─data/                     # Data, config and backups
│   │   ├─models/        
│   │   ├─cache/          
│   │   ├─config/         
│   │   ├─logs/                
│   │   ├─memories/
│   │   │   ├─short_term
│   │   │   └─long_term
│   │   └─backups/
│   ├─scripts/                  # Scripts
│   │   ├─system/
│   │   │   └─reset_system.py   # Reset System
│   │   └─backup/
│   │       ├─restore.py        # Restore 
│   │       └─backup.py         # Backup
│   ├─.venv/                    # Python virtual environment
│   ├─main.py                   # Program entry point
│   └─requirements.txt          # Project dependencies
├─airis-sdk/                    # SDK
│   └─python/                   
│       ├─airis_sdk/            # Python package
│       │   ├─__init__.py      
│       │   ├─websocket.py  
│       │   └─devtools/ 
│       │      └─websocket_proxy.py   
│       └─pyproject.toml            
├─docs/                         # Document
│   ├─airis-api/
│   └─...                      
└─server/                       # External services
    └─...  
```
## Quick Start
- **Copy [.env.example](Airis/data/config/.env.example) as .env and fill in the actual values**
- **Open your [Airis](./Airis) folder with PowerShell**
- **Create virtual environment (first time)**
- `python -m venv .venv`
- **Activate virtual environment (Windows)**
- `.\.venv\Scripts\activate`
- **Install dependencies**
- `pip install -r requirements.txt`
- **Run**
- `python main.py`