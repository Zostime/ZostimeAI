# Airis
**This is an AI project created by Zostime.**

------------

## File structure
```
Airis/
├─airis_sdk/                # SDK
│   ├─__init__.py     
│   ├─core/  
│   │  └─websocket.py   
│   └devtools/ 
│      └─websocket_proxy.py   
├─core/                     # Core
│   ├─__init__.py      
│   ├─common/               # Public module
│   │  ├─__init__.py        
│   │  ├─config.py          
│   │  └─logger.py          
│   ├─llm/                  # LLM module
│   │   ├─__init__.py   
│   │   └─client.py     
│   ├─prompts/              # PROMPTS module
│   │   ├─init.py   
│   │   ├─system.md         # system prompt
│   │   ├─personality.md    # personality prompt
│   │   ├─memory.md         # memory prompt
│   │   ├─runtime_state.md  # runtime state prompt
│   │   └─builder.py        # prompts builder
│   ├─tts/                  # TTS module
│   │   ├─__init__.py    
│   │   └─client.py      
│   ├─stt/                  # STT module
│   │   ├─__init__.py    
│   │   └─client.py      
│   └─memory/               # Memory systrm
│       ├─__init__.py   
│       ├─manager.py        
│       ├─stm.py   
│       └─ltm.py   
├─data/                    # Data and config
│   ├─models/        
│   ├─cache/          
│   ├─config/         
│   ├─logs/          
│   └─memories/           
│       ├─short_term      
│       └─long_term  
├─server/                   # External services
│   └─...  
├─.venv/                    # Python virtual environment
├─main.py                   # Program entry point
└─requirements.txt          # Project dependencies
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