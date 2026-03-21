# Airis
**This is an AI project created by Zostime.**

------------

## File structure
```
Airis/
├─core/                 # 核心
│   ├─llm/              # LLM模块
│   │   ├─__init__.py
│   │   └─client.py  
│   ├─tts/              # TTS模块
│   │   ├─__init__.py
│   │   └─client.py  
│   ├─stt/              # STT模块
│   │   ├─__init__.py
│   │   └─client.py  
│   └─memory/           # 记忆系统
│       ├─__init__.py
│       ├─manager.py    
│       ├─stm.py
│       └─ltm.py
├─Files/                # 数据和配置
│   ├─models/           # 模型
│   ├─cache/            # 缓存
│   ├─config/           # 配置
│   └─logs/             # 日志
├─.venv/                # Python虚拟环境
├─main.py               # 程序主入口
└─requirements.txt      # 项目依赖
```
## Start
- Copy [.env.example](./Files/config/.env.example) as .env and fill in the actual values
- Open your [Airis](../Airis) folder with PowerShell 
- Run `pip install -r requirements.txt`
- Run `python main.py`