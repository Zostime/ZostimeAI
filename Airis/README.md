# Airis
**This is an AI project created by Zostime.**

------------

## File structure
```
Airis/
├─core/                 # 所有核心业务逻辑
│   ├─llm/              # LLM模块
│   │   ├─__init__.py
│   │   └─client.py  
│   ├─tts/              # TTS模块
│   │   ├─__init__.py
│   │   └─client.py  
│   └─memory/           # 记忆系统
│       ├─__init__.py
│       ├─manager.py    
│       ├─stm.py
│       └─ltm.py
├─Files/                # 数据和配置
│   ├─cache/
│   ├─config/
│   ├─logs/
│   └─storage/
├─.venv/                # Python虚拟环境
├─main.py               # 程序主入口
└─requirements.txt      # 项目依赖
```
### LLM
- Deepseek
- https://platform.deepseek.com/usage
### TTS
- xfyun super smart-tts
- https://console.xfyun.cn/services/uts

## Start
- Copy [.env.example](./Files/config/.env.example) as .env and fill in the actual values
- Open your [Airis](../Airis) folder with Powershell 
- Run `pip install -r requirements.txt`
- Run `python main.py`