# Airis
**This is an AI project created by Zostime.**

------------

## File structure
```
D:\Airis\
├── .venv\              # 根虚拟环境
├── Files\
│   ├── cache\          # 缓存目录
│   │   ├── LLM\
│   │   └── TTS\
│   ├── config\         # 配置文件目录
│   │   ├── .env     
│   │   └── config.json  
│   └── logs\           # 日志目录
│       ├── LLM\
│       └── TTS\
├── LLM\
│   └── Python\      
│       └── llm.py     # LLM Python代码 
├── TTS\
│   └── Python\      
│       └── tts.py     # TTS Python代码
└── main.py             #主代码
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