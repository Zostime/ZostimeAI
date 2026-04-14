# Airis
**这是一个由Zostime建立的AI项目.**

------------

## 文件结构
```
Airis/
├─core/                    # 核心
│   ├─__init__.py    
│   ├─common/              # 公共模块
│   │  ├─__init__.py  
│   │  ├─config.py       
│   │  └─logger.py        
│   ├─llm/                 # LLM模块
│   │   ├─__init__.py  
│   │   └─client.py    
│   ├─prompts/             # PROMPTS模块
│   │   ├─__init__.py
│   │   ├─system.md        # 系统prompt
│   │   ├─personality.md   # 人格prompt
│   │   ├─memory.md        # 记忆prompt
│   │   ├─runtime_state.md # 运行时状态prompt
│   │   └─builder.py       # prompts构建器 
│   ├─tts/                 # TTS模块
│   │   ├─__init__.py  
│   │   └─client.py    
│   ├─stt/                 # STT模块
│   │   ├─__init__.py  
│   │   └─client.py    
│   ├─memory/              # 记忆系统
│   │   ├─__init__.py  
│   │   ├─manager.py       
│   │   ├─stm.py  
│   │   └─ltm.py  
│   └─tools/               # 工具系统
│       ├─__init__.py  
│       ├─registry.py      
│       └─...  
├─Files/                   # 数据和配置
│   ├─models/        
│   ├─cache/          
│   ├─config/         
│   ├─logs/          
│   └─memories/           
│       ├─short_term      
│       └─long_term  
├─.venv/                   # Python虚拟环境
├─main.py                  # 程序主入口
└─requirements.txt         # 项目依赖
```
## 开始
- 复制 [.env.example](./Files/config/.env.example) 为 .env 后填充真实值
- 用PowerShell 打开 [Airis](../Airis) 文件夹
- 创建虚拟环境(首次运行时执行)
- `python -m venv .venv`
- **激活虚拟环境**
- `.\.venv\Scripts\activate`
- **安装依赖**
- `pip install -r requirements.txt`
- **运行**
- `python main.py`