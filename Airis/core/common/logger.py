"""
日志管理器
"""
from logging import handlers
import logging
import sys

from .config import ConfigManager    #配置管理器

class LogManager:
    def __init__(self, service_name: str):
        """初始化日志管理器

        Args:
            service_name: 服务名称，用于日志文件命名
        """
        self.config = ConfigManager()
        self.service_name = service_name
        self.log_dir = None
        self._setup_logging()

    def _setup_logging(self):
        """设置日志系统"""
        # 获取日志配置
        log_level_str = self.config.get_json('logging.level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        # 设置日志目录
        log_dir = self.config.get_path(
            f"logging.{self.service_name}_dir",
        )
        # 创建日志目录
        log_dir.mkdir(exist_ok=True)

        #配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        #清除现有处理器
        root_logger.handlers.clear()

        #创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        #控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        #文件处理器(按大小轮转)
        log_file = log_dir / f"{self.service_name.lower()}.log"
        max_size_mb = int(self.config.get_json('logging.max_file_size_mb', 10))
        backup_count = int(self.config.get_json('logging.backup_count', 5))

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"{self.service_name}日志初始化完成")
        logging.info(f"日志级别:{log_level_str}")