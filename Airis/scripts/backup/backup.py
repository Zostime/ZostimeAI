from core.common.config import ConfigManager
from datetime import datetime
import zipfile
import os

def zip_folders(input_paths, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in input_paths:
            if os.path.isdir(p):
                base = os.path.basename(os.path.normpath(p))
                for root, dirs, files in os.walk(p):
                    for file in files:
                        fp = os.path.join(root, file)
                        arc = os.path.join(base, os.path.relpath(fp, p))
                        zf.write(fp, arc)
            elif os.path.isfile(p):
                zf.write(p, os.path.basename(p))
            else:
                raise RuntimeError(f"路径不存在或无效: {p}")

if __name__ == '__main__':
    CONFIG = ConfigManager()

    backup_dir = CONFIG.get_path("backup.dir")
    include = CONFIG.get_json(f"backup.include")
    paths = []
    for name in include:
        paths.append(CONFIG.get_path(f"backup.paths.{name}"))

    backup_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target_path = backup_dir / f"backup_{timestamp}.zip"
    zip_folders(paths, target_path)
    print(f"已成功备份在: {target_path}")

