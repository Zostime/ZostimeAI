from core.common.config import ConfigManager
import shutil
import zipfile
from pathlib import Path

def find_path(base, f_name):
    direct = base / f_name
    if direct.exists():
        return direct

    for p in base.rglob("*"):
        if p.name == f_name:
            return p
    return None

def clear_dir(dir_path: Path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

def restore_file(include_name: str, f_name: str):
    src = find_path(temp_dir, f_name)
    if src is None or not src.exists():
        print(f"{include_name} not found in backup, skipped")
        return
    dst = CONFIG.get_path(f"backup.paths.{include_name}")

    if src.exists():
        try:
            shutil.copy2(src, dst)
            print(f"{include_name} restore success")

        except Exception as e:
            print(f"{include_name} restore failed: {e}")

def restore_dir(include_name: str, f_name: str):
    src = find_path(temp_dir, f_name)
    if src is None or not src.exists():
        print(f"{include_name} not found in backup, skipped")
        return
    dst = CONFIG.get_path(f"backup.paths.{include_name}")
    backup = dst.parent / f"_{include_name}_backup"

    if src.exists():
        try:
            if backup.exists():
                shutil.rmtree(backup)
            tmp_dst = dst.parent / f"_{include_name}_new"

            if tmp_dst.exists():
                shutil.rmtree(tmp_dst)
            shutil.move(src, tmp_dst)

            if dst.exists():
                shutil.move(dst, backup)

            shutil.move(tmp_dst, dst)

            print(f"{include_name} restore success")

            if backup.exists():
                shutil.rmtree(backup)

        except Exception as e:
            print(f"{include_name} restore failed: {e}")
            if backup.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(backup, dst)
                print(f"{include_name} rolled back")


if __name__ == '__main__':
    CONFIG = ConfigManager()

    backup_dir = CONFIG.get_path("backup.dir")
    temp_dir = backup_dir / "_restore_temp"

    include = CONFIG.get_json("backup.include")

    clear_dir(temp_dir)

    file_names = [
        f.name
        for f in sorted(
            backup_dir.glob("backup_*.zip"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        if f.is_file()
    ]
    if not file_names:
        print("无备份文件")
        exit()
    print(f"选择备份文件")
    i = 1
    for file_name in file_names:
        print(f"{i}.{file_name}{'(最新)' if i==1 else ''}")
        i += 1

    while True:
        user_input = input("选择编号或输入文件名: ").strip()

        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(file_names):
                zip_path = backup_dir / file_names[idx]
                break

        zip_path = backup_dir / user_input
        if zip_path.exists():
            break

        print("\n无效输入,请重新输入\n")

    print("\n将恢复以下内容:")
    for name in include:
        print(f"- {name}")
    confirm = input(f"\n确认恢复 {zip_path.name}? (y/n): ").lower()
    if confirm != 'y':
        print("已取消")
        exit()

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
    except Exception as error:
        print(f"解压失败: {error}")
        exit()

    for name in include:
        path = CONFIG.get_path(f"backup.paths.{name}")

        if path.is_dir():
            restore_dir(name, path.name)
        else:
            restore_file(name, path.name)

    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Restore completed")