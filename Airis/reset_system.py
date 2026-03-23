import os
import shutil

DelTargetPath = [

]

DelDirectories = [
    "./Files/memories/long_term",
    "./Files/cache/llm/conversations",
    "./Files/cache/tts",
    "./Files/logs",
]

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if input("真的要初始化Airis吗?[y/n]\n")=="y":
        # 处理单个文件
        for TargetPath in DelTargetPath:
            try:
                os.remove(TargetPath)
                print(f"位于\"{TargetPath}\"的文件删除成功！")
            except FileNotFoundError:
                if os.path.exists(TargetPath):
                    shutil.rmtree(TargetPath)
                    print(f"位于\"{TargetPath}\"的文件夹删除成功！")
                else:
                    print(f"位于\"{TargetPath}\"的文件/文件夹不存在，无法删除！")

        # 处理目录下的所有文件
        for directory in DelDirectories:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'删除{file_path}时出错: {e}')
                print(f"位于\"{directory}\"的目录内容已清空!")
            else:
                print(f"位于\"{directory}\"的目录不存在,无法清空!")