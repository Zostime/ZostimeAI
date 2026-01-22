"""
Airis - LLM-TTS 流处理交互系统
主程序入口
"""

import os
import sys
import threading
import time
import queue
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root / "LLM" / "Python"))
sys.path.append(str(project_root / "TTS" / "Python"))

# 导入LLM和TTS模块
try:
    from llm import LLMClient
    from tts import TTSClient
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保项目结构正确:")
    print("1. LLM模块位于: LLM/Python/llm.py")
    print("2. TTS模块位于: TTS/Python/tts.py")
    sys.exit(1)


class StreamCoordinator:
    """LLM和TTS流式处理协调器"""

    def __init__(self):
        """初始化协调器"""
        # 初始化LLM客户端
        print("正在初始化LLM客户端...")
        self.llm_client = LLMClient()

        # 初始化TTS客户端
        print("正在初始化TTS客户端...")
        self.tts_client = TTSClient()

        # 流式处理状态
        self.is_processing = False
        self.current_task_id = None
        self.processing_lock = threading.Lock()

        # 消息队列
        self.message_queue = queue.Queue()
        self.response_buffer = ""

        # 回调函数
        self.on_llm_chunk = None  # LLM文本块回调
        self.on_tts_audio = None  # TTS音频块回调
        self.on_complete = None  # 完成回调

        # 统计信息
        self.stats = {
            "llm_calls": 0,
            "tts_calls": 0,
            "total_chars": 0,
            "total_audio_ms": 0
        }

        print("✓ 系统初始化完成")

    def process_user_input(self, user_input: str) -> bool:
        """
        处理用户输入，启动LLM-TTS流式处理链

        Args:
            user_input: 用户输入的文本

        Returns:
            bool: 是否成功启动处理
        """
        with self.processing_lock:
            if self.is_processing:
                print("当前已有处理任务进行中，请等待...")
                return False

            if not user_input or user_input.strip() == "":
                print("输入为空，请重新输入")
                return False

            print(f"\n[处理开始] 输入: {user_input[:50]}..." if len(
                user_input) > 50 else f"\n[处理开始] 输入: {user_input}")

            # 重置状态
            self.is_processing = True
            self.current_task_id = f"task_{int(time.time())}"
            self.response_buffer = ""

            # 创建处理线程
            process_thread = threading.Thread(
                target=self._process_chain,
                args=(user_input,),
                daemon=True,
                name=f"Process-{self.current_task_id}"
            )
            process_thread.start()

            return True

    def _process_chain(self, user_input: str):
        """
        LLM-TTS处理链：LLM流式生成 → TTS流式合成

        Args:
            user_input: 用户输入
        """
        try:
            # 步骤1: 构建LLM消息
            messages = [
                {"role": "system", "content": "你叫Airis,创作者是Zostime,请尽量简短地回答用户的提问."},
                {"role": "user", "content": user_input}
            ]

            # 步骤2: LLM流式生成
            print("\n[LLM] 正在生成回答...")

            # 收集LLM响应的完整文本
            llm_response = ""

            # 流式处理LLM响应
            try:
                # 使用生成器获取流式响应
                stream_gen = self.llm_client.chat_stream(messages)

                # 处理每个流式块
                for chunk in stream_gen:
                    if isinstance(chunk, str):
                        llm_response += chunk

                        # 更新响应缓冲区
                        self.response_buffer = llm_response

                        # 调用LLM回调
                        if self.on_llm_chunk:
                            try:
                                self.on_llm_chunk(chunk)
                            except Exception as e:
                                print(f"LLM回调错误: {e}")

                        # 可选：实时打印（不推荐，因为LLM模块已打印）
                        # print(chunk, end="", flush=True)

                # 获取最终结果
                final_result = next(stream_gen)
                if isinstance(final_result, dict):
                    llm_response = final_result.get('full_content', llm_response)

            except StopIteration as e:
                # 从StopIteration异常中获取结果
                if e.value and isinstance(e.value, dict):
                    final_result = e.value
                    llm_response = final_result.get('full_content', llm_response)

            # 检查是否生成了有效响应
            if not llm_response or llm_response.strip() == "":
                print("LLM未生成有效响应")
                self._finish_processing(success=False)
                return

            # 更新统计
            self.stats["llm_calls"] += 1
            self.stats["total_chars"] += len(llm_response)

            print(f"\n[LLM] 生成完成 (长度: {len(llm_response)}字符)")
            print(f"[TTS] 开始语音合成...")

            # 步骤3: TTS流式合成
            # 准备TTS回调
            def tts_callback(audio_data: bytes):
                """TTS音频数据回调"""
                if self.on_tts_audio:
                    try:
                        self.on_tts_audio(audio_data)
                    except Exception as e:
                        print(f"TTS回调错误: {e}")

            # 启动TTS流式合成
            success = self.tts_client.text_to_speech_stream(llm_response, tts_callback)

            if not success:
                print("TTS流式合成启动失败")
                self._finish_processing(success=False)
                return

            # 等待TTS合成完成
            print("[TTS] 正在合成语音...")

            if self.tts_client.wait_for_completion(60):  # 60秒超时
                self.stats["tts_calls"] += 1
                print("✓ TTS合成完成")
                self._finish_processing(success=True, response=llm_response)
            else:
                print("✗ TTS合成超时")
                self._finish_processing(success=False)

        except Exception as e:
            print(f"处理链错误: {e}")
            import traceback
            traceback.print_exc()
            self._finish_processing(success=False)

    def _finish_processing(self, success: bool, response: str = None):
        """完成处理任务"""
        with self.processing_lock:
            self.is_processing = False

            if success and response:
                print(f"\n[处理完成] 响应: {response[:100]}..." if len(
                    response) > 100 else f"\n[处理完成] 响应: {response}")

            # 调用完成回调
            if self.on_complete:
                try:
                    self.on_complete(success, response, self.current_task_id)
                except Exception as e:
                    print(f"完成回调错误: {e}")

            # 重置当前任务
            self.current_task_id = None

    def stop_processing(self):
        """停止当前处理任务"""
        with self.processing_lock:
            if not self.is_processing:
                return

            print("\n[正在停止处理...]")

            # 停止TTS合成
            if hasattr(self, 'tts_client') and self.tts_client:
                self.tts_client.stop_stream()

            # 重置状态
            self.is_processing = False
            self.current_task_id = None

            print("✓ 处理已停止")

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "is_processing": self.is_processing,
            "current_task_id": self.current_task_id,
            "llm_status": None,
            "tts_status": None,
            "stats": self.stats.copy()
        }

        # 获取LLM状态
        if hasattr(self, 'llm_client'):
            status["llm_status"] = {
                "provider": getattr(self.llm_client, 'provider', 'unknown'),
                "model": getattr(self.llm_client, 'model', 'unknown')
            }

        # 获取TTS状态
        if hasattr(self, 'tts_client') and self.tts_client:
            try:
                tts_status = self.tts_client.get_status()
                status["tts_status"] = {
                    "provider": tts_status.get('provider', 'unknown'),
                    "is_streaming": tts_status.get('is_streaming', False),
                    "is_playing": tts_status.get('is_playing', False)
                }
            except:
                status["tts_status"] = {"error": "获取状态失败"}

        return status

    def register_callbacks(self,
                           llm_chunk_callback=None,
                           tts_audio_callback=None,
                           complete_callback=None):
        """
        注册回调函数

        Args:
            llm_chunk_callback: LLM文本块回调函数
            tts_audio_callback: TTS音频块回调函数
            complete_callback: 处理完成回调函数
        """
        self.on_llm_chunk = llm_chunk_callback
        self.on_tts_audio = tts_audio_callback
        self.on_complete = complete_callback


def main():
    """主函数 - 交互式LLM-TTS系统"""
    print("=" * 70)
    print("Airis - LLM-TTS 流处理交互系统")
    print("=" * 70)
    print("功能:")
    print("1. 用户输入文本")
    print("2. LLM流式生成回答")
    print("3. TTS流式合成语音")
    print("4. 实时播放语音")
    print("=" * 70)

    try:
        # 初始化协调器
        coordinator = StreamCoordinator()

        # 示例回调函数
        def on_llm_chunk(chunk):
            """LLM文本块回调示例"""
            # 这里可以添加实时显示或其他处理
            pass

        def on_tts_audio(audio_data):
            """TTS音频块回调示例"""
            # 这里可以添加音频处理或可视化
            pass

        def on_complete(success, response, task_id):
            """处理完成回调示例"""
            if success:
                print(f"\n✓ 任务 {task_id} 完成")
                print(f"  响应长度: {len(response)} 字符")
            else:
                print(f"\n✗ 任务 {task_id} 失败")

        # 注册回调
        coordinator.register_callbacks(
            llm_chunk_callback=on_llm_chunk,
            tts_audio_callback=on_tts_audio,
            complete_callback=on_complete
        )

        while True:
            print("\n" + "-" * 70)
            print("请选择操作:")
            print("1. 输入文本进行对话")
            print("2. 查看系统状态")
            print("3. 停止当前处理")
            print("4. 退出程序")

            try:
                choice = input("\n请选择 (1-4): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n检测到中断信号，正在退出...")
                coordinator.stop_processing()
                break

            if choice == "1":
                # 获取用户输入
                try:
                    user_input = input("\n请输入您的消息 (按Ctrl+C取消): ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n输入已取消")
                    continue

                if not user_input:
                    print("输入不能为空")
                    continue

                # 处理用户输入
                success = coordinator.process_user_input(user_input)

                if success:
                    print("处理已启动，请等待...")

                    # 等待处理完成（可选，可以异步）
                    # 这里简单等待几秒，实际应用中应该是异步的
                    max_wait = 120  # 最大等待2分钟
                    start_time = time.time()

                    while coordinator.is_processing:
                        time.sleep(0.5)

                        # 显示进度点
                        elapsed = time.time() - start_time
                        if elapsed > max_wait:
                            print("\n处理超时，正在停止...")
                            coordinator.stop_processing()
                            break

                        # 每5秒显示一次状态
                        if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                            print(".", end="", flush=True)

                else:
                    print("处理启动失败")

            elif choice == "2":
                # 显示系统状态
                status = coordinator.get_status()

                print("\n" + "=" * 50)
                print("系统状态")
                print("=" * 50)

                print(f"正在处理: {'是' if status['is_processing'] else '否'}")
                if status['current_task_id']:
                    print(f"当前任务: {status['current_task_id']}")

                if status['llm_status']:
                    print(f"\nLLM状态:")
                    print(f"  提供商: {status['llm_status']['provider']}")
                    print(f"  模型: {status['llm_status']['model']}")

                if status['tts_status']:
                    print(f"\nTTS状态:")
                    print(f"  提供商: {status['tts_status']['provider']}")
                    print(f"  正在流式合成: {'是' if status['tts_status'].get('is_streaming') else '否'}")
                    print(f"  正在播放: {'是' if status['tts_status'].get('is_playing') else '否'}")

                print(f"\n统计信息:")
                print(f"  LLM调用次数: {status['stats']['llm_calls']}")
                print(f"  TTS调用次数: {status['stats']['tts_calls']}")
                print(f"  总处理字符数: {status['stats']['total_chars']}")
                print("=" * 50)

            elif choice == "3":
                # 停止当前处理
                if coordinator.is_processing:
                    coordinator.stop_processing()
                else:
                    print("当前没有处理任务")

            elif choice == "4":
                # 退出程序
                print("\n正在退出程序...")
                coordinator.stop_processing()
                break

            else:
                print("无效选择，请重新输入")

    except Exception as e:
        print(f"\n程序发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()