# TTS
### Methods that should be implemented in client.py
- TTSClient class
    - subtitle_queue: queue.Queue\[str\]
        - When the current word of TTS audio is played, put() the current word, including invisible characters like '\\n', ' '
    - feed(text: str)
        - Used for non-blocking text input to be played by TTS
    - interrupt()
        - Used to interrupt the current audio output, subtitles, clear the audio stream, and synthesis queue
    - close()
        - Shutdown
