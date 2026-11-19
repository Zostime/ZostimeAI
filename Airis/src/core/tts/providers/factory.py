from .base import ProviderCallbacks, TTSProvider


def create_provider(callbacks: ProviderCallbacks, config) -> TTSProvider:
    name = config.get_json("tts.provider").strip().lower()
    if name == "azure":
        from .azure import AzureTTSProvider
        return AzureTTSProvider(callbacks, config)
    if name == "edge":
        from .edge import EdgeTTSProvider
        return EdgeTTSProvider(callbacks, config)
    raise ValueError(f"Unknown TTS provider: {name!r}")
