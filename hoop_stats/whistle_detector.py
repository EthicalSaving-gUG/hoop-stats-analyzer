"""
Whistle Detector — Schiedsrichter-Pfiff-Erkennung.

Detects referee whistles in audio data using FFT frequency analysis.
Referee whistles typically produce a strong signal in the 2000-4500 Hz band.
"""
import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class WhistleDetectorConfig:
    """Configuration for the WhistleDetector."""
    sample_rate: int = 44100
    min_freq: float = 2000.0
    max_freq: float = 4500.0
    energy_threshold: float = 0.15
    peak_ratio_threshold: float = 3.0
    cooldown_seconds: float = 1.5


class WhistleDetector:
    """
    Analyses audio chunks for high-frequency whistle peaks.

    Usage:
        config = WhistleDetectorConfig(sample_rate=44100)
        wd = WhistleDetector(config)
        is_whistle = wd.detect(audio_chunk)  # np.ndarray of float32 samples
    """

    def __init__(self, config: WhistleDetectorConfig = None):
        """
        Args:
            config: Configuration settings for whistle detection.
        """
        if config is None:
            config = WhistleDetectorConfig()

        self.sample_rate = config.sample_rate
        self.min_freq = config.min_freq
        self.max_freq = config.max_freq
        self.energy_threshold = config.energy_threshold
        self.peak_ratio_threshold = config.peak_ratio_threshold
        self.cooldown_samples = int(config.cooldown_seconds * config.sample_rate)

        self._samples_since_last = self.cooldown_samples  # allow immediate first detect
        self._history = deque(maxlen=5)  # rolling confidence scores

    def detect(self, audio_chunk: np.ndarray) -> bool:
        """
        Analyse an audio chunk and return True if a whistle is detected.

        Args:
            audio_chunk: 1D numpy array of float32 audio samples (mono).
        Returns:
            True if a referee whistle is detected in this chunk.
        """
        if audio_chunk is None or len(audio_chunk) < 512:
            return False

        # Convert to mono if stereo
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)

        # Normalize
        audio_chunk = audio_chunk.astype(np.float32)
        max_val = np.abs(audio_chunk).max()
        if max_val < 1e-6:
            return False
        audio_chunk = audio_chunk / max_val

        # Check overall energy
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        if energy < self.energy_threshold:
            self._history.append(0.0)
            return False

        # FFT
        n = len(audio_chunk)
        fft_vals = np.abs(np.fft.rfft(audio_chunk))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)

        # Find whistle band energy
        band_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        if not band_mask.any():
            self._history.append(0.0)
            return False

        band_energy = np.mean(fft_vals[band_mask])
        overall_energy = np.mean(fft_vals)

        if overall_energy < 1e-6:
            self._history.append(0.0)
            return False

        # Ratio: how dominant is the whistle band?
        ratio = band_energy / overall_energy
        self._history.append(ratio)

        # Cooldown check
        self._samples_since_last += len(audio_chunk)
        if self._samples_since_last < self.cooldown_samples:
            return False

        # Trigger if ratio exceeds threshold (whistle band is dominant)
        if ratio >= self.peak_ratio_threshold:
            self._samples_since_last = 0
            return True

        # Also trigger if last 3 frames consistently high (sustained whistle)
        if len(self._history) >= 3:
            recent = list(self._history)[-3:]
            if all(r >= self.peak_ratio_threshold * 0.7 for r in recent):
                self._samples_since_last = 0
                return True

        return False

    def get_confidence(self) -> float:
        """Return the latest whistle-band confidence ratio (0.0 to ~10.0)."""
        if self._history:
            return self._history[-1]
        return 0.0
