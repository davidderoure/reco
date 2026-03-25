"""All tuneable constants for the cello sampler pipeline.

Every threshold here has a musically meaningful default.  Adjust these
constants without touching algorithm code to tune the pipeline for a
specific recording session or instrument.
"""

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

#: Frames per streaming chunk (10 s at 48 kHz).  Fits in ~8 MB RAM for
#: a 4-channel float32 signal.
CHUNK_SIZE_FRAMES: int = 480_000

#: Seconds of overlap kept between consecutive chunks to avoid slicing notes
#: at chunk boundaries.
OVERLAP_SECONDS: float = 2.0

#: Maximum allowed note duration in seconds.  A segment longer than this is
#: truncated (likely a bow hold or sustained passage, not a discrete note).
MAX_NOTE_DURATION_SECONDS: float = 8.0

#: Samples prepended before each detected onset to capture the full transient
#: (5 ms at 48 kHz).
PRE_ONSET_SAMPLES: int = 240

#: Samples of gap removed from the end of a note window before the next onset
#: (10 ms at 48 kHz).  Prevents bleed from the following attack into a legato tail.
POST_ONSET_GAP_SAMPLES: int = 480

#: Number of worker processes for parallel per-note analysis.
N_WORKERS: int = 4

# ---------------------------------------------------------------------------
# Onset detection
# ---------------------------------------------------------------------------

#: STFT window length in samples at 48 kHz (≈42 ms).  Long enough to give
#: good frequency resolution on the lowest cello string (~65 Hz).
STFT_NPERSEG: int = 2048

#: STFT hop size in samples (≈5.3 ms at 48 kHz).  Short enough for accurate
#: onset timing.
STFT_HOP: int = 256

#: Lower bound of the analysis frequency band (Hz).  Suppresses sub-bass
#: rumble below the lowest cello pitch.
ANALYSIS_FREQ_MIN_HZ: float = 60.0

#: Upper bound of the analysis frequency band (Hz).  Suppresses bow rosin
#: noise and room reflections above this frequency.
ANALYSIS_FREQ_MAX_HZ: float = 4000.0

#: Onset strength must exceed this multiple of the local mean to be accepted.
#: Increase for noisier recordings; decrease for very quiet passages.
ONSET_THRESHOLD_MULTIPLIER: float = 1.5

#: Half-width in hops for the local mean window used in the adaptive
#: threshold.  10 hops ≈ 265 ms — wide enough to track long crescendi.
ONSET_LOCAL_MEAN_HALF_WIDTH: int = 10

#: Minimum gap between successive onsets in milliseconds.  Shorter apparent
#: onsets are treated as a single note event.
MIN_NOTE_DURATION_MS: float = 50.0

# ---------------------------------------------------------------------------
# Polyphony detection
# ---------------------------------------------------------------------------

#: Harmonic product spectrum order.  4 means we multiply X(f)·X(2f)·X(3f)·X(4f).
HPS_ORDER: int = 4

#: Maximum duration in milliseconds of the analysis frame used for polyphony
#: detection.  Longer windows give finer frequency resolution (fewer Hanning
#: leakage false-positives between adjacent pitches), but cost more CPU.
#: 300 ms → 3.3 Hz/bin at 48 kHz, resolving pitches > 2 semitones apart.
POLYPHONY_ANALYSIS_DURATION_MS: float = 300.0

#: A secondary pitch peak in the multi-pitch salience map must exceed this
#: fraction of the maximum salience to trigger a polyphony rejection.
POLYPHONY_SALIENCE_THRESHOLD: float = 0.30

#: Two HPS peaks within this fractional frequency ratio of each other are
#: treated as the same pitch (vibrato range), not as separate voices.
POLYPHONY_UNISON_RATIO_TOLERANCE: float = 0.03

#: Number of harmonics used when building the multi-pitch salience map.
POLYPHONY_N_HARMONICS: int = 8

# ---------------------------------------------------------------------------
# Pitch estimation and intonation
# ---------------------------------------------------------------------------

#: CREPE frame step size in milliseconds.
CREPE_STEP_SIZE_MS: int = 10

#: Minimum median CREPE confidence required to accept a note.  Below this the
#: recording is too noisy, or the note is a multiphonic / harmonic.
PITCH_CONFIDENCE_THRESHOLD: float = 0.85

#: Fraction of frames trimmed from each end of the note when computing the
#: stable-region pitch statistics (attack and release excluded).
PITCH_STABLE_REGION_TRIM: float = 0.20

#: Maximum allowed absolute deviation from the nearest equal-temperament
#: semitone in cents.  30 ¢ ≈ one third of a semitone — clearly audible.
MAX_INTONATION_DEVIATION_CENTS: float = 30.0

#: A440 reference frequency used for MIDI ↔ Hz conversions.
A4_HZ: float = 440.0

#: MIDI note number of A4.
A4_MIDI: int = 69

# ---------------------------------------------------------------------------
# Articulation classification
# ---------------------------------------------------------------------------

# — Pizzicato —
#: Maximum attack duration (onset → peak) for a plucked note, in ms.
PIZZICATO_MAX_ATTACK_MS: float = 15.0

#: Minimum post-peak amplitude decay rate for pizzicato, in dB/ms.
#: Plucked strings decay faster than bowed ones.
PIZZICATO_MIN_DECAY_RATE_DB_PER_MS: float = 0.50

#: Maximum total sounding duration for a pizzicato note in ms.
PIZZICATO_MAX_DURATION_MS: float = 400.0

# — Staccato —
#: Maximum total sounding duration for a staccato note in ms.
STACCATO_MAX_DURATION_MS: float = 250.0

#: Minimum decay rate for staccato (slower than pizzicato).
STACCATO_MIN_DECAY_RATE_DB_PER_MS: float = 0.20

# — Legato —
#: Minimum total sounding duration for a legato note in ms.  Shorter takes
#: are not useful for sustaining playback and are rejected post-classification.
MIN_LEGATO_DURATION_MS: float = 500.0

# — Vibrato —
#: Minimum total sounding duration for a vibrato note in ms.  At the slowest
#: musically valid rate (4 Hz) a minimum of 750 ms guarantees at least 3 full
#: oscillation cycles, enough to clearly establish the vibrato character.
MIN_VIBRATO_DURATION_MS: float = 750.0

#: Pitch modulation must exceed this depth (in cents peak-to-peak) to be
#: classified as vibrato rather than natural intonation variation.
VIBRATO_MIN_DEPTH_CENTS: float = 20.0

#: Lowest musically valid vibrato oscillation rate in Hz.
VIBRATO_MIN_RATE_HZ: float = 4.0

#: Highest musically valid vibrato oscillation rate in Hz.
VIBRATO_MAX_RATE_HZ: float = 8.0

# — Amplitude envelope —
#: Amplitude envelope is computed via the Hilbert analytic signal.  dB values
#: below this level relative to peak are treated as silence (noise floor).
NOISE_FLOOR_DB: float = -40.0

#: Amplitude level relative to peak used to measure attack duration (-6 dB
#: point on the attack slope).
ATTACK_LEVEL_DB: float = -6.0

#: Duration in ms of the post-peak window used to measure the initial decay
#: rate.
DECAY_MEASUREMENT_WINDOW_MS: float = 100.0

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

#: Sub-directory names for each articulation class.
ARTICULATION_DIR_NAMES: dict[str, str] = {
    "staccato": "staccato",
    "legato": "legato",
    "vibrato": "vibrato",
    "pizzicato": "pizzicato",
}

#: Sub-directory for rejected samples.
REJECTED_DIR: str = "rejected"

#: Name of the CSV index file written alongside each output directory.
INDEX_FILENAME: str = "_index.csv"
