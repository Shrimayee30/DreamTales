from pathlib import Path

# -------------------------
# Project root
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -------------------------
# Data directories
# -------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# -------------------------
# Output directories
# -------------------------
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
SAMPLES_DIR = OUTPUTS_DIR / "samples"
LOGS_DIR = OUTPUTS_DIR / "logs"

# -------------------------
# Dataset paths (IMPORTANT)
# -------------------------
# Use the 50-image subset you downloaded
SCENE_IMAGES_DIR = RAW_DATA_DIR / "scene_subset_50"
SCENE_LABELS_PATH = PROCESSED_DATA_DIR / "labels" / "scene_subset_50_labels.csv"

LABELS_DIR = PROCESSED_DATA_DIR / "labels"

# -------------------------
# Training config
# -------------------------
IMAGE_SIZE = 64
BATCH_SIZE = 16
NUM_EPOCHS = 30   # increased for small dataset
LEARNING_RATE = 0.0002
BETA1 = 0.5

LATENT_DIM = 100
NGF = 64
NDF = 64
NUM_CHANNELS = 3

# -------------------------
# Logging / saving
# -------------------------
SAMPLE_INTERVAL = 1
TRAINING_HISTORY_PATH = LOGS_DIR / "training_history.csv"

# -------------------------
# Label classes
# -------------------------
CHARACTER_CLASSES = [
    "none",
    "mother_child",
    "friends",
    "animal_pair",
]

ACTION_CLASSES = [
    "none",
    "walking",
    "holding_hands",
    "playing",
    "sleeping",
    "shopping",
]

LOCATION_CLASSES = [
    "none",
    "store",
    "park",
    "forest",
    "bedroom",
    "street",
    "home",
]

MOOD_CLASSES = [
    "none",
    "warm",
    "calm",
    "night",
    "sunny",
    "rainy",
]

# -------------------------
# Condition dimension
# -------------------------
CONDITION_DIM = (
    len(CHARACTER_CLASSES)
    + len(ACTION_CLASSES)
    + len(LOCATION_CLASSES)
    + len(MOOD_CLASSES)
)

# -------------------------
# Reproducibility
# -------------------------
SEED = 42