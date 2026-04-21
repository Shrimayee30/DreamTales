CHARACTER_CLASSES = ["none", "mother_child", "friends", "animal_pair"]
ACTION_CLASSES = ["none", "walking", "holding_hands", "playing", "sleeping", "shopping"]
LOCATION_CLASSES = ["none", "store", "park", "forest", "bedroom", "street", "home"]
MOOD_CLASSES = ["none", "warm", "calm", "night", "sunny", "rainy"]


KEYWORD_RULES = {
    "character": {
        "mother_child": ["mother", "child", "parent", "daughter", "son", "family"],
        "friends": ["friends", "friend", "together", "companions"],
        "animal_pair": ["animals", "animal", "foxes", "birds", "rabbits", "wolves", "deer"],
    },
    "action": {
        "walking": ["walk", "walking", "wander", "stroll", "glide"],
        "holding_hands": ["holding hands", "hand in hand", "together"],
        "playing": ["play", "playing", "dance", "laugh", "chase"],
        "sleeping": ["sleep", "sleeping", "rest", "dream", "lullaby"],
        "shopping": ["shop", "shopping", "store", "market"],
    },
    "location": {
        "store": ["store", "market", "shop"],
        "park": ["park", "garden", "meadow"],
        "forest": ["forest", "woods", "trees"],
        "bedroom": ["bedroom", "bed", "pillow"],
        "street": ["street", "road", "town"],
        "home": ["home", "house", "kitchen"],
    },
    "mood": {
        "warm": ["warm", "golden", "glow", "cozy", "kind", "kindest"],
        "calm": ["calm", "peaceful", "gentle", "quiet", "still", "soft"],
        "night": ["night", "moonlit", "moon", "stars", "midnight", "silver"],
        "sunny": ["sunny", "bright", "daylight", "sun"],
        "rainy": ["rain", "rainy", "storm", "mist"],
    },
}


POSITIVE_WORDS = {
    "bright",
    "calm",
    "cozy",
    "dream",
    "gentle",
    "glow",
    "golden",
    "happy",
    "hope",
    "kind",
    "laugh",
    "light",
    "peaceful",
    "safe",
    "soft",
    "sunny",
    "warm",
    "wonder",
}

NEGATIVE_WORDS = {
    "afraid",
    "alone",
    "cold",
    "dark",
    "lost",
    "nightmare",
    "rain",
    "sad",
    "scared",
    "shadow",
    "storm",
    "tense",
    "thunder",
    "worried",
}


MOOD_TO_MUSIC_STYLE = {
    "warm": "warm acoustic lullaby, soft piano, gentle strings, cozy and hopeful",
    "calm": "calm ambient lullaby, soft pads, celesta, slow tempo, peaceful",
    "night": "moonlit ambient music, quiet harp, airy synths, soft reverb",
    "sunny": "bright whimsical music, light marimba, ukulele, playful rhythm",
    "rainy": "gentle rainy day piano, soft bells, muted strings, reflective",
    "none": "soft cinematic background music, gentle instrumental score",
}

