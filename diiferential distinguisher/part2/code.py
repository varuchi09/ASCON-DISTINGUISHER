# ============================================================
# ASCON DIFFERENTIAL DATASET GENERATOR
# Leakage-safe version
#
# Output:
#   20,000 samples
#   320 binary features: bit_0 ... bit_319
#   1 binary label: 0 / 1
#
# Change ROUND = 4 for:
#   ASCON_differential_round_4.csv
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROUND = 6                 # <-- CHANGE THIS: 1, 2, 3, 4, or 5
TOTAL_SAMPLES = 20000
SEED = 2025

OUTPUT_FILE = f"ASCON_differential_round_{ROUND}.csv"

# Equal number of samples in both classes
SAMPLES_PER_CLASS = TOTAL_SAMPLES // 2


# ============================================================
# ASCON CONSTANTS
# ============================================================

MASK64 = (1 << 64) - 1

ROUND_CONSTANTS = [
    0xF0,
    0xE1,
    0xD2,
    0xC3,
    0xB4,
    0xA5,
    0x96,
    0x87,
    0x78,
    0x69,
    0x5A,
    0x4B
]


# ============================================================
# 64-BIT ROTATION
# ============================================================

def rotr(x, n):
    """
    Rotate a 64-bit integer right by n bits.
    """
    x &= MASK64

    if n == 0:
        return x

    return ((x >> n) | (x << (64 - n))) & MASK64


# ============================================================
# ASCON S-BOX
# ============================================================

def ascon_sbox(x0, x1, x2, x3, x4):

    x0 ^= x4
    x4 ^= x3
    x2 ^= x1

    t0 = (~x0) & x1
    t1 = (~x1) & x2
    t2 = (~x2) & x3
    t3 = (~x3) & x4
    t4 = (~x4) & x0

    x0 ^= t1
    x1 ^= t2
    x2 ^= t3
    x3 ^= t4
    x4 ^= t0

    x1 ^= x0
    x0 ^= x4
    x3 ^= x2
    x2 = (~x2) & MASK64

    return (
        x0 & MASK64,
        x1 & MASK64,
        x2 & MASK64,
        x3 & MASK64,
        x4 & MASK64
    )


# ============================================================
# ASCON LINEAR DIFFUSION
# ============================================================

def linear_layer(x0, x1, x2, x3, x4):

    x0 ^= rotr(x0, 19) ^ rotr(x0, 28)
    x1 ^= rotr(x1, 61) ^ rotr(x1, 39)
    x2 ^= rotr(x2, 1)  ^ rotr(x2, 6)
    x3 ^= rotr(x3, 10) ^ rotr(x3, 17)
    x4 ^= rotr(x4, 7)  ^ rotr(x4, 41)

    return (
        x0 & MASK64,
        x1 & MASK64,
        x2 & MASK64,
        x3 & MASK64,
        x4 & MASK64
    )


# ============================================================
# ONE ASCON ROUND
# ============================================================

def ascon_round(state, round_constant):

    x0, x1, x2, x3, x4 = state

    # --------------------------------------------------------
    # ADD ROUND CONSTANT
    # --------------------------------------------------------

    x2 ^= round_constant

    # --------------------------------------------------------
    # SUBSTITUTION LAYER
    # --------------------------------------------------------

    x0, x1, x2, x3, x4 = ascon_sbox(
        x0, x1, x2, x3, x4
    )

    # --------------------------------------------------------
    # LINEAR DIFFUSION
    # --------------------------------------------------------

    x0, x1, x2, x3, x4 = linear_layer(
        x0, x1, x2, x3, x4
    )

    return (
        x0 & MASK64,
        x1 & MASK64,
        x2 & MASK64,
        x3 & MASK64,
        x4 & MASK64
    )


# ============================================================
# ASCON PERMUTATION
# ============================================================

def ascon_permutation(state, rounds):

    """
    Apply exactly 'rounds' ASCON permutation rounds.

    rounds can be 1, 2, 3, 4, or 5.
    """

    if rounds < 1 or rounds > 12:
        raise ValueError("ROUND must be between 1 and 12.")

    # For reduced-round experiments we use the first
    # ROUND_CONSTANTS entries consistently.
    for r in range(rounds):
        state = ascon_round(
            state,
            ROUND_CONSTANTS[r]
        )

    return state


# ============================================================
# RANDOM ASCON STATE
# ============================================================

def random_state(rng):
    """
    Generate a random 320-bit ASCON state.
    """

    return tuple(
        int(rng.integers(0, 2**64, dtype=np.uint64))
        for _ in range(5)
    )


# ============================================================
# STATE -> 320 BITS
# ============================================================

def state_to_bits(state):
    """
    Convert five 64-bit ASCON words into exactly
    320 binary features.

    bit_0 ... bit_319
    """

    bits = []

    for word in state:

        for i in range(64):
            bits.append(
                (word >> i) & 1
            )

    return bits


# ============================================================
# DIFFERENTIAL CONSTRUCTION
# ============================================================

def generate_differential_state(rng, rounds):

    """
    Generate a reduced-round ASCON differential sample.

    A random base state is created and a fixed non-zero
    input difference is injected.

    Both states are independently permuted.

    The resulting differential state is returned.
    """

    # --------------------------------------------------------
    # Random base state
    # --------------------------------------------------------

    state1 = random_state(rng)

    # --------------------------------------------------------
    # Fixed non-zero differential
    #
    # The difference is kept constant across samples.
    # This avoids label-dependent random-generation artifacts.
    # --------------------------------------------------------

    delta = (
        0x0000000000000001,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000
    )

    state2 = tuple(
        (state1[i] ^ delta[i]) & MASK64
        for i in range(5)
    )

    # --------------------------------------------------------
    # Apply the requested number of ASCON rounds
    # --------------------------------------------------------

    out1 = ascon_permutation(
        state1,
        rounds
    )

    out2 = ascon_permutation(
        state2,
        rounds
    )

    # --------------------------------------------------------
    # Differential state
    # --------------------------------------------------------

    diff = tuple(
        out1[i] ^ out2[i]
        for i in range(5)
    )

    return state_to_bits(diff)


# ============================================================
# NEGATIVE SAMPLE GENERATION
# ============================================================

def generate_negative_state(rng, rounds):

    """
    Generate a negative/random differential sample.

    Two completely independent random states are generated.

    Importantly, this does NOT reuse a positive sample,
    preventing sample-pair leakage.
    """

    state1 = random_state(rng)
    state2 = random_state(rng)

    out1 = ascon_permutation(
        state1,
        rounds
    )

    out2 = ascon_permutation(
        state2,
        rounds
    )

    diff = tuple(
        out1[i] ^ out2[i]
        for i in range(5)
    )

    return state_to_bits(diff)


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(
    rounds,
    samples_per_class,
    seed
):

    rng = np.random.default_rng(seed)

    X = []
    y = []

    # --------------------------------------------------------
    # CLASS 1
    # Differential samples
    # --------------------------------------------------------

    print(
        f"Generating {samples_per_class} "
        f"class-1 samples..."
    )

    for _ in range(samples_per_class):

        bits = generate_differential_state(
            rng,
            rounds
        )

        X.append(bits)
        y.append(1)

    # --------------------------------------------------------
    # CLASS 0
    # Independent/random samples
    # --------------------------------------------------------

    print(
        f"Generating {samples_per_class} "
        f"class-0 samples..."
    )

    for _ in range(samples_per_class):

        bits = generate_negative_state(
            rng,
            rounds
        )

        X.append(bits)
        y.append(0)

    X = np.asarray(
        X,
        dtype=np.uint8
    )

    y = np.asarray(
        y,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Shuffle AFTER generation.
    #
    # This prevents:
    # first 10,000 rows -> class 1
    # next 10,000 rows  -> class 0
    #
    # which can create an artificial ordering leakage.
    # --------------------------------------------------------

    permutation = rng.permutation(
        len(y)
    )

    X = X[permutation]
    y = y[permutation]

    return X, y


# ============================================================
# CREATE DATASET
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ASCON DIFFERENTIAL DATASET GENERATOR")
    print("=" * 60)

    print(f"Round             : {ROUND}")
    print(f"Total samples     : {TOTAL_SAMPLES}")
    print(f"Samples/class     : {SAMPLES_PER_CLASS}")
    print(f"Random seed       : {SEED}")
    print(f"Output file       : {OUTPUT_FILE}")
    print("=" * 60)

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    X, y = generate_dataset(
        rounds=ROUND,
        samples_per_class=SAMPLES_PER_CLASS,
        seed=SEED
    )

    # --------------------------------------------------------
    # Create column names
    # --------------------------------------------------------

    feature_columns = [
        f"bit_{i}"
        for i in range(320)
    ]

    columns = feature_columns + ["label"]

    # --------------------------------------------------------
    # Construct DataFrame
    # --------------------------------------------------------

    data = np.column_stack(
        (X, y)
    )

    df = pd.DataFrame(
        data,
        columns=columns
    )

    # --------------------------------------------------------
    # Final safety checks
    # --------------------------------------------------------

    assert df.shape == (
        TOTAL_SAMPLES,
        321
    )

    assert list(df.columns) == columns

    assert set(
        df["label"].unique()
    ) == {0, 1}

    assert (
        df["label"].value_counts()[0]
        == SAMPLES_PER_CLASS
    )

    assert (
        df["label"].value_counts()[1]
        == SAMPLES_PER_CLASS
    )

    # Every feature must be binary
    assert np.all(
        df[feature_columns].values >= 0
    )

    assert np.all(
        df[feature_columns].values <= 1
    )

    # No duplicated complete samples
    duplicate_count = df.duplicated().sum()

    print()
    print("=" * 60)
    print("DATASET CHECK")
    print("=" * 60)

    print("Shape              :", df.shape)
    print(
        "Class 0 samples    :",
        (df["label"] == 0).sum()
    )
    print(
        "Class 1 samples    :",
        (df["label"] == 1).sum()
    )
    print(
        "Duplicate samples  :",
        duplicate_count
    )
    print(
        "Feature count      :",
        len(feature_columns)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Saved: {OUTPUT_FILE}")