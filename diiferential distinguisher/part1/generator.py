import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Number of samples for EACH differential class.
#
# Example:
# 2000 -> 2000 class 0 + 2000 class 1 = 4000 rows
#
# Increase this whenever you want a larger dataset.
NUM_SAMPLES = 10000

# Reduced rounds to generate
ROUNDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Differential mask used in the published ASCON
# machine-learning differential distinguisher experiment.
#
# We start with:
#       mask = 0x1000
#
MASK = 0x1000

# Reproducibility
SEED = 42

# Dataset output folder
OUTPUT_DIR = os.path.join(
    "datasets",
    "differential"
)


# ============================================================
# ASCON 64-BIT CONSTANTS
# ============================================================

MASK64 = 0xFFFFFFFFFFFFFFFF

# ASCON permutation round constants.
#
# For the 12-round permutation:
#
# B4 A5 96 87 78 69 5A 4B 3C 2D 1E 0F
#
ROUND_CONSTANTS = [
    0xB4,
    0xA5,
    0x96,
    0x87,
    0x78,
    0x69,
    0x5A,
    0x4B,
    0x3C,
    0x2D,
    0x1E,
    0x0F
]


# ============================================================
# 64-BIT ROTATE RIGHT
# ============================================================

def ror(x, n):

    return (
        (x >> n) |
        ((x << (64 - n)) & MASK64)
    ) & MASK64


# ============================================================
# ASCON PERMUTATION ROUND
# ============================================================

def ascon_round(state, constant):

    x0, x1, x2, x3, x4 = state

    # --------------------------------------------------------
    # CONSTANT ADDITION
    # --------------------------------------------------------

    x2 ^= constant

    # --------------------------------------------------------
    # SUBSTITUTION LAYER
    # --------------------------------------------------------

    x0 ^= x4
    x4 ^= x3
    x2 ^= x1

    t0 = x0 ^ ((~x1) & x2)
    t1 = x1 ^ ((~x2) & x3)
    t2 = x2 ^ ((~x3) & x4)
    t3 = x3 ^ ((~x4) & x0)
    t4 = x4 ^ ((~x0) & x1)

    # Restrict to 64 bits
    t0 &= MASK64
    t1 &= MASK64
    t2 &= MASK64
    t3 &= MASK64
    t4 &= MASK64

    t1 ^= t0
    t0 ^= t4
    t3 ^= t2
    t2 = (~t2) & MASK64

    # --------------------------------------------------------
    # LINEAR DIFFUSION
    # --------------------------------------------------------

    x0 = (
        t0 ^
        ror(t0, 19) ^
        ror(t0, 28)
    ) & MASK64

    x1 = (
        t1 ^
        ror(t1, 61) ^
        ror(t1, 39)
    ) & MASK64

    x2 = (
        t2 ^
        ror(t2, 1) ^
        ror(t2, 6)
    ) & MASK64

    x3 = (
        t3 ^
        ror(t3, 10) ^
        ror(t3, 17)
    ) & MASK64

    x4 = (
        t4 ^
        ror(t4, 7) ^
        ror(t4, 41)
    ) & MASK64

    return (
        x0,
        x1,
        x2,
        x3,
        x4
    )


# ============================================================
# REDUCED-ROUND ASCON-P
# ============================================================

def ascon_p(state, rounds):

    if rounds < 1 or rounds > 12:
        raise ValueError(
            "rounds must be between 1 and 12"
        )

    # Reduced-round ASCON-P uses the final r
    # round constants.
    #
    # Example:
    #
    # p[1] -> 4B
    # p[2] -> 5A 4B
    # p[3] -> 69 5A 4B
    #
    constants = ROUND_CONSTANTS[-rounds:]

    for constant in constants:

        state = ascon_round(
            state,
            constant
        )

    return state


# ============================================================
# RANDOM 320-BIT ASCON STATE
# ============================================================

def random_state(rng):

    return tuple(
        int(x)
        for x in rng.integers(
            0,
            2**64,
            size=5,
            dtype=np.uint64
        )
    )


# ============================================================
# CREATE DIFFERENTIAL STATE
# ============================================================

def apply_difference(state, delta):

    return tuple(
        (state[i] ^ delta[i]) & MASK64
        for i in range(5)
    )


# ============================================================
# CONVERT 320-BIT STATE TO 320 FEATURES
# ============================================================

def state_to_bits(state):

    bits = []

    for word in state:

        for position in range(63, -1, -1):

            bits.append(
                (word >> position) & 1
            )

    return bits


# ============================================================
# GENERATE ONE DIFFERENTIAL SAMPLE
# ============================================================

def generate_differential_sample(
    rng,
    rounds,
    delta
):

    # --------------------------------------------------------
    # Generate random base state
    # --------------------------------------------------------

    state = random_state(rng)

    # --------------------------------------------------------
    # Apply input difference
    # --------------------------------------------------------

    state_delta = apply_difference(
        state,
        delta
    )

    # --------------------------------------------------------
    # Permute both states
    # --------------------------------------------------------

    output_1 = ascon_p(
        state,
        rounds
    )

    output_2 = ascon_p(
        state_delta,
        rounds
    )

    # --------------------------------------------------------
    # Calculate output difference
    # --------------------------------------------------------

    output_difference = tuple(
        output_1[i] ^ output_2[i]
        for i in range(5)
    )

    # --------------------------------------------------------
    # Convert to 320 bits
    # --------------------------------------------------------

    return state_to_bits(
        output_difference
    )


# ============================================================
# GENERATE RANDOM CONTROL DIFFERENCE
# ============================================================

def generate_random_control_sample(rng):

    return rng.integers(
        0,
        2,
        size=320,
        dtype=np.uint8
    ).tolist()


# ============================================================
# GENERATE DATASET FOR ONE ROUND
# ============================================================

def generate_dataset(
    rounds,
    num_samples,
    mask,
    rng
):

    print()
    print("=" * 70)
    print(f"GENERATING {rounds}-ROUND DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Differential used for ASCON class
    #
    # delta = (mask, 0, 0, 0, 0)
    # --------------------------------------------------------

    delta = (
        mask,
        0,
        0,
        0,
        0
    )

    data = []
    labels = []

    # ========================================================
    # CLASS 1
    #
    # ASCON DIFFERENTIAL OUTPUT
    # ========================================================

    print("Generating ASCON differential samples...")

    for i in range(num_samples):

        sample = generate_differential_sample(
            rng,
            rounds,
            delta
        )

        data.append(sample)

        labels.append(1)

        if (i + 1) % 500 == 0:

            print(
                f"ASCON: {i + 1}/{num_samples}"
            )

    # ========================================================
    # CLASS 0
    #
    # RANDOM CONTROL
    # ========================================================

    print("Generating random control samples...")

    for i in range(num_samples):

        sample = generate_random_control_sample(
            rng
        )

        data.append(sample)

        labels.append(0)

        if (i + 1) % 500 == 0:

            print(
                f"Random: {i + 1}/{num_samples}"
            )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    columns = [
        f"bit_{i}"
        for i in range(320)
    ]

    df = pd.DataFrame(
        data,
        columns=columns
    )

    df["label"] = labels

    # ========================================================
    # SHUFFLE
    # ========================================================

    df = df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    filename = (
        f"ascon_diff_"
        f"{rounds}_round_"
        f"mask_{mask:X}.csv"
    )

    filepath = os.path.join(
        OUTPUT_DIR,
        filename
    )

    df.to_csv(
        filepath,
        index=False
    )

    # ========================================================
    # INFORMATION
    # ========================================================

    print()
    print("Saved:")
    print(filepath)

    print()
    print("Dataset shape:")
    print(df.shape)

    print()
    print("Class distribution:")
    print(df["label"].value_counts())

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ASCON-P DIFFERENTIAL DATASET GENERATOR")
    print("Target: NIST SP 800-232 / Ascon-AEAD128")
    print("=" * 70)

    print(
        f"Samples per class : {NUM_SAMPLES}"
    )

    print(
        f"Total rows        : {NUM_SAMPLES * 2}"
    )

    print(
        f"Rounds            : {ROUNDS}"
    )

    print(
        f"Differential mask : {hex(MASK)}"
    )

    print("=" * 70)

    rng = np.random.default_rng(
        SEED
    )

    # --------------------------------------------------------
    # Generate every requested round
    # --------------------------------------------------------

    for rounds in ROUNDS:

        generate_dataset(
            rounds=rounds,
            num_samples=NUM_SAMPLES,
            mask=MASK,
            rng=rng
        )

    print()
    print("=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()