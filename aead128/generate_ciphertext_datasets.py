import numpy as np
import pandas as pd
from pathlib import Path

MASK64 = (1 << 64) - 1
ROUND_CONSTANTS = [0xF0,0xE1,0xD2,0xC3,0xB4,0xA5,0x96,0x87,0x78,0x69,0x5A,0x4B]
ASCON_128A_IV = 0x00001000808C0001

def ror(x,n):
    return ((x >> n) | ((x << (64-n)) & MASK64)) & MASK64

def permute(s, rounds):
    s=list(s)
    for c in ROUND_CONSTANTS[12-rounds:]:
        s[2] ^= c
        s[0] ^= s[4]
        s[4] ^= s[3]
        s[2] ^= s[1]
        x0,x1,x2,x3,x4=s
        t0=x0 ^ ((~x1) & x2)
        t1=x1 ^ ((~x2) & x3)
        t2=x2 ^ ((~x3) & x4)
        t3=x3 ^ ((~x4) & x0)
        t4=x4 ^ ((~x0) & x1)
        t1 ^= t0
        t0 ^= t4
        t3 ^= t2
        t2 = (~t2) & MASK64
        s=[
            (t0 ^ ror(t0,19) ^ ror(t0,28)) & MASK64,
            (t1 ^ ror(t1,61) ^ ror(t1,39)) & MASK64,
            (t2 ^ ror(t2,1) ^ ror(t2,6)) & MASK64,
            (t3 ^ ror(t3,10) ^ ror(t3,17)) & MASK64,
            (t4 ^ ror(t4,7) ^ ror(t4,41)) & MASK64,
        ]
    return s

def load64(b):
    return int.from_bytes(b,'little')

def store64(x):
    return (x & MASK64).to_bytes(8,'little')

def ascon_aead128_encrypt(key, nonce, ad, plaintext, rounds):
    if len(key)!=16 or len(nonce)!=16:
        raise ValueError("Ascon-AEAD128 requires 16-byte key and nonce")
    s=[ASCON_128A_IV,load64(key[:8]),load64(key[8:]),
       load64(nonce[:8]),load64(nonce[8:])]
    permute_in_place = lambda state: None
    s=permute(s,rounds)
    s[3] ^= load64(key[:8])
    s[4] ^= load64(key[8:])

    # Associated data
    if len(ad):
        pos=0
        while pos+16 <= len(ad):
            s[0] ^= load64(ad[pos:pos+8])
            s[1] ^= load64(ad[pos+8:pos+16])
            s=permute(s,rounds)
            pos += 16
        rem=ad[pos:]
        if len(rem)>=8:
            s[0] ^= load64(rem[:8])
            s[1] ^= load64(rem[8:])
            s[1] ^= 1 << (8*(len(rem)-8))
        else:
            s[0] ^= load64(rem)
            s[0] ^= 1 << (8*len(rem))
        s=permute(s,rounds)

    # Domain separation: DSEP() = 0x80 in the most significant byte of x4.
    s[4] ^= 0x80 << 56

    # Plaintext / ciphertext
    ct=bytearray()
    pos=0
    while pos+16 <= len(plaintext):
        s[0] ^= load64(plaintext[pos:pos+8])
        s[1] ^= load64(plaintext[pos+8:pos+16])
        ct += store64(s[0]) + store64(s[1])
        s=permute(s,rounds)
        pos += 16

    rem=plaintext[pos:]
    if len(rem)>=8:
        s[0] ^= load64(rem[:8])
        s[1] ^= load64(rem[8:])
        ct += store64(s[0])[:8] + store64(s[1])[:len(rem)-8]
        s[1] ^= 1 << (8*(len(rem)-8))
    else:
        s[0] ^= load64(rem)
        ct += store64(s[0])[:len(rem)]
        s[0] ^= 1 << (8*len(rem))

    # Finalization
    s[2] ^= load64(key[:8])
    s[3] ^= load64(key[8:])
    s=permute(s,rounds)
    s[3] ^= load64(key[:8])
    s[4] ^= load64(key[8:])
    tag=store64(s[3])+store64(s[4])
    return bytes(ct),tag

def bytes_to_bits(b):
    return np.unpackbits(np.frombuffer(b,dtype=np.uint8),bitorder='big')

def make_dataset(rounds,n_per_class=2000,plaintext_len=64,seed=20260818):
    rng=np.random.default_rng(seed+rounds)
    rows=[]; labels=[]
    for _ in range(n_per_class):
        key=rng.bytes(16)
        nonce=rng.bytes(16)
        plaintext=rng.bytes(plaintext_len)
        ciphertext,tag=ascon_aead128_encrypt(
            key,nonce,b'',plaintext,rounds
        )
        rows.append(bytes_to_bits(ciphertext+tag))
        labels.append(1)
    for _ in range(n_per_class):
        rows.append(bytes_to_bits(rng.bytes(plaintext_len+16)))
        labels.append(0)
    X=np.asarray(rows,dtype=np.uint8)
    y=np.asarray(labels,dtype=np.uint8)
    idx=rng.permutation(len(y))
    X,y=X[idx],y[idx]
    df=pd.DataFrame(X,columns=[f'bit_{i}' for i in range(X.shape[1])])
    df['label']=y
    return df

if __name__ == '__main__':
    out=Path('datasets')
    out.mkdir(exist_ok=True)
    for r in range(1,13):
        make_dataset(r).to_csv(
            out/f'ascon_ciphertext_{r}_rounds.csv',
            index=False
        )
