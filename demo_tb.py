import numpy as np
import scipy.io.wavfile as wav

def generate_tb_cough_demo(filename="demo_tb_cough.wav", duration=3.0, fs=22050):
    t = np.linspace(0, duration, int(fs * duration))
    
    # 1. Base 'Heavy' Rumble (Low frequencies typical in TB)
    rumble = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 2) 
    
    # 2. Spectral Noise (The 'Crackle' of inflammation)
    # We filter white noise to keep it in the 2kHz-4kHz range
    noise = np.random.normal(0, 1, len(t))
    
    # 3. Combine with 'Cough Bursts' (3 distinct coughs)
    envelope = np.zeros_like(t)
    for start in [0.2, 1.2, 2.2]: # Timing of the 3 coughs
        idx = (t >= start) & (t <= start + 0.4)
        envelope[idx] = np.exp(-(t[idx] - start) * 10)
    
    # Final signal: Heavy rumble + high frequency noise * burst pattern
    signal = (0.6 * rumble + 0.4 * noise) * envelope
    
    # Normalize to 16-bit PCM for WAV compatibility
    signal = (signal / np.max(np.abs(signal)) * 32767).astype(np.int16)
    
    wav.write(filename, fs, signal)
    print(f"✅ Created: {filename}")

generate_tb_cough_demo()