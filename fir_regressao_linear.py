import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.linalg import lstsq

# Ler o arquivo de áudio
fs, signal = wavfile.read('audios/whatsapp_audio.wav')
signal = signal / np.max(np.abs(signal))  # Normalização

# Número de coeficientes do filtro FIR
n = 100

# Preparar a matriz de regressão linear
X = np.zeros((len(signal) - n, n))
for i in range(n):
    X[:, i] = signal[i: len(signal) - n + i]

# Sinal de saída desejado (sinal original deslocado)
y = signal[n:]

# Resolver a equação linear X*b = y
b, residuals, rank, s = lstsq(X, y)

# Plotar o sinal de amostra escolhido
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(signal)
plt.title('Sinal de Amostra Escolhido')
plt.xlabel('Amostras')
plt.ylabel('Amplitude')

# Aplicar o filtro FIR obtido
filtered_signal = np.convolve(signal, b, mode='same')

# Plotar a resposta do filtro FIR
plt.subplot(2, 1, 2)
plt.plot(filtered_signal)
plt.title('Resposta do Modelo FIR')
plt.xlabel('Amostras')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()