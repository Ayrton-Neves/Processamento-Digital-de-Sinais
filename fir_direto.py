import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

# Leia o arquivo de áudio
taxa_amostragem, sinal_amostrado = wavfile.read('audios/whatsapp_audio.wav')

# a) Use o método de regressão linear para obter modelos de resposta a impulso finito (FIR)
n = 1000  # Número de pontos no modelo FIR
a = [1] + [0]*(n-1)
b = signal.firwin(n, cutoff=0.3, window="hamming")

# b) Plote o sinal de amostra escolhido
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(sinal_amostrado)
plt.title('Sinal de Amostra Escolhido')

# c) Encontre o modelo FIR
modelo_fir = signal.lfilter(b, a, sinal_amostrado)

# d) Aplique o modelo FIR e plote a resposta
resposta = signal.convolve(sinal_amostrado, modelo_fir, mode='same')

plt.subplot(2, 1, 2)
plt.plot(resposta)
plt.title('Resposta do Modelo FIR')

plt.tight_layout()
plt.show()