import os

import torch
import librosa
import numpy as np
import torch.nn.functional as F

def snr_db(x, desired_noise):
    return 20 * torch.log10(torch.sum(x ** 2) / torch.sum(desired_noise ** 2))

def relative_eps(snr):
    return torch.pow(torch.tensor(10.0), float(snr) / 20)

def pad_audio(audio, orig_size):
    max_size = (16000,)

    if audio.shape[1] < max_size[0]:
        torch.nn.functional.pad(audio,(0,max_size[0] - orig_size[1]))
    elif audio.shape[1] > max_size[0]:
        audio = audio[:max_size[0]]
    return audio

def attack_wav(dataloader, x,y, model, n_iter, epsilon, alpha, rand_init_uniform=True,rand_init_normal=False, clip_min_max=True):
    model.eval()
    correct = 0
    total = 0

    y = torch.LongTensor([y]).cuda()
    audio, sr = librosa.load(x, sr=None)
    min_val, max_val = -1, 1 # audio.min(), audio.max()
    audio = np.expand_dims(audio, 0)

    audio = torch.FloatTensor(audio)
    delta = torch.zeros_like(audio, requires_grad=True)

    if rand_init_normal:
        delta.data = delta.data.normal_(std=1)
    elif rand_init_uniform:
        delta.data = delta.data.uniform_(-epsilon, epsilon)
    
    padded_audio = pad_audio(audio + delta,audio.shape)

    
    for i in range(n_iter):
        spect, phase = dataloader.dataset.stft.transform(padded_audio)

        spect = spect.unsqueeze(0).cuda()
        yhat = model(spect)
        loss = F.nll_loss(yhat, y)
        loss.backward()
        
        delta.data = delta.data + alpha * torch.sign(delta.grad.data)
        delta.data = torch.clamp(delta.data, min=-epsilon, max=epsilon)
        if clip_min_max:
            delta.data = torch.clamp(audio+delta.data, min=-min_val, max=max_val)-audio
        
    total += 1
    correct += int(yhat.argmax() != y.item())

    print(f"PGD accuracy {correct*100/total} ({correct}/{total})")
    return audio+delta.detach(), sr, correct


def attack(dataloader, example_idx, model, n_iter=10, eps=0.01, alpha=0.01/2, rand_init=True):
    x, y = dataloader.dataset.spects[example_idx]
    adv_wav, sr, cor = attack_wav(dataloader, x, y, model, n_iter, eps, alpha, rand_init)
    return adv_wav


