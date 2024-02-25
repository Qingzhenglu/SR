import matplotlib.pyplot as plt
import torchaudio
import torch
from IPython.display import Audio, display


def plot_specgram(wf, sr, title="Spectrogram", xlim=None):
    wave_form = wf.numpy()

    num_channels, num_frames = wave_form.shape
    time_axis = torch.arange(0, num_frames) / sr

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sr)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)


def play_audio(wf, sr):
    wf = wf.numpy()

    num_channels, num_frames = wf.shape
    if num_channels == 1:
        display(Audio(wf[0], rate=sr))
    elif num_channels == 2:
        display(Audio((wf[0], wf[1]), rate=sr))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


if __name__ == "__main__":
    dataset = torchaudio.datasets.SPEECHCOMMANDS(root="", download=False)

    for i in [1]:
        waveform, sample_rate, label, speaker_id, num = dataset[i]
        plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
        play_audio(waveform, sample_rate)
