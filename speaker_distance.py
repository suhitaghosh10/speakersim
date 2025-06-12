import shutil

from speechbrain.pretrained import SpeakerRecognition
import torchaudio
from collections import defaultdict
import os
import torch
from scipy.spatial.distance import pdist, squareform

import itertools


import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.manifold import TSNE


# Extract embeddings for each speaker
def get_embeddings(speaker_audio_dict:dict, savedir:str):
    embeddings = {}
    model = SpeakerRecognition.from_hparams(savedir=savedir, source="speechbrain/spkrec-ecapa-voxceleb")

    for speaker, files in speaker_audio_dict.items(): # e.g., {'A': 'A.wav', 'B': 'B.wav', ...}
        embeddings_speaker = []
        for file in files:
            try:
                print(file)
                signal, fs = torchaudio.load(file)
            except UnicodeDecodeError:
                print(f"Unicode decode error in file: {file}")
                continue
            except Exception as e:
                print(f"Other error in file {file}: {e}")
                continue
            emb = model.encode_batch(signal)
            embeddings_speaker.append(emb.squeeze().detach().numpy())
        embeddings[speaker] = torch.tensor(embeddings_speaker)
        print(speaker)
    return embeddings

def get_distance_matrix(embeddings:dict):
    speaker_names = list(embeddings.keys())
    embedding_matrix = np.stack([embeddings[s] for s in speaker_names])
    distance_matrix = squareform(pdist(embedding_matrix, metric='cosine'))  # or 'euclidean'
    return distance_matrix

# def get_heatmap(distance_matrix, speaker_names:list):
#
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(distance_matrix, xticklabels=speaker_names, yticklabels=speaker_names, annot=True, cmap="viridis")
#     plt.title("Speaker Distance Matrix (Cosine)")
#     plt.show()

def pca(embedding_matrix, speaker_names):
    # Use PCA first to reduce noise
    pca = PCA(n_components=10).fit_transform(embedding_matrix)
    tsne = TSNE(n_components=2, perplexity=5).fit_transform(pca)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(speaker_names):
        plt.scatter(tsne[i, 0], tsne[i, 1])
        plt.text(tsne[i, 0] + 0.01, tsne[i, 1] + 0.01, label)
    plt.title("Speaker Embeddings Visualized with t-SNE")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.show()

def create_speaker_dict(directory):
    speaker_dict = defaultdict(list)

    for speaker in os.listdir(directory):
        dir_path = os.path.join(directory, speaker)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(".flac"):
                    speaker_dict[speaker].append(os.path.join(dir_path, filename))
        print(speaker)

    return dict(speaker_dict)

def create_speaker_audio_dict(directory):
    speaker_dict = defaultdict(list)

    for speaker in os.listdir(directory):
        dir_path = os.path.join(directory, speaker)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(".flac"):
                    speaker_dict[speaker] = os.path.join(dir_path, filename)
                    break
        print(speaker)

    return dict(speaker_dict)

def compute_average_inter_speaker_distances(embeddings):
    distance_accumulator = defaultdict(list)
    speakers = list(embeddings.keys())

    for spk1, spk2 in itertools.combinations(speakers, 2):
        emb1 = embeddings[spk1]  # (N1, D)
        emb2 = embeddings[spk2]  # (N2, D)

        for i in range(emb1.size(0)):
            for j in range(emb2.size(0)):
                dist = torch.nn.functional.pairwise_distance(
                    emb1[i].unsqueeze(0), emb2[j].unsqueeze(0)
                ).item()
                distance_accumulator[(spk1, spk2)].append(dist)
        print(spk1, spk2)

    # Compute average
    avg_distances = {
        pair: sum(dists) / len(dists)
        for pair, dists in distance_accumulator.items()
    }

    return avg_distances

def create_speaker_audio_dict(directory):
    speaker_dict = defaultdict(list)

    for speaker in os.listdir(directory):
        dir_path = os.path.join(directory, speaker)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(".flac"):
                    speaker_dict[speaker] = os.path.join(dir_path, filename)
                    break

    return dict(speaker_dict)


def interactive_speaker_audio_plot(embeddings_dict, speaker_audio_dict):
    data = []

    for speaker, emb in embeddings_dict.items():
        emb_np = emb.squeeze().numpy()  # assume shape (embedding_dim,)
        audio_file = speaker_audio_dict[speaker] # one audio per speaker
        shutil.copy(audio_file, '/project/sghosh/code/anti-stotter/audio/')
        audio_file = audio_file.split('/')[-1]
        print(audio_file)

        audio_tag = f'<b>{speaker}</b><br><audio controls src="{audio_file}" style="width:200px;"></audio>'
        data.append({"speaker": speaker, "embedding": emb_np, "audio": audio_file, "audio_html": audio_tag})

    df = pd.DataFrame(data)
    embedding_matrix = np.stack(df["embedding"].values)

    # Reduce to 2D
    coords = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(embedding_matrix)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    # Plot
    fig = px.scatter(
        df, x="x", y="y", color="speaker", hover_data={"audio_html": True, "speaker": True, "x": False, "y": False}
    )
    fig.write_html("speaker_audio_plot.html", auto_open=False)
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(title="Speaker Embeddings with Audio Playback", hovermode='closest')
    fig.show()

if __name__ == '__main__':
    os.environ['HTTP_PROXY'] = 'http://fp.cs.ovgu.de:3210/'
    os.environ['HTTPS_PROXY'] = 'http://fp.cs.ovgu.de:3210/'
    audio_dir: str = '/data/share/speech/vctk/wav48_silence_trimmed/'
    # speaker_audio_dict = create_speaker_dict(audio_dir)
    # print(speaker_audio_dict)
    #
    # emb = get_embeddings(speaker_audio_dict, '/project/sghosh/')
    # print(len(emb))
    # torch.save(emb, "speaker_embeddings.pt")
    # avg_emb = compute_average_inter_speaker_distances(torch.load("speaker_embeddings.pt"))
    # torch.save(avg_emb, "avg_speaker_embeddings_dist.pt")
    #create_speaker_audio_dict(audio_dir)
    speaker_audio_dict = create_speaker_audio_dict('/data/share/speech/vctk/wav48_silence_trimmed/')

    embeddings_raw = torch.load("/project/sghosh/code/anti-stotter/speakerdistance/speaker_embeddings.pt")


    def average_embeddings(embeddings_dict):
        return {spk: emb.mean(dim=0) for spk, emb in embeddings_dict.items()}


    embeddings_avg = average_embeddings(embeddings_raw)

    interactive_speaker_audio_plot(embeddings_avg, speaker_audio_dict)
