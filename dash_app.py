import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
from collections import defaultdict

import pickle
import torch

def run_dash_audio_app(embeddings_dict, speaker_audio_dict):
    # Prepare data
    data = []
    for speaker, emb in embeddings_dict.items():
        emb_np = emb.squeeze().numpy()
        audio_file = os.path.basename(speaker_audio_dict[speaker]).replace('.flac','.wav') # e.g., "p226.wav"
        audio_url = f"/assets/audio/{audio_file}"
        print(audio_url)
        data.append({"speaker": speaker, "embedding": emb_np, "audio": audio_url})

    df = pd.DataFrame(data)
    emb_matrix = np.stack(df["embedding"].values)
    coords = TSNE(n_components=2, random_state=42).fit_transform(emb_matrix)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    # Create scatter plot
    fig = px.scatter(df, x="x", y="y", color="speaker", hover_name="speaker")
    fig.update_traces(marker=dict(size=10))

    # Create scatter plot
    fig = px.scatter(df, x="x", y="y", color="speaker", hover_name="speaker")
    fig.update_traces(marker=dict(size=10))

    # Save to standalone HTML
    fig.write_html("speaker_plot.html")
    print("✅ Plot saved as speaker_plot.html")


    # Build Dash app
    app = dash.Dash(__name__)
    app.title = "Speaker Audio Explorer"
    app.layout = html.Div([
        html.H2("Speaker Embeddings: Click to Play Audio"),
        dcc.Graph(id="scatter-plot", figure=fig),
        html.Div(id="audio-player", style={"marginTop": "20px"})
    ])

    # Callback to play audio on click
    @app.callback(
        Output("audio-player", "children"),
        Input("scatter-plot", "clickData")
    )
    def update_audio(clickData):
        if clickData is None:
            return "Click a point to hear the speaker."
        speaker = clickData["points"][0]["hovertext"]
        audio_file = df[df["speaker"] == speaker]["audio"].values[0]
        return html.Div([
            html.H4(f"Speaker: {speaker}"),
            html.Audio(src=audio_file, controls=True, autoPlay=True, style={"width": "300px"})
        ])

    # Run app locally
    app.run(debug=False, port=8050, host="127.0.0.1")

def average_embeddings(embeddings_dict):
    return {spk: emb.mean(dim=0) for spk, emb in embeddings_dict.items()}

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

if __name__ == '__main__':

    embeddings_raw = torch.load("/Users/sughosh/PycharmProjects/Anti-Stotter/speakerdistance/speaker_embeddings.pt")

    embeddings_avg = average_embeddings(embeddings_raw)
    with open("speaker_audio_dict.pkl", "rb") as f:
        speaker_audio_dict = pickle.load(f)


    run_dash_audio_app(embeddings_avg, speaker_audio_dict)
