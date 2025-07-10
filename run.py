import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
from collections import defaultdict
import sys
import webbrowser
import threading

def get_matrix(embeddings_dict, speaker_audio_dict):
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
    df.to_pickle("speaker_df.pkl")

def run_app(df, port=8050):
    # Create scatter plot
    fig = px.scatter(df, x="x", y="y", color="speaker", hover_name="speaker")
    fig.update_traces(marker=dict(size=10))

    # Create scatter plot
    fig = px.scatter(df, x="x", y="y", color="speaker", hover_name="speaker")
    fig.update_traces(marker=dict(size=10))

    # Make the plot longer (taller or wider)
    fig.update_layout(width=1400, height=1000)  # adjust values as needed


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
    app.run(debug=False, port=port, host="127.0.0.1")

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

def get_resource_path(filename):
    """ Get path to resource whether bundled by PyInstaller or not. """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller temp folder
        return os.path.join(sys._MEIPASS, filename)
    else:
        # Dev mode: load from script dir
        return os.path.join(os.path.dirname(__file__), filename)


# Auto-open browser after Dash starts
def open_browser(port=8050):
    webbrowser.open("http://localhost:"+str(port))

threading.Timer(1.5, open_browser).start()

if __name__ == '__main__':
    # Use this to load your file
    df_path = get_resource_path("speaker_df.pkl")
    port = 8050

    print(df_path)
    df = pd.read_pickle(df_path)

    threading.Timer(1.5, open_browser).start()

    run_app(df, port=port)
