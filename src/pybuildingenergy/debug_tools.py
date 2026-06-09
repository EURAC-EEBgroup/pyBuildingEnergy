"""Small local helpers for inspecting pandas objects while debugging."""

from pathlib import Path
import tempfile


DEBUG_DIR = Path(tempfile.gettempdir()) / "df_debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def view_df(df, name="df"):
    """Save a dataframe-like object to a temporary CSV file and return the path."""

    path = DEBUG_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def plotly_df(df, x=None, y=None, name="plot"):
    import plotly.express as px
    path = DEBUG_DIR / f"{name}.html"
    fig = px.line(df, x=x, y=y)
    fig.write_html(path)
    print(f"Saved: {path}")
    return path