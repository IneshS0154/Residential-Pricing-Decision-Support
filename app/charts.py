"""Plotly charts for the dashboard (transparent background, so they follow the Streamlit light/dark theme)."""
import plotly.graph_objects as go

AMBER = "#e0a030"
BLUE, NAVY, GREEN, RED, GREY = "#5b8def", "#1f3b73", "#2e9d6a", "#d4604d", "rgba(128,128,128,0.6)"


FONT = dict(family="Poppins, sans-serif")


def usd(v):
    return f"${v:,.0f}"


def kusd(v):
    s = f"${abs(v)/1000:,.1f}k"
    return ("−" if v < 0 else "") + s


def coverage_chart(items, nominal=0.80):
    """items: [(label, coverage_share, colour_key)] -> horizontal bars against the nominal level."""
    labels = [i[0] for i in items]
    vals = [i[1] for i in items]
    cols = [GREEN if abs(v - nominal) <= 0.02 else (AMBER if v < nominal else BLUE) for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=cols, text=[f"{v:.1%}" for v in vals], textposition="outside",
                           cliponaxis=False, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.add_vline(x=nominal, line=dict(color="rgba(150,150,150,1)", dash="dot", width=2), annotation_text="nominal 80%", annotation_position="top")
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=60 + 46 * len(items), margin=dict(l=10, r=50, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=FONT)
    return fig
