import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def candlesticks(df, title='', upColor='blue', downColor='red'):
    """
    Create candlestick plot for the given bars. The bars have to  be given as
    a DataFrame.
    """

    if 'date' not in df.columns:
        df = df.reset_index()

    df = df[['date', 'open', 'high', 'low', 'close']].copy()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.grid(True)
    ax.margins(0)

    for row in df.itertuples():
        if row.close >= row.open:
            color = upColor
            bodyHi, bodyLo = row.close, row.open
        else:
            color = downColor
            bodyHi, bodyLo = row.open, row.close
        line = Line2D(
            xdata=(row.Index, row.Index),
            ydata=(row.low, bodyLo),
            color=color,
            linewidth=1)
        ax.add_line(line)
        line = Line2D(
            xdata=(row.Index, row.Index),
            ydata=(row.high, bodyHi),
            color=color,
            linewidth=1)
        ax.add_line(line)
        rect = Rectangle(
            xy=(row.Index - 0.3, bodyLo),
            width=0.6,
            height=bodyHi - bodyLo,
            edgecolor=color,
            facecolor=color,
            alpha=0.4,
            antialiased=True
        )
        ax.add_patch(rect)

    ax.autoscale_view()
    plt.show()
