import matplotlib.patches as patches

def visualize_texts(ax, text_list):
    color_dict = {0: 'blue', 1: 'red'}
    for text in text_list:
        x0, y0 = text["x0"], text["y0"]
        x1, y1 = text["x1"], text["y1"]

        width = x1 - x0
        height = y1 - y0
        color = color_dict[int(text['is_rotated'])]

        rect = patches.Rectangle(
            (x0, y0),
            width,
            height,
            linewidth=0.5,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)

        # 中心にテキスト表示
        ax.text(
            text["cx"], text["cy"],
            text["text"],
            fontsize=4,
            color=color
        )