import numpy as np


def project(anchor_xyz: np.ndarray, trace: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points = np.concatenate([anchor_xyz[:, None], anchor_xyz[:, None] + trace], 1)
    z = np.clip(points[..., 2], 1e-3, None)
    u = intrinsics[0, 0] * points[..., 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * points[..., 1] / z + intrinsics[1, 2]
    return np.stack([u, v], -1)


def draw_panel(ax, rgb, paths: dict, mask, title: str, dynamic=None) -> None:
    ax.imshow(rgb)
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)
    ax.axis("off")
    ax.set_title(title, fontsize=8, wrap=True)
    for name, (pixels, valid, color) in paths.items():
        for i in np.nonzero(mask)[0]:
            keep = np.concatenate([[True], valid[i]])
            line = pixels[i][keep]
            width = 1.6 if dynamic is None or dynamic[i] else 0.6
            ax.plot(line[:, 0], line[:, 1], color=color, linewidth=width, alpha=0.9)
            ax.scatter(line[-1:, 0], line[-1:, 1], color=color, s=6)
    ax.scatter(*[paths[next(iter(paths))][0][mask, 0, k] for k in (0, 1)], color="white", s=5, edgecolors="black",
               linewidths=0.3)


def render_case(path, rgb, panels: list[dict], suptitle: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.0), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        draw_panel(ax, rgb, panel["paths"], panel["mask"], panel["title"], panel.get("dynamic"))
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
