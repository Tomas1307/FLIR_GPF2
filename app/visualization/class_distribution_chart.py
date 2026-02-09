from typing import Dict, Optional

import matplotlib.pyplot as plt

from app.config import settings


class ClassDistributionChart:
    """Renders bar charts showing the distribution of object classes.

    Attributes:
        class_names: Mapping of class id to display name.
    """

    def __init__(
        self, class_names: Optional[Dict[int, str]] = None
    ) -> None:
        """Initialise the ClassDistributionChart.

        Args:
            class_names: Mapping of class id to human-readable name.
        """
        self.class_names = class_names or settings.CLASS_NAMES

    def show(
        self,
        distribution: Dict[int, int],
        title: str = "Class Distribution",
    ) -> None:
        """Display a bar chart for a single distribution.

        Args:
            distribution: Mapping of class id to count.
            title: Chart title.
        """
        sorted_ids = sorted(distribution.keys())
        names = [self.class_names.get(cid, f"Class {cid}") for cid in sorted_ids]
        counts = [distribution[cid] for cid in sorted_ids]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(names, counts)
        ax.set_ylabel("Count")
        ax.set_title(title)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    def show_per_split(
        self,
        per_split: Dict[str, Dict[int, int]],
        title: str = "Class Distribution per Split",
    ) -> None:
        """Display grouped bar charts for multiple splits.

        Args:
            per_split: ``{split_name: {class_id: count}}``.
            title: Chart title.
        """
        all_ids = sorted(
            set(cid for dist in per_split.values() for cid in dist)
        )
        names = [self.class_names.get(cid, f"Class {cid}") for cid in all_ids]

        n_splits = len(per_split)
        fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5))
        if n_splits == 1:
            axes = [axes]

        for ax, (split, dist) in zip(axes, per_split.items()):
            counts = [dist.get(cid, 0) for cid in all_ids]
            ax.bar(names, counts)
            ax.set_title(split.upper())
            ax.set_ylabel("Count")
            plt.sca(ax)
            plt.xticks(rotation=30, ha="right")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
