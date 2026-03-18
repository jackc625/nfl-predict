"""
Visualization utilities for evaluation metrics.

This module provides plotting and charting capabilities for model evaluation,
calibration analysis, and performance visualization.
"""

import numpy as np

from utils.logging_config import get_logger

from .metrics import CalibrationMetrics, ComprehensiveMetrics, EdgeBucketMetrics

logger = get_logger(__name__)

# Try to import plotting libraries with graceful fallback
try:
    import matplotlib.pyplot as plt
    from matplotlib import patches  # noqa: F401
    from matplotlib.patches import Rectangle  # noqa: F401

    MATPLOTLIB_AVAILABLE = True
    logger.info("Matplotlib available for plotting")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available, plotting will be disabled")

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
    logger.info("Seaborn available for enhanced plotting")
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("Seaborn not available, using basic matplotlib styling")


class MetricsVisualizer:
    """
    Creates visualizations for model evaluation metrics.

    Provides reliability diagrams, calibration curves, edge bucket analysis,
    and performance comparison plots.
    """

    def __init__(self, style: str = "default", figsize: tuple[int, int] = (12, 8)):
        """Initialize visualizer with style settings."""
        self.style = style
        self.figsize = figsize

        if MATPLOTLIB_AVAILABLE:
            plt.style.use(style)
            if SEABORN_AVAILABLE:
                sns.set_palette("husl")

        logger.info(f"MetricsVisualizer initialized with style: {style}")

    def plot_reliability_diagram(
        self,
        calibration_metrics: CalibrationMetrics,
        title: str = "Reliability Diagram",
        save_path: str | None = None,
    ) -> str | None:
        """
        Create reliability diagram showing calibration quality.

        Args:
            calibration_metrics: Calibration metrics to plot
            title: Plot title
            save_path: Path to save plot (optional)

        Returns:
            Path to saved plot or None if matplotlib unavailable
        """

        if not MATPLOTLIB_AVAILABLE:
            logger.warning(
                "Matplotlib not available, cannot create reliability diagram"
            )
            return self._create_text_reliability_diagram(calibration_metrics, save_path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Reliability diagram
        confidences = calibration_metrics.bin_confidences
        accuracies = calibration_metrics.bin_accuracies
        counts = calibration_metrics.bin_counts

        # Filter out empty bins
        valid_indices = [i for i, count in enumerate(counts) if count > 0]
        if not valid_indices:
            logger.warning("No valid bins for reliability diagram")
            plt.close(fig)
            return None

        valid_confidences = [confidences[i] for i in valid_indices]
        valid_accuracies = [accuracies[i] for i in valid_indices]
        valid_counts = [counts[i] for i in valid_indices]

        # Plot reliability curve
        ax1.plot(
            valid_confidences,
            valid_accuracies,
            "o-",
            linewidth=2,
            markersize=8,
            label="Model",
        )
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")

        # Add bin counts as bubble sizes
        sizes = [count * 50 / max(valid_counts) for count in valid_counts]
        ax1.scatter(
            valid_confidences, valid_accuracies, s=sizes, alpha=0.3, color="blue"
        )

        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Fraction of Positives")
        ax1.set_title("Reliability Curve")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        # Add ECE annotation
        ax1.text(
            0.05,
            0.95,
            f"ECE: {calibration_metrics.ece:.4f}",
            transform=ax1.transAxes,
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

        # Histogram of predictions
        all_predictions = []
        for i, count in enumerate(counts):
            if count > 0:
                bin_center = calibration_metrics.bin_confidences[i]
                all_predictions.extend([bin_center] * count)

        if all_predictions:
            ax2.hist(all_predictions, bins=20, alpha=0.7, edgecolor="black")
            ax2.set_xlabel("Predicted Probability")
            ax2.set_ylabel("Count")
            ax2.set_title("Distribution of Predictions")
            ax2.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Reliability diagram saved to {save_path}")

        return save_path

    def plot_edge_bucket_analysis(
        self,
        edge_metrics: EdgeBucketMetrics,
        title: str = "Edge Bucket Analysis",
        save_path: str | None = None,
    ) -> str | None:
        """
        Create edge bucket analysis visualization.

        Args:
            edge_metrics: Edge bucket metrics to plot
            title: Plot title
            save_path: Path to save plot (optional)

        Returns:
            Path to saved plot or None if matplotlib unavailable
        """

        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, cannot create edge bucket plot")
            return self._create_text_edge_analysis(edge_metrics, save_path)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Prepare data
        edge_labels = [
            f"{low:.2f} to {high:.2f}" for low, high in edge_metrics.edge_ranges
        ]
        valid_buckets = [
            i for i, count in enumerate(edge_metrics.bucket_counts) if count > 0
        ]

        if not valid_buckets:
            logger.warning("No valid buckets for edge analysis")
            plt.close(fig)
            return None

        # 1. ROI by Edge Bucket
        rois = [edge_metrics.bucket_rois[i] for i in valid_buckets]
        bucket_labels = [edge_labels[i] for i in valid_buckets]

        colors = ["red" if roi < 0 else "green" for roi in rois]
        bars1 = ax1.bar(range(len(valid_buckets)), rois, color=colors, alpha=0.7)
        ax1.set_xlabel("Edge Bucket")
        ax1.set_ylabel("ROI")
        ax1.set_title("ROI by Edge Bucket")
        ax1.set_xticks(range(len(valid_buckets)))
        ax1.set_xticklabels(bucket_labels, rotation=45, ha="right")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.5)

        # Add value labels on bars
        for bar, roi in zip(bars1, rois, strict=False):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (0.01 if height >= 0 else -0.01),
                f"{roi:.2f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
            )

        # 2. Hit Rate by Edge Bucket
        hit_rates = [edge_metrics.bucket_hit_rates[i] for i in valid_buckets]
        ax2.bar(range(len(valid_buckets)), hit_rates, alpha=0.7, color="blue")
        ax2.set_xlabel("Edge Bucket")
        ax2.set_ylabel("Hit Rate")
        ax2.set_title("Hit Rate by Edge Bucket")
        ax2.set_xticks(range(len(valid_buckets)))
        ax2.set_xticklabels(bucket_labels, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random (50%)")
        ax2.legend()

        # 3. Bet Count by Edge Bucket
        counts = [edge_metrics.bucket_counts[i] for i in valid_buckets]
        ax3.bar(range(len(valid_buckets)), counts, alpha=0.7, color="orange")
        ax3.set_xlabel("Edge Bucket")
        ax3.set_ylabel("Number of Bets")
        ax3.set_title("Bet Volume by Edge Bucket")
        ax3.set_xticks(range(len(valid_buckets)))
        ax3.set_xticklabels(bucket_labels, rotation=45, ha="right")
        ax3.grid(True, alpha=0.3)

        # 4. Cumulative Profit
        profits = [edge_metrics.bucket_profits[i] for i in valid_buckets]
        cumulative_profits = np.cumsum(profits)
        ax4.plot(
            range(len(valid_buckets)),
            cumulative_profits,
            "o-",
            linewidth=2,
            markersize=6,
        )
        ax4.set_xlabel("Edge Bucket")
        ax4.set_ylabel("Cumulative Profit")
        ax4.set_title("Cumulative Profit by Edge Bucket")
        ax4.set_xticks(range(len(valid_buckets)))
        ax4.set_xticklabels(bucket_labels, rotation=45, ha="right")
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color="black", linestyle="-", alpha=0.5)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Edge bucket analysis saved to {save_path}")

        return save_path

    def plot_performance_comparison(
        self,
        metrics_list: list[ComprehensiveMetrics],
        model_names: list[str],
        title: str = "Model Performance Comparison",
        save_path: str | None = None,
    ) -> str | None:
        """
        Compare performance across multiple models.

        Args:
            metrics_list: List of metrics for each model
            model_names: Names for each model
            title: Plot title
            save_path: Path to save plot (optional)

        Returns:
            Path to saved plot or None if matplotlib unavailable
        """

        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, cannot create comparison plot")
            return self._create_text_comparison(metrics_list, model_names, save_path)

        _fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # 1. Overall Scores
        overall_scores = [
            m.overall_score if m.overall_score else 0 for m in metrics_list
        ]
        ax = axes[0]
        bars = ax.bar(model_names, overall_scores, alpha=0.7)
        ax.set_ylabel("Overall Score")
        ax.set_title("Overall Performance Score")
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, score in zip(bars, overall_scores, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{score:.1f}",
                ha="center",
                va="bottom",
            )

        # 2. Accuracy Comparison
        accuracies = []
        for m in metrics_list:
            if m.classification_metrics:
                accuracies.append(m.classification_metrics.accuracy)
            else:
                accuracies.append(0)

        ax = axes[1]
        ax.bar(model_names, accuracies, alpha=0.7, color="green")
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification Accuracy")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        # 3. Calibration (ECE)
        eces = []
        for m in metrics_list:
            if m.calibration_metrics:
                eces.append(m.calibration_metrics.ece)
            else:
                eces.append(0)

        ax = axes[2]
        ax.bar(model_names, eces, alpha=0.7, color="red")
        ax.set_ylabel("Expected Calibration Error")
        ax.set_title("Model Calibration (Lower is Better)")
        ax.grid(True, alpha=0.3)

        # 4. Betting ROI (if available)
        rois = []
        for m in metrics_list:
            if m.edge_bucket_metrics:
                rois.append(m.edge_bucket_metrics.total_roi)
            else:
                rois.append(0)

        if any(roi != 0 for roi in rois):
            ax = axes[3]
            colors = ["red" if roi < 0 else "green" for roi in rois]
            ax.bar(model_names, rois, alpha=0.7, color=colors)
            ax.set_ylabel("Return on Investment")
            ax.set_title("Betting Performance")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        else:
            # Hide the subplot if no betting data
            axes[3].axis("off")

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Performance comparison saved to {save_path}")

        return save_path

    def plot_seasonal_performance(
        self,
        seasonal_metrics: dict[int, ComprehensiveMetrics],
        metric_name: str = "accuracy",
        title: str = "Seasonal Performance Trends",
        save_path: str | None = None,
    ) -> str | None:
        """
        Plot performance trends across seasons.

        Args:
            seasonal_metrics: Dict mapping season to metrics
            metric_name: Name of metric to plot
            title: Plot title
            save_path: Path to save plot (optional)

        Returns:
            Path to saved plot or None if matplotlib unavailable
        """

        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, cannot create seasonal plot")
            return self._create_text_seasonal_trends(
                seasonal_metrics, metric_name, save_path
            )

        seasons = sorted(seasonal_metrics.keys())
        values = []

        for season in seasons:
            metrics = seasonal_metrics[season]
            if metric_name == "accuracy" and metrics.classification_metrics:
                values.append(metrics.classification_metrics.accuracy)
            elif metric_name == "ece" and metrics.calibration_metrics:
                values.append(metrics.calibration_metrics.ece)
            elif metric_name == "roi" and metrics.edge_bucket_metrics:
                values.append(metrics.edge_bucket_metrics.total_roi)
            else:
                values.append(0)

        plt.figure(figsize=self.figsize)
        plt.plot(seasons, values, "o-", linewidth=2, markersize=8)
        plt.xlabel("Season")
        plt.ylabel(metric_name.capitalize())
        plt.title(title)
        plt.grid(True, alpha=0.3)

        # Add trend line
        if len(seasons) > 2:
            z = np.polyfit(seasons, values, 1)
            p = np.poly1d(z)
            plt.plot(
                seasons,
                p(seasons),
                "--",
                alpha=0.5,
                color="red",
                label=f"Trend: {z[0]:.4f}/year",
            )
            plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Seasonal performance plot saved to {save_path}")

        return save_path

    def create_metrics_dashboard(
        self,
        comprehensive_metrics: ComprehensiveMetrics,
        output_dir: str,
        model_name: str = "Model",
    ) -> dict[str, str]:
        """
        Create a complete dashboard of all metrics visualizations.

        Args:
            comprehensive_metrics: Complete metrics to visualize
            output_dir: Directory to save plots
            model_name: Name of the model

        Returns:
            Dict mapping plot type to file path
        """

        import os

        os.makedirs(output_dir, exist_ok=True)

        plots_created = {}

        # Reliability diagram
        if comprehensive_metrics.calibration_metrics:
            reliability_path = os.path.join(output_dir, f"{model_name}_reliability.png")
            result = self.plot_reliability_diagram(
                comprehensive_metrics.calibration_metrics,
                title=f"{model_name} - Reliability Analysis",
                save_path=reliability_path,
            )
            if result:
                plots_created["reliability"] = result

        # Edge bucket analysis
        if comprehensive_metrics.edge_bucket_metrics:
            edge_path = os.path.join(output_dir, f"{model_name}_edge_buckets.png")
            result = self.plot_edge_bucket_analysis(
                comprehensive_metrics.edge_bucket_metrics,
                title=f"{model_name} - Edge Bucket Analysis",
                save_path=edge_path,
            )
            if result:
                plots_created["edge_buckets"] = result

        logger.info(f"Created {len(plots_created)} plots in dashboard for {model_name}")
        return plots_created

    # Text-based fallback methods for when matplotlib is not available

    def _create_text_reliability_diagram(
        self, cal_metrics: CalibrationMetrics, save_path: str | None = None
    ) -> str | None:
        """Create text-based reliability diagram."""

        output = []
        output.append("RELIABILITY DIAGRAM (Text Format)")
        output.append("=" * 40)
        output.append(f"Expected Calibration Error (ECE): {cal_metrics.ece:.4f}")
        output.append(f"Maximum Calibration Error (MCE): {cal_metrics.mce:.4f}")
        output.append(f"Average Calibration Error (ACE): {cal_metrics.ace:.4f}")
        output.append("")
        output.append("Bin Analysis:")
        output.append("-" * 40)

        for i, (conf, acc, count) in enumerate(
            zip(
                cal_metrics.bin_confidences,
                cal_metrics.bin_accuracies,
                cal_metrics.bin_counts,
                strict=False,
            )
        ):
            if count > 0:
                output.append(
                    f"Bin {i + 1}: Confidence={conf:.3f}, Accuracy={acc:.3f}, Count={count}, Diff={abs(conf - acc):.3f}"
                )

        text_content = "\n".join(output)

        if save_path:
            with open(save_path.replace(".png", ".txt"), "w") as f:
                f.write(text_content)
            return save_path.replace(".png", ".txt")

        return None

    def _create_text_edge_analysis(
        self, edge_metrics: EdgeBucketMetrics, save_path: str | None = None
    ) -> str | None:
        """Create text-based edge bucket analysis."""

        output = []
        output.append("EDGE BUCKET ANALYSIS (Text Format)")
        output.append("=" * 50)
        output.append(f"Total ROI: {edge_metrics.total_roi:.4f}")
        output.append(f"Total Profit: ${edge_metrics.total_profit:.2f}")
        output.append(f"Total Bets: {edge_metrics.total_bets}")
        output.append("")
        output.append("Bucket Performance:")
        output.append("-" * 50)

        for _i, (edge_range, count, roi, hit_rate) in enumerate(
            zip(
                edge_metrics.edge_ranges,
                edge_metrics.bucket_counts,
                edge_metrics.bucket_rois,
                edge_metrics.bucket_hit_rates,
                strict=False,
            )
        ):
            if count > 0:
                output.append(
                    f"Edge {edge_range[0]:.2f} to {edge_range[1]:.2f}: Count={count}, ROI={roi:.4f}, Hit Rate={hit_rate:.3f}"
                )

        text_content = "\n".join(output)

        if save_path:
            with open(save_path.replace(".png", ".txt"), "w") as f:
                f.write(text_content)
            return save_path.replace(".png", ".txt")

        return None

    def _create_text_comparison(
        self,
        metrics_list: list[ComprehensiveMetrics],
        model_names: list[str],
        save_path: str | None = None,
    ) -> str | None:
        """Create text-based model comparison."""

        output = []
        output.append("MODEL PERFORMANCE COMPARISON (Text Format)")
        output.append("=" * 60)

        for metrics, name in zip(metrics_list, model_names, strict=False):
            output.append(f"\n{name}:")
            output.append("-" * 20)
            output.append(
                f"Overall Score: {metrics.overall_score:.2f}"
                if metrics.overall_score
                else "Overall Score: N/A"
            )

            if metrics.classification_metrics:
                output.append(
                    f"Accuracy: {metrics.classification_metrics.accuracy:.4f}"
                )
                output.append(
                    f"Log Loss: {metrics.classification_metrics.log_loss:.4f}"
                )

            if metrics.calibration_metrics:
                output.append(f"ECE: {metrics.calibration_metrics.ece:.4f}")

            if metrics.edge_bucket_metrics:
                output.append(
                    f"Betting ROI: {metrics.edge_bucket_metrics.total_roi:.4f}"
                )

        text_content = "\n".join(output)

        if save_path:
            with open(save_path.replace(".png", ".txt"), "w") as f:
                f.write(text_content)
            return save_path.replace(".png", ".txt")

        return None

    def _create_text_seasonal_trends(
        self,
        seasonal_metrics: dict[int, ComprehensiveMetrics],
        metric_name: str,
        save_path: str | None = None,
    ) -> str | None:
        """Create text-based seasonal trends."""

        output = []
        output.append(f"SEASONAL {metric_name.upper()} TRENDS (Text Format)")
        output.append("=" * 60)

        seasons = sorted(seasonal_metrics.keys())
        for season in seasons:
            metrics = seasonal_metrics[season]

            if metric_name == "accuracy" and metrics.classification_metrics:
                value = metrics.classification_metrics.accuracy
            elif metric_name == "ece" and metrics.calibration_metrics:
                value = metrics.calibration_metrics.ece
            elif metric_name == "roi" and metrics.edge_bucket_metrics:
                value = metrics.edge_bucket_metrics.total_roi
            else:
                value = None

            if value is not None:
                output.append(f"Season {season}: {metric_name} = {value:.4f}")

        text_content = "\n".join(output)

        if save_path:
            with open(save_path.replace(".png", ".txt"), "w") as f:
                f.write(text_content)
            return save_path.replace(".png", ".txt")

        return None
