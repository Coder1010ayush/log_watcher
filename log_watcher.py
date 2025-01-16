# --------------------------------------------- utf-8 encoding ----------------------------------------------------
"""  

    keep tracking of changes happening in a specified file and send these change to the end
    user after a particular interval via email.
    
"""
import argparse
import json
import time
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from watcher import MLLogWatcher
from utils import (
    MetricPluginManager, StandardMetricParser,
    WERMetricParser, MetricResult
)
import pandas as pd
import seaborn as sns
from pathlib import Path


class ModularMLLogParser:
    """Modular parser for machine learning training logs."""

    def __init__(self):
        self.plugin_manager = MetricPluginManager()
        self._setup_default_parsers()

    def _setup_default_parsers(self):
        """Set up default metric parsers."""
        # Standard metrics
        self.plugin_manager.add_parser(
            StandardMetricParser('Loss', r'loss[:\s]+([\d\.]+)')
        )
        self.plugin_manager.add_parser(
            StandardMetricParser('Accuracy', r'accuracy[:\s]+([\d\.]+)')
        )
        self.plugin_manager.add_parser(
            StandardMetricParser('Val_Loss', r'val_loss[:\s]+([\d\.]+)')
        )
        self.plugin_manager.add_parser(
            StandardMetricParser('Val_Accuracy', r'val_accuracy[:\s]+([\d\.]+)')
        )

        # WER metric
        self.plugin_manager.add_parser(WERMetricParser())

    def add_custom_metric(self, name: str, pattern: str):
        """Add a custom metric parser."""
        self.plugin_manager.add_parser(
            StandardMetricParser(name, pattern)
        )

    def parse_line(self, line: str) -> List[MetricResult]:
        """Parse a line of log output."""
        return self.plugin_manager.parse_line(line)

    def generate_training_plots(self, save_dir: str) -> List[str]:
        """Generate and save training progress plots."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        saved_plots = []

        # Get all metrics data
        plot_data = self.plugin_manager.get_all_plot_data()

        # Create main metrics plot
        plt.figure(figsize=(12, 6))
        for metric_name, values in plot_data.items():
            if metric_name in ['Loss', 'Val_Loss', 'Accuracy', 'Val_Accuracy']:
                plt.plot(values, label=metric_name)
        plt.title('Training Progress - Main Metrics')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        main_plot_path = str(Path(save_dir) / 'main_metrics.png')
        plt.savefig(main_plot_path)
        plt.close()
        saved_plots.append(main_plot_path)

        # Create WER-specific plot if WER data exists
        if 'WER' in plot_data:
            plt.figure(figsize=(12, 6))

            # Plot WER
            plt.plot(plot_data['WER'], label='WER', color='red')

            # Plot error components if available
            if 'Substitutions' in plot_data:
                plt.plot(plot_data['Substitutions'],
                         label='Substitutions', linestyle='--')
            if 'Deletions' in plot_data:
                plt.plot(plot_data['Deletions'],
                         label='Deletions', linestyle=':')
            if 'Insertions' in plot_data:
                plt.plot(plot_data['Insertions'],
                         label='Insertions', linestyle='-.')

            plt.title('WER Progress')
            plt.xlabel('Iteration')
            plt.ylabel('Rate')
            plt.legend()
            plt.grid(True)
            wer_plot_path = str(Path(save_dir) / 'wer_progress.png')
            plt.savefig(wer_plot_path)
            plt.close()
            saved_plots.append(wer_plot_path)

        return saved_plots


class ModularMLLogWatcher(MLLogWatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = ModularMLLogParser()

    def add_custom_metric(self, name: str, pattern: str):
        """Add a custom metric to track."""
        self.parser.add_custom_metric(name, pattern)

    def format_email_body(self, new_content: str) -> str:
        """Format the email body with training metrics and analysis."""
        # Get all current metrics
        plot_data = self.parser.plugin_manager.get_all_plot_data()

        # Calculate training duration
        duration = time.time() - self.training_start_time
        hours = duration // 3600
        minutes = (duration % 3600) // 60

        # Create metrics summary
        metrics_summary = ""
        for metric_name, values in plot_data.items():
            if values:
                current = values[-1]
                best = min(values) if 'loss' in metric_name.lower() else max(values)
                metrics_summary += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{current:.4f}</td>
                    <td>{best:.4f}</td>
                </tr>"""

        body = f"""
        <html>
        <body>
        <h2>Training Progress Report</h2>
        <p>Training Duration: {int(hours)}h {int(minutes)}m</p>
        
        <h3>Metrics Summary:</h3>
        <table border="1">
            <tr>
                <th>Metric</th>
                <th>Current</th>
                <th>Best</th>
            </tr>
            {metrics_summary}
        </table>

        <h3>Recent Training Log:</h3>
        <pre>{new_content}</pre>
        </body>
        </html>
        """
        return body


def main():
    parser = argparse.ArgumentParser(
        description='Watch ML training logs with modular metric tracking'
    )
    parser.add_argument('log_file', help='Path to the training log file to watch')
    parser.add_argument(
        '--check-interval',
        type=int,
        default=30,
        help='How often to check for changes (in seconds)'
    )
    parser.add_argument(
        '--email-interval',
        type=int,
        default=30,
        help='How often to send progress reports (in seconds)'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='training_plots',
        help='Directory to save training plots'
    )
    parser.add_argument(
        '--custom-metrics',
        type=str,
        default=None,
        help='JSON file containing custom metric patterns to track'
    )
    args = parser.parse_args()

    watcher = ModularMLLogWatcher(
        log_file=args.log_file,
        check_interval=args.check_interval,
        email_interval=args.email_interval,
        plot_dir=args.plot_dir
    )

    # Add custom metrics if specified
    if args.custom_metrics:
        with open(args.custom_metrics, 'r') as f:
            custom_metrics = json.load(f)
            for name, pattern in custom_metrics.items():
                watcher.add_custom_metric(name, pattern)

    watcher.watch()


if __name__ == "__main__":
    main()
