# --------------------------------------- utf-8 encoding ----------------------------------------------
from abc import ABC, abstractmethod
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# wwil kljz rakd yeyn


@dataclass
class MetricResult:
    """Container for a parsed metric result."""
    name: str
    value: float
    epoch: Optional[int] = None
    step: Optional[int] = None
    extra_info: Optional[Dict[str, Any]] = None


class BaseMetricParser(ABC):
    """Abstract base class for metric parsers."""

    @abstractmethod
    def parse(self, line: str) -> Optional[MetricResult]:
        """Parse a line of text and return metric if found."""
        pass

    @abstractmethod
    def get_plot_data(self) -> Dict[str, List[float]]:
        """Return data for plotting."""
        pass


class StandardMetricParser(BaseMetricParser):
    """Parser for standard metrics like loss and accuracy."""

    def __init__(self, metric_name: str, pattern: str):
        self.metric_name = metric_name
        self.pattern = pattern
        self.values = []

    def parse(self, line: str) -> Optional[MetricResult]:
        match = re.search(self.pattern, line.lower())
        if match:
            value = float(match.group(1))
            self.values.append(value)
            return MetricResult(name=self.metric_name, value=value)
        return None

    def get_plot_data(self) -> Dict[str, List[float]]:
        return {self.metric_name: self.values}


class WERMetricParser(BaseMetricParser):
    """Parser specifically for Word Error Rate metric."""

    def __init__(self):
        self.wer_values = []
        self.substitutions = []
        self.deletions = []
        self.insertions = []

        # Patterns for different WER components
        self.patterns = {
            'wer': r'wer[:\s]+([\d\.]+)',
            'sub': r'substitutions[:\s]+(\d+)',
            'del': r'deletions[:\s]+(\d+)',
            'ins': r'insertions[:\s]+(\d+)'
        }

    def parse(self, line: str) -> Optional[MetricResult]:
        line = line.lower()
        metrics = {}

        for key, pattern in self.patterns.items():
            match = re.search(pattern, line)
            if match:
                metrics[key] = float(match.group(1))

        if 'wer' in metrics:
            self.wer_values.append(metrics['wer'])

            # Track error components if available
            if 'sub' in metrics:
                self.substitutions.append(metrics['sub'])
            if 'del' in metrics:
                self.deletions.append(metrics['del'])
            if 'ins' in metrics:
                self.insertions.append(metrics['ins'])

            return MetricResult(
                name='WER',
                value=metrics['wer'],
                extra_info={
                    'substitutions': metrics.get('sub'),
                    'deletions': metrics.get('del'),
                    'insertions': metrics.get('ins')
                }
            )
        return None

    def get_plot_data(self) -> Dict[str, List[float]]:
        data = {'WER': self.wer_values}
        if self.substitutions:
            data['Substitutions'] = self.substitutions
        if self.deletions:
            data['Deletions'] = self.deletions
        if self.insertions:
            data['Insertions'] = self.insertions
        return data


class MetricPluginManager:
    """Manager for metric parser plugins."""

    def __init__(self):
        self.parsers: List[BaseMetricParser] = []

    def add_parser(self, parser: BaseMetricParser):
        """Add a new metric parser."""
        self.parsers.append(parser)

    def parse_line(self, line: str) -> List[MetricResult]:
        """Parse a line using all registered parsers."""
        results = []
        for parser in self.parsers:
            result = parser.parse(line)
            if result:
                results.append(result)
        return results

    def get_all_plot_data(self) -> Dict[str, Dict[str, List[float]]]:
        """Get plot data from all parsers."""
        plot_data = {}
        for parser in self.parsers:
            parser_data = parser.get_plot_data()
            for metric_name, values in parser_data.items():
                plot_data[metric_name] = values
        return plot_data
