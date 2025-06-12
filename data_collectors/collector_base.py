from abc import ABC, abstractmethod
from pathlib import Path
from datasets import Dataset
import json
import re
from data_collectors.utils import UNICODE_TO_LATEX, EXCLUDE_SECTIONS

class CorpusCollectorBase(ABC):
    def __init__(self, output_file: str):
        self.output_file = Path(output_file)

    @abstractmethod
    def collect(self):
        """Fetch raw documents. Subclass must implement."""
        pass

    @abstractmethod
    def clean(self, raw_text: str) -> str:
        """Clean a single raw document. Subclass must implement."""
        pass

    @abstractmethod
    def run(self, output_file: str = None):
        """End-to-end collect + clean + save. Subclass must implement."""
        pass

    def push_to_hub(self, repo_id: str, output_file: str = None):
        path = Path(output_file) if output_file else self.output_file
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        data = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    "text": item["text"],
                    "source": item["source"]
                })

        dataset = Dataset.from_list(data)
        dataset.push_to_hub(repo_id)

    # shared text cleaning utils
    def _remove_invisible_unicode(self, text):
        return re.sub(r"[\u200b\u200c\u200d\u2060-\u206f\ufeff]", "", text)

    def _remove_citation_markers(self, text):
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
        return text

    def _clean_latex_noise(self, text):
        text = re.sub(r"\\displaystyle\s*", "", text)
        text = re.sub(r"\\mathrm\s*{(.*?)}", r"\1", text)
        text = re.sub(r"\\mathsf\s*{(.*?)}", r"\1", text)
        text = re.sub(r"\\textrm\s*{(.*?)}", r"\1", text)
        text = text.replace("\\left", "").replace("\\right", "")
        return text

    def _fix_newline_artifacts(self, text):
        return re.sub(r"([a-zA-Z0-9])\s*\n\s*([a-zA-Z0-9])", r"\1\2", text)

    def _normalize_whitespace(self, text):
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()
    
    def _remove_visual_artifacts(self, text):
        text = re.sub(r"\[\[File:.*?\]\]", "", text)
        text = re.sub(r"\[\[Image:.*?\]\]", "", text)
        # text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)  # tables
        text = re.sub(r"\{\{Infobox.*?\}\}", "", text, flags=re.DOTALL)
        return text


    def _apply_unicode_to_latex(self, text):
        for uni, latex in UNICODE_TO_LATEX.items():
            text = text.replace(uni, latex)
        return text

    def _remove_unmapped_unicode(self, text):
        allowed = set(UNICODE_TO_LATEX.keys())
        return ''.join(ch for ch in text if ord(ch) < 128 or ch in allowed)