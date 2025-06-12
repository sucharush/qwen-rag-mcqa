import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json
import re
from typing import List
from data_collectors.collector_base import CorpusCollectorBase

class ShervineHTMLCollector(CorpusCollectorBase):
    def __init__(self, urls: List[str], output_file: str):
        super().__init__(output_file)
        self.urls = urls

    def collect(self) -> List[dict]:
        documents = []

        for url in self.urls:
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")
                body = soup.find("body")
                text = body.get_text(separator="\n", strip=True) if body else ""
            except Exception as e:
                print(f"[error] failed to parse {url}: {e}")
                continue

            if len(text) < 200:
                continue

            documents.append({
                "title": url.split("/")[-1].replace("-", " ").capitalize(),
                "raw": text,
                "source": url
            })

        return documents

    def clean(self, raw_text: str) -> str:
        text = self._remove_unmapped_unicode(raw_text)
        text = self._apply_unicode_to_latex(text)
        text = self._normalize_whitespace(text)
        return text

    def run(self, output_file: str = None):
        output_path = Path(output_file) if output_file else self.output_file
        cleaned_docs = []

        for doc in self.collect():
            cleaned = self.clean(doc["raw"])
            cleaned_docs.append({
                "title": doc["title"],
                "text": cleaned,
                "source": doc["source"]
            })

        with open(output_path, "w", encoding="utf-8") as fout:
            for doc in cleaned_docs:
                json.dump(doc, fout, ensure_ascii=False)
                fout.write("\n")

        return output_path
