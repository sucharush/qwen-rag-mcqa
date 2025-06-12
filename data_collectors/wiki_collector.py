import re
import json
from typing import List
from pathlib import Path
from wikipediaapi import Wikipedia, WikipediaPage
from datasets import Dataset
from data_collectors.collector_base import CorpusCollectorBase
from data_collectors.utils import UNICODE_TO_LATEX, EXCLUDE_SECTIONS

class WikiCollector(CorpusCollectorBase):
    def __init__(self, topics: list, wiki: Wikipedia, output_file: str):
        super().__init__(output_file)
        self.topics = topics
        self.wiki = wiki

    def collect(self) -> List[dict]:
        documents = []
        exclude_titles = {title.lower() for title in EXCLUDE_SECTIONS}

        for topic in self.topics:
            page = self.wiki.page(topic)
            if not page.exists():
                continue

            lines = [page.summary.strip()]
            skip_level = None

            def recurse(section):
                nonlocal skip_level
                title = section.title.strip().lower()

                if skip_level is not None:
                    if section.level > skip_level:
                        return
                    else:
                        skip_level = None

                if any(excl in title for excl in exclude_titles):
                    skip_level = section.level
                    return

                if section.text.strip():
                    lines.append(section.text.strip())

                for subsection in section.sections:
                    recurse(subsection)

            for section in page.sections:
                recurse(section)

            raw_text = "\n\n".join(lines)
            documents.append({
                "title": topic,
                "raw": raw_text,
                "source": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
            })

        return documents


    def clean(self, raw_text: str) -> str:
        # use methods inherited from base (or mixin)
        text = self._remove_invisible_unicode(raw_text)
        text = self._remove_visual_artifacts(text)
        text = self._remove_citation_markers(text)
        text = self._clean_latex_noise(text)
        text = self._remove_unmapped_unicode(text)
        text = self._apply_unicode_to_latex(text)
        text = self._fix_newline_artifacts(text)
        text = self._normalize_whitespace(text)
        return text

    def run(self, output_file: str = None):
        output_path = Path(output_file) if output_file else self.output_file
        cleaned_docs = []

        for doc in self.collect():
            cleaned = self.clean(doc["raw"])
            if len(cleaned) < 300:
                continue
            cleaned_docs.append({
                "title": doc["title"],
                "text": cleaned,
                "source": doc["source"]
            })

        with open(output_path, "w") as fout:
            for doc in cleaned_docs:
                json.dump(doc, fout)
                fout.write("\n")

        return output_path

