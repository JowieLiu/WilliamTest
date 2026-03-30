import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: Dict
    year: int
    section_title: str
    parent_id: Optional[str] = None
    chunk_type: str = "child"


class DataProcessor:
    def __init__(self, json_path: str):
        self.json_path = json_path

    def load_data(self) -> List[Dict]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return list(data.values())[0]

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def split_into_chunks(self, text: str, chunk_size: int = 350,
                          overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks

    def process_all(self) -> List[DocumentChunk]:
        raw_data = self.load_data()
        chunks = []
        chunk_id = 0

        for item in raw_data:
            year = item.get('file_fiscal_year', 0)
            section_title = item.get('section_title', '')
            section_text = item.get('section_text', '')

            if not section_text:
                continue

            cleaned_text = self.clean_text(section_text)
            
            # 创建父块（整个章节的摘要）
            parent_chunk = DocumentChunk(
                id=f"parent_{chunk_id}",
                text=f"Section: {section_title}\n\n{cleaned_text[:1000]}",
                metadata={
                    'year': year,
                    'section_title': section_title,
                    'section_id': item.get('section_id', 0),
                    'is_parent': True
                },
                year=year,
                section_title=section_title,
                parent_id=None,
                chunk_type="parent"
            )
            chunks.append(parent_chunk)
            parent_id = f"parent_{chunk_id}"
            chunk_id += 1
            
            # 创建子块（详细内容切块）
            text_chunks = self.split_into_chunks(cleaned_text)
            
            for i, text_chunk in enumerate(text_chunks):
                chunk = DocumentChunk(
                    id=f"child_{chunk_id}",
                    text=text_chunk,
                    metadata={
                        'year': year,
                        'section_title': section_title,
                        'section_id': item.get('section_id', 0),
                        'is_parent': False,
                        'parent_id': parent_id
                    },
                    year=year,
                    section_title=section_title,
                    parent_id=parent_id,
                    chunk_type="child"
                )
                chunks.append(chunk)
                chunk_id += 1

        return chunks
