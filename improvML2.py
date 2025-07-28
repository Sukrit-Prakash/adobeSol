#!/usr/bin/env python3
"""
Super Fine-Tuned Adobe Hackathon Round 1A - Approach 2: Hybrid ML Feature Classification
Maximized accuracy for all edge cases with enhanced features, heuristics, and ensemble
Fixed runtime errors for NoneType and undefined names
"""

import os
import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import re
import math
import pymupdf as fitz
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

class SuperMLHeadingExtractor:
    def __init__(self):
        self.title = ""
        self.outline = []
        self.font_stats = {}
        self.base_font_size = 0
        self.classifier = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced')
        self.is_trained = False
        self.feature_names = [
            'font_size_ratio', 'is_bold', 'is_italic', 'line_height_ratio',
            'word_count', 'char_count', 'has_number_prefix', 'all_caps_ratio',
            'position_x_norm', 'position_y_norm', 'line_width_ratio',
            'vertical_spacing_ratio', 'font_size_absolute', 'special_chars_ratio',
            'title_case_ratio', 'uppercase_density', 'is_centered', 'has_bullet',
            'text_density', 'has_section_pattern', 'is_all_upper',
            'horizontal_alignment_score', 'is_form_field', 'page_density_ratio',
            'is_quoted_text', 'is_table_header', 'line_fragment_score'
        ]
    
    def extract_font_statistics(self, doc: fitz.Document) -> Dict[str, Any]:
        font_sizes = []
        font_names = []
        line_heights = []
        total_text_length = 0
        total_pages = len(doc)
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            page_text_length = 0
            
            for block in blocks:
                if block.get("type") == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
                            font_names.append(span["font"])
                            page_text_length += len(span["text"])
                        
                        bbox = line["bbox"]
                        line_heights.append(bbox[3] - bbox[1])
            
            total_text_length += page_text_length
        
        stats = {
            'font_sizes': Counter(font_sizes),
            'font_names': Counter(font_names),
            'avg_font_size': np.mean(font_sizes) if font_sizes else 12,
            'base_font_size': Counter(font_sizes).most_common(1)[0][0] if font_sizes else 12,
            'avg_line_height': np.mean(line_heights) if line_heights else 15,
            'font_size_std': np.std(font_sizes) if font_sizes else 2,
            'avg_page_density': total_text_length / max(total_pages, 1)
        }
        return stats
    
    # Helper methods for checks (fixes NameError and function/variable inconsistency)
    def _has_bullet(self, text: str) -> bool:
        return bool(text and text.startswith(('*', '-', '•')))
    
    def _is_quoted_text(self, text: str) -> bool:
        return bool(text and (text.startswith('"') or text.endswith('"') or '“' in text))
    
    def _is_table_header(self, text: str, word_count: int) -> bool:
        return bool(text and word_count < 3 and (text.isupper() or re.match(r'^\d{4}$', text)))
    
    def extract_features(self, line_data: Dict, font_stats: Dict, 
                        page_height: float, page_width: float, prev_y: float) -> List[float]:
        text = line_data.get('text') or ""  # Safe default to avoid NoneType
        font_size = line_data.get('font_size') or 0.0
        font_flags = line_data.get('font_flags') or 0
        bbox = line_data.get('bbox') or (0, 0, 0, 0)
        
        # Safe calculations with defaults
        font_size_ratio = font_size / font_stats['base_font_size'] if font_stats['base_font_size'] > 0 else 1.0
        line_height_ratio = (bbox[3] - bbox[1]) / font_stats['avg_line_height'] if font_stats['avg_line_height'] > 0 else 1.0
        line_width_ratio = (bbox[2] - bbox[0]) / max(page_width, 1)
        word_count = len(text.split())
        char_count = len(text)
        is_bold = float(bool(font_flags & (1 << 4)))
        is_italic = float(bool(font_flags & (1 << 1)))
        has_number_prefix = float(bool(re.match(r'^\d+\.?\s+', text)))
        all_caps_ratio = (sum(1 for c in text if c.isupper()) / max(char_count, 1)) if char_count > 0 else 0.0
        title_case_ratio = (sum(1 for word in text.split() if word and word[0].isupper()) / max(word_count, 1)) if word_count > 0 else 0.0
        special_chars_ratio = (sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(char_count, 1)) if char_count > 0 else 0.0
        position_x_norm = bbox[0] / max(page_width, 1)
        position_y_norm = bbox[1] / max(page_height, 1)
        vertical_spacing_ratio = abs(bbox[1] - prev_y) / font_stats['avg_line_height'] if prev_y > 0 and font_stats['avg_line_height'] > 0 else 0.0
        uppercase_density = all_caps_ratio * (1 if text.isupper() else 0.5)
        is_centered = float(abs(position_x_norm - 0.5) < 0.1 and line_width_ratio > 0.4)
        has_bullet = float(self._has_bullet(text))
        text_density = char_count / max(bbox[2] - bbox[0], 1) if (bbox[2] - bbox[0]) > 0 else 0.0
        has_section_pattern = float(bool(re.match(r'^(Appendix|Section|Chapter|Phase)\s*[A-Z0-9]', text)))
        is_all_upper = float(text.isupper() and char_count > 5)
        horizontal_alignment_score = position_x_norm + (1 - line_width_ratio)
        is_form_field = float(bool(re.match(r'^\d+\.\s', text)) and word_count < 10)
        page_density_ratio = font_stats['avg_page_density'] / 1000
        
        # New features with safe defaults
        is_quoted_text = float(self._is_quoted_text(text))
        is_table_header = float(self._is_table_header(text, word_count))
        line_fragment_score = float(char_count < 10 and not text.endswith('.'))
        
        return [
            font_size_ratio, is_bold, is_italic, line_height_ratio,
            word_count, char_count, has_number_prefix, all_caps_ratio,
            position_x_norm, position_y_norm, line_width_ratio,
            vertical_spacing_ratio, font_size, special_chars_ratio,
            title_case_ratio, uppercase_density, is_centered, has_bullet,
            text_density, has_section_pattern, is_all_upper,
            horizontal_alignment_score, is_form_field, page_density_ratio,
            is_quoted_text, is_table_header, line_fragment_score
        ]
    
    def generate_training_data(self, doc: fitz.Document) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            prev_y = 0
            page_width = page.rect.width
            page_height = page.rect.height
            
            for block in blocks:
                if block.get("type") == 0:
                    for line in block["lines"]:
                        try:
                            text = self._clean_text(line.get("spans", []))  # Safe get
                            if not text or len(text) < 3:
                                continue
                            
                            span = line["spans"][0] if line["spans"] else None
                            if not span:
                                continue
                            
                            line_data = {
                                'text': text,
                                'font_size': span.get("size", 0.0),
                                'font_flags': span.get("flags", 0),
                                'bbox': line.get("bbox", (0, 0, 0, 0))
                            }
                            
                            feature_vector = self.extract_features(
                                line_data, self.font_stats, page_height, page_width, prev_y
                            )
                            
                            label = self._super_heuristic_label(line_data, self.font_stats, page_num, page_height)
                            
                            features.append(feature_vector)
                            labels.append(label)
                            
                            prev_y = line["bbox"][1]
                        except Exception as e:
                            print(f"Skipping bad line: {e}")
                            continue
        
        return np.array(features), np.array(labels)
    
    def _clean_text(self, spans: List[Dict]) -> str:
        """Clean and merge spans to fix garbling/truncation"""
        if not spans:
            return ""
        text = "".join(s.get("text", "") for s in spans).strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        return text
    
    def _super_heuristic_label(self, line_data: Dict, font_stats: Dict, page_num: int, page_height: float) -> str:
        """Super-enhanced heuristics for max accuracy"""
        text = line_data.get('text') or ""
        font_size = line_data.get('font_size') or 0.0
        font_flags = line_data.get('font_flags') or 0
        bbox = line_data.get('bbox') or (0, 0, 0, 0)
        position_y_norm = bbox[1] / page_height if page_height > 0 else 0.0
        
        size_ratio = font_size / font_stats['base_font_size'] if font_stats['base_font_size'] > 0 else 1.0
        is_bold = bool(font_flags & (1 << 4))
        is_sparse_doc = font_stats['avg_page_density'] < 500
        
        # Edge case: Sparse docs (invites/forms/flyers)
        if is_sparse_doc:
            if text.isupper() and len(text.split()) > 2 and position_y_norm < 0.8:  # Avoid footers
                return 'H1' if size_ratio > 1.1 else 'H2'
            if re.match(r'^\d+\.\s', text):
                return 'H1'  # Form fields as flat H1
            if self._has_bullet(text) or self._is_quoted_text(text):
                return 'TEXT'  # Demote bullets/quotes in flyers
        
        # Title (prominent, early, long)
        if size_ratio > 1.5 and page_num < 2 and position_y_norm < 0.3 and len(text) > 20:
            return 'TITLE'
        
        # RFP/structured doc patterns
        if re.match(r'^(Appendix|Phase|OVERVIEW OF)\s', text):
            return 'H1'
        if re.match(r'^\d+\.\s+[A-Z]', text) and (size_ratio > 1.1 or is_bold):
            return 'H1' if size_ratio > 1.3 else 'H2'
        if re.match(r'^\d+\.\d+\.?\s+', text):
            return 'H2'
        if re.match(r'^\d+\.\d+\.\d+\.?\s+', text):
            return 'H3'
        
        # All-caps in flyers/invites
        if text.isupper() and 3 <= len(text.split()) <= 10 and (size_ratio > 1.05 or is_bold):
            return 'H1'
        
        # Bold/size-based fallback
        if is_bold and size_ratio > 1.05 and len(text.split()) <= 12:
            if size_ratio > 1.2:
                return 'H1'
            elif size_ratio > 1.1:
                return 'H2'
            else:
                return 'H3'
        
        # Filters for false positives
        word_count = len(text.split())
        if self._is_table_header(text, word_count) or text.lower() in ['funding source', 'page']:  # Table headers/footers
            return 'TEXT'
        
        return 'TEXT'
    
    def train_classifier(self, doc: fitz.Document):
        X, y = self.generate_training_data(doc)
        if len(X) > 50:  # Increased threshold for better training; fallback otherwise
            unique_classes = np.unique(y)
            class_weights = dict(zip(unique_classes, compute_class_weight('balanced', classes=unique_classes, y=y)))
            self.classifier.set_params(class_weight=class_weights)
            self.classifier.fit(X, y)
            self.is_trained = True
            print(f"Trained on {len(X)} samples with balanced RandomForest")
    
    def classify_line(self, line_data: Dict, page_height: float, 
                     page_width: float, prev_y: float) -> str:
        if not self.is_trained:
            return self._super_heuristic_label(line_data, self.font_stats, 0, page_height)
        
        features = self.extract_features(line_data, self.font_stats, 
                                       page_height, page_width, prev_y)
        probs = self.classifier.predict_proba([features])[0]
        prediction = self.classifier.predict([features])[0]
        
        # Probability threshold: Filter low-confidence
        if max(probs) < 0.6 and prediction != 'TEXT':
            return 'TEXT'
        
        # Ensemble rules
        text = line_data.get('text') or ""
        word_count = len(text.split())
        if prediction in ['H1', 'H2', 'H3'] and (len(text) < 5 or self._is_quoted_text(text) or self._is_table_header(text, word_count)):
            return 'TEXT'
        if self.font_stats['avg_page_density'] < 300 and text.isupper():
            return 'H1'  # Boost for invites
        
        return prediction
    
    def extract_title(self, doc: fitz.Document) -> str:
        candidates = []
        is_sparse = self.font_stats['avg_page_density'] < 500
        
        for page_num in range(min(2, len(doc))):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:
                    for line in block["lines"]:
                        text = self._clean_text(line.get("spans", []))
                        if len(text) < 10 or len(text) > 150:  # Adjusted range for long titles
                            continue
                        
                        span = line["spans"][0] if line["spans"] else None
                        if not span:
                            continue
                        
                        font_size = span.get("size", 0.0)
                        bbox = line.get("bbox", (0, 0, 0, 0))
                        
                        size_score = font_size / self.font_stats['base_font_size'] * (1.5 if is_sparse else 1) if self.font_stats['base_font_size'] > 0 else 1.0
                        position_score = (1 - (bbox[1] / page.rect.height)) * 2 if page.rect.height > 0 else 0.0
                        length_score = min(len(text) / 50, 2)  # Favor longer for descriptive titles
                        upper_score = 1 if text.isupper() else 0
                        
                        total_score = size_score * 2 + position_score + length_score + upper_score
                        
                        candidates.append({
                            'text': text,
                            'score': total_score,
                            'page': page_num
                        })
        
        if candidates:
            best = max(candidates, key=lambda x: x['score'])
            return best['text']
        
        return "Untitled Document"
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        try:
            doc = fitz.open(pdf_path)
            
            self.font_stats = self.extract_font_statistics(doc)
            
            self.train_classifier(doc)
            
            self.title = self.extract_title(doc)
            
            self.outline = []
            
            for page_num, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                prev_y = 0
                page_width = page.rect.width
                page_height = page.rect.height
                
                for block in blocks:
                    if block.get("type") == 0:
                        for line in block["lines"]:
                            try:
                                text = self._clean_text(line.get("spans", []))
                                if not text or len(text) < 3:
                                    continue
                                
                                span = line["spans"][0] if line["spans"] else None
                                if not span:
                                    continue
                                
                                line_data = {
                                    'text': text,
                                    'font_size': span.get("size", 0.0),
                                    'font_flags': span.get("flags", 0),
                                    'bbox': line.get("bbox", (0, 0, 0, 0))
                                }
                                
                                classification = self.classify_line(
                                    line_data, page_height, page_width, prev_y
                                )
                                
                                if classification in ['H1', 'H2', 'H3']:
                                    self.outline.append({
                                        "level": classification,
                                        "text": text,
                                        "page": page_num + 1
                                    })
                                
                                prev_y = line["bbox"][1]
                            except Exception as e:
                                print(f"Skipping bad line in processing: {e}")
                                continue
            
            # Post-process outline for hierarchy refinement
            self.outline = self._refine_outline(self.outline)
            
            doc.close()
            
            return {
                "title": self.title,
                "outline": self.outline
            }
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return {
                "title": "Error Processing Document",
                "outline": []
            }
    
    def _refine_outline(self, outline: List[Dict]) -> List[Dict]:
        """Post-process to fix levels and remove duplicates/fragments"""
        refined = []
        prev_text = ""
        for item in outline:
            text = item['text']
            if text == prev_text or len(text) < 5:  # Remove duplicates/short
                continue
            # Promote if too many H3 (e.g., in flyers)
            if item['level'] == 'H3' and re.match(r'^[A-Z\s]+$', text) and len(refined) > 5:
                item['level'] = 'H2'
            refined.append(item)
            prev_text = text
        return refined

def main():
    """Main execution function"""
    project_root = Path(__file__).parent.resolve()
    input_dir = project_root / "input"
    output_dir = project_root / "output5"
    
    output_dir.mkdir(exist_ok=True)
    
    for pdf_file in input_dir.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        
        extractor = SuperMLHeadingExtractor()
        result = extractor.process_pdf(str(pdf_file))
        
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {output_file.name}")
        print(f"Title: {result['title']}")
        print(f"Headings found: {len(result['outline'])}")

if __name__ == "__main__":
    main()
