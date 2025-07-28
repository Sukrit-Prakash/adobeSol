#!/usr/bin/env python3
"""
Fine-Tuned Adobe Hackathon Round 1A - Approach 2: Hybrid ML Feature Classification
Enhanced for super accuracy on edge cases (invitations, forms, flyers, long RFPs)
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
from sklearn.ensemble import RandomForestClassifier  # Upgraded from SimpleDecisionTree

class EnhancedMLHeadingExtractor:
    def __init__(self):
        self.title = ""
        self.outline = []
        self.font_stats = {}
        self.base_font_size = 0
        self.classifier = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        self.is_trained = False
        self.feature_names = [
            'font_size_ratio', 'is_bold', 'is_italic', 'line_height_ratio',
            'word_count', 'char_count', 'has_number_prefix', 'all_caps_ratio',
            'position_x_norm', 'position_y_norm', 'line_width_ratio',
            'vertical_spacing_ratio', 'font_size_absolute', 'special_chars_ratio',
            'title_case_ratio',  # New features below for fine-tuning
            'uppercase_density', 'is_centered', 'has_bullet', 'text_density',
            'has_section_pattern', 'is_all_upper', 'horizontal_alignment_score',
            'is_form_field', 'page_density_ratio'
        ]
    
    def extract_font_statistics(self, doc: fitz.Document) -> Dict[str, Any]:
        # Same as original, but add page_density calculation
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
            'avg_page_density': total_text_length / max(total_pages, 1)  # New: for sparse docs
        }
        return stats
    
    def extract_features(self, line_data: Dict, font_stats: Dict, 
                        page_height: float, page_width: float, prev_y: float) -> List[float]:
        text = line_data['text']
        font_size = line_data['font_size']
        font_flags = line_data['font_flags']
        bbox = line_data['bbox']
        
        # Original features
        font_size_ratio = font_size / font_stats['base_font_size']
        line_height = bbox[3] - bbox[1]
        line_height_ratio = line_height / font_stats['avg_line_height']
        line_width = bbox[2] - bbox[0]
        line_width_ratio = line_width / max(page_width, 1)
        word_count = len(text.split())
        char_count = len(text)
        is_bold = float(bool(font_flags & (1 << 4)))
        is_italic = float(bool(font_flags & (1 << 1)))
        has_number_prefix = float(bool(re.match(r'^\d+\.?\s+', text)))
        all_caps_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
        title_case_ratio = sum(1 for word in text.split() if word and word[0].isupper()) / max(word_count, 1)
        special_chars_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(char_count, 1)
        position_x_norm = bbox[0] / max(page_width, 1)
        position_y_norm = bbox[1] / max(page_height, 1)
        vertical_spacing = abs(bbox[1] - prev_y) if prev_y > 0 else 0
        vertical_spacing_ratio = vertical_spacing / font_stats['avg_line_height']
        
        # New fine-tuned features for edge cases
        uppercase_density = all_caps_ratio * (1 if text.isupper() else 0.5)  # Boost for all-caps in invites/flyers
        is_centered = float(abs(position_x_norm - 0.5) < 0.1 and line_width_ratio > 0.4)  # Centered text in invites
        has_bullet = float(text.startswith('*') or text.startswith('-') or text.startswith('•'))  # For flyers/lists
        text_density = char_count / max(line_width, 1)  # Low density for sparse forms/invites
        has_section_pattern = float(bool(re.match(r'^(Appendix|Section|Chapter|Phase)\s*[A-Z0-9]', text)))  # For RFPs
        is_all_upper = float(text.isupper() and char_count > 5)  # All-caps headings in flyers
        horizontal_alignment_score = position_x_norm + (1 - line_width_ratio)  # Left-aligned vs. full-width
        is_form_field = float(bool(re.match(r'^\d+\.\s', text)) and word_count < 10)  # Numbered forms
        page_density_ratio = font_stats['avg_page_density'] / 1000  # Normalize for sparse docs
        
        return [
            font_size_ratio, is_bold, is_italic, line_height_ratio,
            word_count, char_count, has_number_prefix, all_caps_ratio,
            position_x_norm, position_y_norm, line_width_ratio,
            vertical_spacing_ratio, font_size, special_chars_ratio,
            title_case_ratio, uppercase_density, is_centered, has_bullet,
            text_density, has_section_pattern, is_all_upper,
            horizontal_alignment_score, is_form_field, page_density_ratio
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
                        if not line["spans"]:
                            continue
                        
                        text = "".join(s["text"] for s in line["spans"]).strip()
                        if len(text) < 3:
                            continue
                        
                        span = line["spans"][0]
                        line_data = {
                            'text': text,
                            'font_size': span["size"],
                            'font_flags': span["flags"],
                            'bbox': line["bbox"]
                        }
                        
                        feature_vector = self.extract_features(
                            line_data, self.font_stats, page_height, page_width, prev_y
                        )
                        
                        label = self._enhanced_heuristic_label(line_data, self.font_stats, page_num)
                        
                        features.append(feature_vector)
                        labels.append(label)
                        
                        prev_y = line["bbox"][1]
        
        return np.array(features), np.array(labels)
    
    def _enhanced_heuristic_label(self, line_data: Dict, font_stats: Dict, page_num: int) -> str:
        """Improved heuristics with edge case rules"""
        text = line_data['text']
        font_size = line_data['font_size']
        font_flags = line_data['font_flags']
        bbox = line_data['bbox']
        
        size_ratio = font_size / font_stats['base_font_size']
        is_bold = bool(font_flags & (1 << 4))
        is_sparse_doc = font_stats['avg_page_density'] < 500  # For invitations/forms
        
        # Edge case: Sparse docs (invites/forms) - Promote all-caps or numbered items
        if is_sparse_doc:
            if text.isupper() and len(text.split()) > 2:
                return 'H1' if size_ratio > 1.2 else 'H2'
            if re.match(r'^\d+\.\s', text):
                return 'H1'  # Form fields
        
        # Title detection (larger, early on page)
        if (size_ratio > 1.6 and len(text.split()) <= 12 and page_num < 2 and 
            not re.match(r'^\d+\.', text)):
            return 'TITLE'
        
        # Enhanced heading patterns for RFPs/flyers
        if re.match(r'^\d+\.\s+[A-Z]', text) and (size_ratio > 1.2 or is_bold):
            return 'H1'
        if re.match(r'^\d+\.\d+\.?\s+', text) and (size_ratio > 1.1 or is_bold):
            return 'H2'
        if re.match(r'^\d+\.\d+\.\d+\.?\s+', text) or (is_bold and size_ratio > 1.05):
            return 'H3'
        if re.match(r'^(Appendix|Phase|Section)\s*[A-Z]', text):
            return 'H1'  # Appendices in RFPs
        
        # All-caps short text in flyers
        if text.isupper() and 3 <= len(text.split()) <= 8 and (size_ratio > 1.1 or is_bold):
            return 'H1'
        
        # Bold text with size increase
        if is_bold and size_ratio > 1.08 and len(text.split()) <= 10:
            if size_ratio > 1.15:
                return 'H1'
            elif size_ratio > 1.1:
                return 'H2'
            else:
                return 'H3'
        
        return 'TEXT'
    
    def train_classifier(self, doc: fitz.Document):
        X, y = self.generate_training_data(doc)
        if len(X) > 10:  # Minimum samples for training
            self.classifier.fit(X, y)
            self.is_trained = True
            print(f"Trained on {len(X)} samples with RandomForest")
    
    def classify_line(self, line_data: Dict, page_height: float, 
                     page_width: float, prev_y: float) -> str:
        if not self.is_trained:
            return self._enhanced_heuristic_label(line_data, self.font_stats, 0)
        
        features = self.extract_features(line_data, self.font_stats, 
                                       page_height, page_width, prev_y)
        prediction = self.classifier.predict([features])[0]
        
        # Post-ML ensemble rules for fine-tuning edge cases
        text = line_data['text']
        if prediction in ['H1', 'H2', 'H3'] and len(text) < 3:  # Filter short false positives
            return 'TEXT'
        if self.font_stats['avg_page_density'] < 300 and text.isupper():  # Promote in sparse invites
            return 'H1' if prediction == 'H2' else prediction
        if re.match(r'^\d+\.$', text):  # Demote pure numbers in forms
            return 'TEXT'
        
        return prediction
    
    def extract_title(self, doc: fitz.Document) -> str:
        # Enhanced with density check for sparse docs
        candidates = []
        is_sparse = self.font_stats['avg_page_density'] < 500
        
        for page_num in range(min(2, len(doc))):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:
                    for line in block["lines"]:
                        if not line["spans"]:
                            continue
                        
                        text = "".join(s["text"] for s in line["spans"]).strip()
                        if len(text) < 5 or len(text) > 100:
                            continue
                        
                        span = line["spans"][0]
                        font_size = span["size"]
                        bbox = line["bbox"]
                        
                        size_score = font_size / self.font_stats['base_font_size'] * (1.5 if is_sparse else 1)
                        position_score = 1.0 - (bbox[1] / page.rect.height)
                        length_score = max(0, 1.0 - abs(len(text.split()) - 7) * 0.1)
                        upper_score = 1 if text.isupper() else 0  # Boost for all-caps in invites
                        
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
                            if not line["spans"]:
                                continue
                            
                            text = "".join(s["text"] for s in line["spans"]).strip()
                            if len(text) < 3:
                                continue
                            
                            span = line["spans"][0]
                            line_data = {
                                'text': text,
                                'font_size': span["size"],
                                'font_flags': span["flags"],
                                'bbox': line["bbox"]
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

def main():
    """Main execution function"""
    project_root = Path(__file__).parent.resolve()
    input_dir = project_root / "input"
    output_dir = project_root / "output4"
    
    output_dir.mkdir(exist_ok=True)
    
    for pdf_file in input_dir.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        
        extractor = EnhancedMLHeadingExtractor()
        result = extractor.process_pdf(str(pdf_file))
        
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {output_file.name}")
        print(f"Title: {result['title']}")
        print(f"Headings found: {len(result['outline'])}")

if __name__ == "__main__":
    main()
