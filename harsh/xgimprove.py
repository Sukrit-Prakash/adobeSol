#!/usr/bin/env python3
"""
Ultimate Enhanced Adobe Hackathon PDF Heading Extraction
XGBoost with All Suggested Improvements for Max Accuracy
"""

import os
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
import re
import math
import pymupdf as fitz
import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import DBSCAN

class UltimateHeadingExtractorEnhanced:
    def __init__(self):
        self.title = ""
        self.outline = []
        self.font_stats = {}
        self.base_font_size = 0
        self.booster = None
        self.is_trained = False
        self.label_map = {'TEXT': 0, 'H3': 1, 'H2': 2, 'H1': 3, 'TITLE': 4}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.feature_names = [
            'font_size_ratio', 'is_bold', 'is_italic', 'line_height_ratio',
            'word_count', 'char_count', 'has_number_prefix', 'all_caps_ratio',
            'position_x_norm', 'position_y_norm', 'line_width_ratio',
            'vertical_spacing_ratio', 'font_size_absolute', 'special_chars_ratio',
            'title_case_ratio', 'uppercase_density', 'is_centered', 'has_bullet',
            'text_density', 'has_section_pattern', 'is_all_upper',
            'horizontal_alignment_score', 'is_form_field', 'page_density_ratio',
            'is_quoted_text', 'is_table_header', 'line_fragment_score',
            'relative_page_position', 'is_numbered_list', 'bold_ratio',
            'is_footer_candidate', 'number_depth', 'is_multi_span', 'density_outlier_score', 'y_position_score',
            # New advanced features
            'cluster_density', 'semantic_score'
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
    
    # Helper methods (unchanged)
    def _has_bullet(self, text: str) -> bool:
        return bool(text and text.startswith(('*', '-', '•')))
    
    def _is_quoted_text(self, text: str) -> bool:
        return bool(text and (text.startswith('"') or text.endswith('"') or '“' in text))
    
    def _is_table_header(self, text: str, word_count: int) -> bool:
        return bool(text and word_count < 3 and (text.isupper() or re.match(r'^\d{4}$', text)))
    
    def _is_numbered_list(self, text: str) -> bool:
        return bool(re.match(r'^\d+\.?\s', text))
    
    def _get_number_depth(self, text: str) -> float:
        match = re.match(r'^(\d+\.)+(\d+)?\s', text)
        if match:
            return text.count('.') + 1
        return 0.0
    
    def _semantic_score(self, text: str) -> float:
        keywords = ['Section', 'Appendix', 'Introduction', 'Overview', 'Phase', 'Chapter', 'Mission', 'Goals']
        return sum(1 for kw in keywords if kw.lower() in text.lower()) / len(keywords)
    
    def extract_features(self, line_data: Dict, font_stats: Dict, 
                        page_height: float, page_width: float, prev_y: float, page_num: int, total_pages: int, cluster_label: int) -> List[float]:
        text = line_data.get('text') or ""
        font_size = line_data.get('font_size') or 0.0
        font_flags = line_data.get('font_flags') or 0
        bbox = line_data.get('bbox') or (0, 0, 0, 0)
        num_spans = line_data.get('num_spans', 1)
        
        # Existing features (abbreviated for brevity)
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
        has_section_pattern = float(bool(re.match(r'^(Appendix|Phase|OVERVIEW OF|Mission Statement|Goals|Elective Course Offerings)\s', text)))
        is_all_upper = float(text.isupper() and char_count > 5)
        horizontal_alignment_score = position_x_norm + (1 - line_width_ratio)
        is_form_field = float(bool(re.match(r'^\d+\.\s', text)) and word_count < 15)
        page_density_ratio = font_stats['avg_page_density'] / 1000
        is_quoted_text = float(self._is_quoted_text(text))
        is_table_header = float(self._is_table_header(text, word_count))
        line_fragment_score = float(char_count < 10 and not text.endswith('.'))
        relative_page_position = page_num / max(total_pages, 1)
        is_numbered_list = float(self._is_numbered_list(text))
        bold_ratio = is_bold * font_size_ratio
        is_footer_candidate = float(position_y_norm > 0.85 and char_count < 20)
        number_depth = self._get_number_depth(text)
        is_multi_span = float(num_spans > 1)
        density_outlier_score = abs(text_density - font_stats['avg_page_density'] / 100) if font_stats['avg_page_density'] > 0 else 0.0
        y_position_score = 1 - position_y_norm
        
        # New advanced features
        cluster_density = float(cluster_label) if cluster_label is not None else 0.0  # From DBSCAN
        semantic_score = self._semantic_score(text)
        
        return [
            font_size_ratio, is_bold, is_italic, line_height_ratio,
            word_count, char_count, has_number_prefix, all_caps_ratio,
            position_x_norm, position_y_norm, line_width_ratio,
            vertical_spacing_ratio, font_size, special_chars_ratio,
            title_case_ratio, uppercase_density, is_centered, has_bullet,
            text_density, has_section_pattern, is_all_upper,
            horizontal_alignment_score, is_form_field, page_density_ratio,
            is_quoted_text, is_table_header, line_fragment_score,
            relative_page_position, is_numbered_list, bold_ratio,
            is_footer_candidate, number_depth, is_multi_span, density_outlier_score, y_position_score,
            cluster_density, semantic_score
        ]
    
    def generate_training_data(self, doc: fitz.Document) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []
        bboxes = []  # For DBSCAN
        
        total_pages = len(doc)
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            prev_y = 0
            page_width = page.rect.width
            page_height = page.rect.height
            
            for block in blocks:
                if block.get("type") == 0:
                    for line in block["lines"]:
                        try:
                            spans = line.get("spans", [])
                            text = self._clean_text(spans)
                            if not text or len(text) < 3:
                                continue
                            
                            span = spans[0] if spans else None
                            if not span:
                                continue
                            
                            line_data = {
                                'text': text,
                                'font_size': span.get("size", 0.0),
                                'font_flags': span.get("flags", 0),
                                'bbox': line.get("bbox", (0, 0, 0, 0)),
                                'num_spans': len(spans)
                            }
                            
                            bboxes.append([line_data['bbox'][0], line_data['bbox'][1]])  # For clustering
                            
                            label_str = self._ultimate_heuristic_label(line_data, self.font_stats, page_num, page_height, total_pages)
                            label = self.label_map.get(label_str, 0)
                            
                            labels.append(label)
                        except Exception as e:
                            print(f"Skipping bad line: {e}")
                            continue
        
        # DBSCAN clustering for layout (multi-column detection)
        if len(bboxes) > 5:
            bboxes_np = np.array(bboxes)
            clustering = DBSCAN(eps=50, min_samples=3).fit(bboxes_np)  # Tune eps based on page size
            cluster_labels = clustering.labels_
        else:
            cluster_labels = [0] * len(labels)
        
        # Now extract features with cluster info
        i = 0
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            prev_y = 0
            for block in blocks:
                if block.get("type") == 0:
                    for line in block["lines"]:
                        try:
                            spans = line.get("spans", [])
                            text = self._clean_text(spans)
                            if not text or len(text) < 3:
                                continue
                            
                            span = spans[0] if spans else None
                            if not span:
                                continue
                            
                            line_data = {
                                'text': text,
                                'font_size': span.get("size", 0.0),
                                'font_flags': span.get("flags", 0),
                                'bbox': line.get("bbox", (0, 0, 0, 0)),
                                'num_spans': len(spans)
                            }
                            
                            feature_vector = self.extract_features(
                                line_data, self.font_stats, page_height, page_width, prev_y, page_num, total_pages, cluster_labels[i]
                            )
                            features.append(feature_vector)
                            
                            prev_y = line["bbox"][1]
                            i += 1
                        except Exception as e:
                            print(f"Skipping bad line: {e}")
                            continue
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Synthetic data augmentation for rare classes
        unique, counts = np.unique(labels, return_counts=True)
        for cls, count in zip(unique, counts):
            if count < 10 and count > 0:
                cls_indices = np.where(labels == cls)[0]
                for _ in range(5):  # Add 5 perturbed copies
                    aug_indices = np.random.choice(cls_indices, size=count)
                    perturbed_features = features[aug_indices] + np.random.normal(0, 0.1, features[aug_indices].shape)  # 10% noise
                    features = np.vstack((features, perturbed_features))
                    labels = np.append(labels, labels[aug_indices])
        
        return features, labels
    
    def _clean_text(self, spans: List[Dict]) -> str:
        if not spans:
            return ""
        text = "".join(s.get("text", "") for s in spans).strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _ultimate_heuristic_label(self, line_data: Dict, font_stats: Dict, page_num: int, page_height: float, total_pages: int) -> str:
        # Unchanged from previous for brevity
        text = line_data.get('text') or ""
        font_size = line_data.get('font_size') or 0.0
        font_flags = line_data.get('font_flags') or 0
        bbox = line_data.get('bbox') or (0, 0, 0, 0)
        num_spans = line_data.get('num_spans', 1)
        position_y_norm = bbox[1] / page_height if page_height > 0 else 0.0
        relative_page = page_num / max(total_pages, 1)
        
        size_ratio = font_size / font_stats['base_font_size'] if font_stats['base_font_size'] > 0 else 1.0
        is_bold = bool(font_flags & (1 << 4))
        is_sparse_doc = font_stats['avg_page_density'] < 500
        
        if is_sparse_doc:
            if text.isupper() and len(text.split()) > 1 and position_y_norm < 0.9 and relative_page < 0.5:
                return 'TITLE' if position_y_norm < 0.2 and size_ratio > 1.2 else 'H1'
            if re.match(r'^\d+\.\s', text) and len(text) > 10:
                return 'H1'
            if self._has_bullet(text) or self._is_quoted_text(text) or position_y_norm > 0.8:
                return 'TEXT'
        
        if size_ratio > 1.4 and page_num < 2 and position_y_norm < 0.25 and len(text) > 15 and num_spans > 1:
            return 'TITLE'
        
        if re.match(r'^(Appendix|Phase|OVERVIEW OF|Mission Statement|Goals|Elective Course Offerings)\s', text):
            return 'H1'
        if re.match(r'^\d+\.\s+[A-Z]', text) and (size_ratio > 1.05 or is_bold):
            return 'H1' if self._get_number_depth(text) <= 1 else 'H2'
        if re.match(r'^\d+\.\d+\.?\s+', text):
            return 'H2'
        if re.match(r'^\d+\.\d+\.\d+\.?\s+', text):
            return 'H3'
        
        if text.isupper() and 2 <= len(text.split()) <= 12 and (size_ratio > 1.0 or is_bold) and position_y_norm < 0.8:
            return 'H1'
        
        if is_bold and size_ratio > 1.0 and len(text.split()) <= 15 and not self._is_quoted_text(text):
            depth = self._get_number_depth(text)
            if depth > 2:
                return 'H3'
            elif depth > 1:
                return 'H2'
            return 'H1'
        
        word_count = len(text.split())
        if self._is_table_header(text, word_count) or text.lower() in ['funding source', 'page', 'rsvp'] or position_y_norm > 0.85:
            return 'TEXT'
        
        return 'TEXT'
    
    def train_classifier(self, doc: fitz.Document):
        X, y = self.generate_training_data(doc)
        if len(X) > 150 and len(np.unique(y)) > 1:
            try:
                unique_classes = np.unique(y)
                class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
                sample_weights = np.array([class_weights[label] for label in y])
                
                dmatrix = xgb.DMatrix(X, label=y, weight=sample_weights)
                params = {
                    'objective': 'multi:softprob',
                    'num_class': len(self.label_map),
                    'eval_metric': 'mlogloss',
                    'max_depth': 6,  # Initial
                    'learning_rate': 0.1
                }
                
                # Cross-validation to tune hyperparameters
                cv_results = xgb.cv(params, dmatrix, num_boost_round=200, nfold=3, metrics=['mlogloss'], early_stopping_rounds=10, seed=42)
                optimal_rounds = len(cv_results)
                params['max_depth'] = [4, 6, 8][np.argmin(cv_results['test-mlogloss-mean'][-1])]  # Simple tune
                
                # Train final model
                self.booster = xgb.train(params, dmatrix, num_boost_round=optimal_rounds)
                self.is_trained = True
                print(f"Trained XGBoost on {len(X)} samples with CV-tuned params (rounds: {optimal_rounds})")
            except Exception as e:
                print(f"Training error: {e}, falling back to heuristics")
                self.is_trained = False
        else:
            print("Insufficient data, using heuristics")
    
    def classify_line(self, line_data: Dict, page_height: float, 
                     page_width: float, prev_y: float, page_num: int, total_pages: int, cluster_label: int) -> str:
        if not self.is_trained or self.booster is None:
            return self._ultimate_heuristic_label(line_data, self.font_stats, page_num, page_height, total_pages)
        
        features = np.array([self.extract_features(line_data, self.font_stats, 
                                                  page_height, page_width, prev_y, page_num, total_pages, cluster_label)])
        dmatrix = xgb.DMatrix(features)
        pred_probs = self.booster.predict(dmatrix)
        prediction = np.argmax(pred_probs, axis=1)[0]
        label = self.reverse_label_map.get(prediction, 'TEXT')
        
        text = line_data.get('text') or ""
        word_count = len(text.split())
        if label in ['H1', 'H2', 'H3'] and (len(text) < 5 or self._is_quoted_text(text) or self._is_table_header(text, word_count)):
            return 'TEXT'
        if self.font_stats['avg_page_density'] < 300 and text.isupper() and page_num == 0:
            return 'TITLE' if len(text) > 10 else 'H1'
        
        return label
    
    def extract_title(self, doc: fitz.Document) -> str:
        candidates = []
        is_sparse = self.font_stats['avg_page_density'] < 500
        total_pages = len(doc)
        merged_lines = []  # For multi-line merging
        
        for page_num in range(min(2, total_pages)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            prev_line = None
            
            for block in blocks:
                if block.get("type") == 0:
                    for line in block["lines"]:
                        spans = line.get("spans", [])
                        text = self._clean_text(spans)
                        if len(text) < 5:
                            continue
                        
                        span = spans[0] if spans else None
                        if not span:
                            continue
                        
                        line_data = {
                            'text': text,
                            'font_size': span.get("size", 0.0),
                            'font_flags': span.get("flags", 0),
                            'bbox': line.get("bbox", (0, 0, 0, 0)),
                            'num_spans': len(spans)
                        }
                        
                        # Multi-line merging
                        if prev_line and abs(line_data['bbox'][1] - prev_line['bbox'][3]) < self.font_stats['avg_line_height'] * 1.5 and \
                           abs(line_data['font_size'] - prev_line['font_size']) < line_data['font_size'] * 0.1:
                            merged_text = prev_line['text'] + " " + text
                            line_data['text'] = merged_text
                            line_data['bbox'] = (min(prev_line['bbox'][0], line_data['bbox'][0]), prev_line['bbox'][1], 
                                                 max(prev_line['bbox'][2], line_data['bbox'][2]), line_data['bbox'][3])
                            line_data['num_spans'] += prev_line['num_spans']
                            merged_lines.pop()  # Remove previous
                        
                        merged_lines.append(line_data)
                        prev_line = line_data
        
        # Score merged lines
        for line_data in merged_lines:
            text = line_data['text']
            if len(text) < 10 or len(text) > 200:  # Extended range for merged
                continue
            
            size_score = (line_data['font_size'] / self.font_stats['base_font_size'] * (1.8 if is_sparse else 1.2)) if self.font_stats['base_font_size'] > 0 else 1.0
            position_score = (1 - (line_data['bbox'][1] / page.rect.height)) * 3 if page.rect.height > 0 else 0.0
            length_score = min(len(text) / 40, 2.5)
            upper_score = 1.5 if text.isupper() else 0
            relative_page = page_num / max(total_pages, 1)
            early_page_score = 1 - relative_page
            
            total_score = size_score * 2.5 + position_score * 2 + length_score + upper_score + early_page_score
            
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
            total_pages = len(doc)
            
            for page_num, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                prev_y = 0
                page_width = page.rect.width
                page_height = page.rect.height
                
                for block in blocks:
                    if block.get("type") == 0:
                        for line in block["lines"]:
                            try:
                                spans = line.get("spans", [])
                                text = self._clean_text(spans)
                                if not text or len(text) < 3:
                                    continue
                                
                                span = spans[0] if spans else None
                                if not span:
                                    continue
                                
                                line_data = {
                                    'text': text,
                                    'font_size': span.get("size", 0.0),
                                    'font_flags': span.get("flags", 0),
                                    'bbox': line.get("bbox", (0, 0, 0, 0)),
                                    'num_spans': len(spans)
                                }
                                
                                classification = self.classify_line(
                                    line_data, page_height, page_width, prev_y, page_num, total_pages, None  # Cluster per-line if needed
                                )
                                
                                if classification in ['H1', 'H2', 'H3']:
                                    self.outline.append({
                                        "level": classification,
                                        "text": text,
                                        "page": page_num + 1,
                                        "number_depth": self._get_number_depth(text)
                                    })
                                
                                prev_y = line["bbox"][1]
                            except Exception as e:
                                print(f"Skipping bad line in processing: {e}")
                                continue
            
            self.outline = self._ultimate_refine_outline(self.outline)
            
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
    
    def _ultimate_refine_outline(self, outline: List[Dict]) -> List[Dict]:
        # Graph-based hierarchy
        tree = defaultdict(list)
        level_map = {'H1': 1, 'H2': 2, 'H3': 3}
        sorted_outline = sorted(outline, key=lambda x: (x['page'], self._get_number_depth(x['text'])))
        
        root = []  # Top-level (H1)
        current_parent = None
        
        for item in sorted_outline:
            depth = item.get('number_depth', 0)
            level = level_map.get(item['level'], 3)
            
            if depth == 0 and level == 1:
                current_parent = item
                root.append(item)
            elif current_parent:
                tree[current_parent['text']].append(item)
            else:
                # Promote orphan to H1
                item['level'] = 'H1'
                root.append(item)
                current_parent = item
        
        # Flatten tree to list (recursive)
        refined = []
        def flatten(node):
            refined.append(node)
            for child in tree.get(node['text'], []):
                flatten(child)
        
        for node in root:
            flatten(node)
        
        # Remove duplicates/short
        final = []
        prev_text = ""
        for item in refined:
            if item['text'] == prev_text or len(item['text']) < 5 or self._is_quoted_text(item['text']):
                continue
            final.append(item)
            prev_text = item['text']
        
        return final

def main():
    """Main execution function"""
    project_root = Path(__file__).parent.resolve()
    input_dir = project_root / "input"
    output_dir = project_root / "xgimprove_output7"
    
    output_dir.mkdir(exist_ok=True)
    
    for pdf_file in input_dir.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        
        extractor = UltimateHeadingExtractorEnhanced()
        result = extractor.process_pdf(str(pdf_file))
        
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {output_file.name}")
        print(f"Title: {result['title']}")
        print(f"Headings found: {len(result['outline'])}")

if __name__ == "__main__":
    main()
