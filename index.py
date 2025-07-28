#!/usr/bin/env python3
"""
Adobe Hackathon Round 1A - Approach 2: Hybrid ML Feature Classification
PDF Heading Detection using lightweight machine learning
"""

import os
import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import re
import math
import pickle
import pymupdf as fitz
import numpy as np

# Simple Decision Tree implementation to avoid external dependencies
class SimpleDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_names = None
    
    def _entropy(self, y):
        """Calculate entropy of a dataset"""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-15))
        return entropy
    
    def _information_gain(self, X, y, feature_idx, threshold):
        """Calculate information gain for a split"""
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        n = len(y)
        left_entropy = self._entropy(y[left_mask])
        right_entropy = self._entropy(y[right_mask])
        
        weighted_entropy = (np.sum(left_mask) / n) * left_entropy + \
                          (np.sum(right_mask) / n) * right_entropy
        
        information_gain = self._entropy(y) - weighted_entropy
        return information_gain
    
    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or n_classes == 1 or 
            n_samples < self.min_samples_split):
            # Return leaf node
            most_common_class = Counter(y).most_common(1)[0][0]
            return {'class': most_common_class, 'is_leaf': True}
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        if best_gain == 0:
            # No good split found, return leaf
            most_common_class = Counter(y).most_common(1)[0][0]
            return {'class': most_common_class, 'is_leaf': True}
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree,
            'is_leaf': False
        }
    
    def fit(self, X, y, feature_names=None):
        """Train the decision tree"""
        self.feature_names = feature_names
        self.tree = self._build_tree(X, y)
    
    def _predict_sample(self, x, node):
        """Predict a single sample"""
        if node['is_leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X):
        """Predict multiple samples"""
        if self.tree is None:
            raise ValueError("Tree not trained. Call fit() first.")
        
        predictions = []
        for x in X:
            pred = self._predict_sample(x, self.tree)
            predictions.append(pred)
        
        return np.array(predictions)

class MLHeadingExtractor:
    def __init__(self):
        self.title = ""
        self.outline = []
        self.font_stats = {}
        self.base_font_size = 0
        self.classifier = SimpleDecisionTree(max_depth=8, min_samples_split=5)
        self.is_trained = False
        
        # Feature names for interpretability
        self.feature_names = [
            'font_size_ratio', 'is_bold', 'is_italic', 'line_height_ratio',
            'word_count', 'char_count', 'has_number_prefix', 'all_caps_ratio',
            'position_x_norm', 'position_y_norm', 'line_width_ratio',
            'vertical_spacing_ratio', 'font_size_absolute', 'special_chars_ratio',
            'title_case_ratio'
        ]
    
    def extract_font_statistics(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract comprehensive font statistics"""
        font_sizes = []
        font_names = []
        line_heights = []
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
                            font_names.append(span["font"])
                        
                        # Calculate line height
                        bbox = line["bbox"]
                        line_heights.append(bbox[3] - bbox[1])
        
        return {
            'font_sizes': Counter(font_sizes),
            'font_names': Counter(font_names),
            'avg_font_size': np.mean(font_sizes) if font_sizes else 12,
            'base_font_size': Counter(font_sizes).most_common(1)[0][0] if font_sizes else 12,
            'avg_line_height': np.mean(line_heights) if line_heights else 15,
            'font_size_std': np.std(font_sizes) if font_sizes else 2
        }
    
    def extract_features(self, line_data: Dict, font_stats: Dict, 
                        page_height: float, prev_y: float) -> List[float]:
        """Extract features for ML classification"""
        text = line_data['text']
        font_size = line_data['font_size']
        font_flags = line_data['font_flags']
        bbox = line_data['bbox']
        
        # Basic ratios
        font_size_ratio = font_size / font_stats['base_font_size']
        line_height = bbox[3] - bbox[1]
        line_height_ratio = line_height / font_stats['avg_line_height']
        line_width = bbox[2] - bbox[0]
        line_width_ratio = line_width / 500  # Normalize to typical page width
        
        # Text characteristics
        word_count = len(text.split())
        char_count = len(text)
        
        # Font flags
        is_bold = bool(font_flags & (1 << 4))
        is_italic = bool(font_flags & (1 << 1))
        
        # Text patterns
        has_number_prefix = bool(re.match(r'^\d+\.?\s+', text))
        all_caps_count = sum(1 for c in text if c.isupper())
        all_caps_ratio = all_caps_count / max(char_count, 1)
        
        title_case_count = sum(1 for word in text.split() 
                              if word and word[0].isupper())
        title_case_ratio = title_case_count / max(word_count, 1)
        
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_chars_ratio = special_chars / max(char_count, 1)
        
        # Position features
        position_x_norm = bbox[0] / 500  # Normalize position
        position_y_norm = bbox[1] / max(page_height, 1)
        
        # Vertical spacing
        vertical_spacing = abs(bbox[1] - prev_y) if prev_y > 0 else 0
        vertical_spacing_ratio = vertical_spacing / font_stats['avg_line_height']
        
        return [
            font_size_ratio, float(is_bold), float(is_italic), line_height_ratio,
            word_count, char_count, float(has_number_prefix), all_caps_ratio,
            position_x_norm, position_y_norm, line_width_ratio,
            vertical_spacing_ratio, font_size, special_chars_ratio,
            title_case_ratio
        ]
    
    def generate_training_data(self, doc: fitz.Document) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data from document analysis"""
        features = []
        labels = []
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            prev_y = 0
            
            for block in blocks:
                if block.get("type") == 0:
                    for line in block["lines"]:
                        if not line["spans"]:
                            continue
                        
                        span = line["spans"][0]
                        text = "".join(s["text"] for s in line["spans"]).strip()
                        
                        if len(text) < 3:
                            continue
                        
                        line_data = {
                            'text': text,
                            'font_size': span["size"],
                            'font_flags': span["flags"],
                            'bbox': line["bbox"]
                        }
                        
                        # Extract features
                        feature_vector = self.extract_features(
                            line_data, self.font_stats, page.rect.height, prev_y
                        )
                        
                        # Generate label using heuristics
                        label = self._heuristic_label(line_data, self.font_stats)
                        
                        features.append(feature_vector)
                        labels.append(label)
                        
                        prev_y = line["bbox"][1]
        
        return np.array(features), np.array(labels)
    
    def _heuristic_label(self, line_data: Dict, font_stats: Dict) -> str:
        """Generate training labels using rule-based heuristics"""
        text = line_data['text']
        font_size = line_data['font_size']
        font_flags = line_data['font_flags']
        
        size_ratio = font_size / font_stats['base_font_size']
        is_bold = bool(font_flags & (1 << 4))
        
        # Title detection
        if (size_ratio > 1.6 and len(text.split()) <= 12 and 
            not re.match(r'^\d+\.', text)):
            return 'TITLE'
        
        # Heading patterns
        if re.match(r'^\d+\.\s+[A-Z]', text) and (size_ratio > 1.2 or is_bold):
            return 'H1'
        
        if re.match(r'^\d+\.\d+\.?\s+', text) and (size_ratio > 1.1 or is_bold):
            return 'H2'
        
        if (re.match(r'^\d+\.\d+\.\d+\.?\s+', text) or 
            (is_bold and size_ratio > 1.05)):
            return 'H3'
        
        # All caps short text
        if (text.isupper() and 3 <= len(text.split()) <= 8 and 
            (size_ratio > 1.1 or is_bold)):
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
        """Train the ML classifier on the document"""
        X, y = self.generate_training_data(doc)
        
        if len(X) > 0:
            self.classifier.fit(X, y, self.feature_names)
            self.is_trained = True
            print(f"Trained on {len(X)} samples")
    
    def classify_line(self, line_data: Dict, page_height: float, 
                     prev_y: float) -> str:
        """Classify a line using the trained model"""
        if not self.is_trained:
            return self._heuristic_label(line_data, self.font_stats)
        
        features = self.extract_features(line_data, self.font_stats, 
                                       page_height, prev_y)
        prediction = self.classifier.predict([features])[0]
        return prediction
    
    def extract_title(self, doc: fitz.Document) -> str:
        """Extract document title"""
        candidates = []
        
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
                        
                        # Title scoring
                        size_score = font_size / self.font_stats['base_font_size']
                        position_score = 1.0 - (bbox[1] / page.rect.height)
                        length_score = max(0, 1.0 - abs(len(text.split()) - 7) * 0.1)
                        
                        total_score = size_score * 2 + position_score + length_score
                        
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
        """Main processing function"""
        try:
            doc = fitz.open(pdf_path)
            
            # Extract font statistics
            self.font_stats = self.extract_font_statistics(doc)
            
            # Train classifier on this document
            self.train_classifier(doc)
            
            # Extract title
            self.title = self.extract_title(doc)
            
            # Extract headings
            self.outline = []
            
            for page_num, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                prev_y = 0
                
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
                            
                            # Classify the line
                            classification = self.classify_line(
                                line_data, page.rect.height, prev_y
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
    print("hoo")
    project_root = Path(__file__).parent.resolve()
    input_dir = project_root / "input"
    output_dir = project_root / "output2"
    
    output_dir.mkdir(exist_ok=True)
    
    for pdf_file in input_dir.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        
        extractor = MLHeadingExtractor()
        result = extractor.process_pdf(str(pdf_file))
        
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {output_file.name}")
        print(f"Title: {result['title']}")
        print(f"Headings found: {len(result['outline'])}")

if __name__ == "__main__":
    main()
