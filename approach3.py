#!/usr/bin/env python3
"""
Adobe Hackathon Round 1A - Approach 3: Advanced Layout Analysis
PDF Heading Detection using pdfplumber with sophisticated layout analysis
"""

import os
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
import re
import statistics
import pdfplumber
from decimal import Decimal

class AdvancedLayoutExtractor:
    def __init__(self):
        self.title = ""
        self.outline = []
        self.font_stats = {}
        self.base_font_size = 0
        self.base_font_name = ""
        
    def analyze_font_characteristics(self, pdf) -> Dict[str, Any]:
        """Comprehensive font analysis across all pages"""
        font_data = []
        size_data = []
        
        for page in pdf.pages:
            chars = page.chars
            
            for char in chars:
                if 'fontname' in char and 'size' in char:
                    font_data.append({
                        'name': char['fontname'],
                        'size': float(char['size']),
                        'text': char.get('text', ''),
                        'bold': 'Bold' in char['fontname'] or 'bold' in char['fontname'].lower(),
                        'italic': 'Italic' in char['fontname'] or 'italic' in char['fontname'].lower()
                    })
                    size_data.append(float(char['size']))
        
        if not size_data:
            return {
                'base_font_size': 12.0,
                'base_font_name': 'Unknown',
                'size_distribution': {},
                'font_names': {},
                'avg_size': 12.0,
                'std_size': 2.0
            }
        
        # Calculate statistics
        size_counter = Counter(size_data)
        font_names = [f['name'] for f in font_data]
        font_counter = Counter(font_names)
        
        # Determine base font (most common)
        base_size = size_counter.most_common(1)[0][0]
        base_name = font_counter.most_common(1)[0][0]
        
        return {
            'base_font_size': base_size,
            'base_font_name': base_name,
            'size_distribution': dict(size_counter),
            'font_names': dict(font_counter),
            'avg_size': statistics.mean(size_data),
            'std_size': statistics.stdev(size_data) if len(size_data) > 1 else 2.0,
            'size_range': (min(size_data), max(size_data))
        }
    
    def extract_line_objects(self, page) -> List[Dict]:
        """Extract line-level objects with detailed attributes"""
        lines = []
        
        # Get words with font information
        words = page.extract_words(extra_attrs=['fontname', 'size'])
        
        if not words:
            return lines
        
        # Group words into lines based on vertical position
        current_line = []
        current_top = None
        tolerance = 3  # Vertical tolerance for line grouping
        
        for word in sorted(words, key=lambda w: (w['top'], w['x0'])):
            if current_top is None or abs(word['top'] - current_top) <= tolerance:
                current_line.append(word)
                current_top = word['top'] if current_top is None else current_top
            else:
                if current_line:
                    lines.append(self._process_line(current_line))
                current_line = [word]
                current_top = word['top']
        
        # Process the last line
        if current_line:
            lines.append(self._process_line(current_line))
        
        return lines
    
    def _process_line(self, words: List[Dict]) -> Dict:
        """Process a line of words into a unified line object"""
        if not words:
            return {}
        
        # Sort words by horizontal position
        words = sorted(words, key=lambda w: w['x0'])
        
        # Combine text
        text = ' '.join(word['text'] for word in words)
        
        # Calculate bounding box
        x0 = min(word['x0'] for word in words)
        x1 = max(word['x1'] for word in words)
        top = min(word['top'] for word in words)
        bottom = max(word['bottom'] for word in words)
        
        # Determine dominant font characteristics
        font_sizes = [word['size'] for word in words if 'size' in word]
        font_names = [word['fontname'] for word in words if 'fontname' in word]
        
        # Weight by character count
        char_weighted_size = 0
        total_chars = 0
        
        for word in words:
            if 'size' in word:
                char_count = len(word['text'])
                char_weighted_size += word['size'] * char_count
                total_chars += char_count
        
        avg_font_size = char_weighted_size / max(total_chars, 1)
        
        # Determine if bold/italic
        is_bold = any('Bold' in name or 'bold' in name.lower() 
                     for name in font_names if name)
        is_italic = any('Italic' in name or 'italic' in name.lower() 
                       for name in font_names if name)
        
        return {
            'text': text.strip(),
            'x0': x0, 'x1': x1, 'top': top, 'bottom': bottom,
            'width': x1 - x0,
            'height': bottom - top,
            'font_size': avg_font_size,
            'font_names': font_names,
            'is_bold': is_bold,
            'is_italic': is_italic,
            'word_count': len(text.split()),
            'char_count': len(text.strip())
        }
    
    def calculate_line_features(self, line: Dict, page_width: float, 
                               page_height: float, prev_y: float) -> Dict:
        """Calculate comprehensive features for a line"""
        if not line or not line.get('text'):
            return {}
        
        text = line['text']
        font_size = line['font_size']
        
        # Basic measurements
        size_ratio = font_size / self.base_font_size if self.base_font_size > 0 else 1
        position_x_norm = line['x0'] / max(page_width, 1)
        position_y_norm = line['top'] / max(page_height, 1)
        width_ratio = line['width'] / max(page_width, 1)
        
        # Vertical spacing
        vertical_spacing = abs(line['top'] - prev_y) if prev_y > 0 else 0
        avg_line_height = self.font_stats.get('avg_size', 12) * 1.2
        spacing_ratio = vertical_spacing / avg_line_height
        
        # Text analysis
        word_count = line['word_count']
        char_count = line['char_count']
        
        # Pattern matching
        has_number_prefix = bool(re.match(r'^\d+\.?\s+', text))
        has_section_number = bool(re.match(r'^\d+\.\d+', text))
        has_subsection_number = bool(re.match(r'^\d+\.\d+\.\d+', text))
        
        # Case analysis
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
        title_case_words = sum(1 for word in text.split() 
                              if word and word[0].isupper())
        title_case_ratio = title_case_words / max(word_count, 1)
        
        # Special patterns
        is_all_caps = text.isupper() and len(text) > 2
        starts_with_capital = text and text[0].isupper()
        
        return {
            'size_ratio': size_ratio,
            'position_x_norm': position_x_norm,
            'position_y_norm': position_y_norm,
            'width_ratio': width_ratio,
            'spacing_ratio': spacing_ratio,
            'word_count': word_count,
            'char_count': char_count,
            'has_number_prefix': has_number_prefix,
            'has_section_number': has_section_number,
            'has_subsection_number': has_subsection_number,
            'uppercase_ratio': uppercase_ratio,
            'title_case_ratio': title_case_ratio,
            'is_all_caps': is_all_caps,
            'starts_with_capital': starts_with_capital,
            'is_bold': line['is_bold'],
            'is_italic': line['is_italic']
        }
    
    def classify_heading_advanced(self, features: Dict, text: str) -> Optional[str]:
        """Advanced heading classification using multiple criteria"""
        
        # Skip obvious non-headings
        if (features['char_count'] < 3 or features['char_count'] > 200 or
            re.match(r'^\d+$', text.strip()) or
            re.match(r'^page\s+\d+', text.lower())):
            return None
        
        # Title detection
        if (features['size_ratio'] > 1.7 or
            (features['size_ratio'] > 1.4 and features['position_y_norm'] < 0.3 and
             features['word_count'] <= 12)):
            return 'TITLE'
        
        # H1 detection
        if (features['has_section_number'] or
            features['size_ratio'] > 1.3 or
            (features['is_bold'] and features['size_ratio'] > 1.15) or
            (features['is_all_caps'] and 3 <= features['word_count'] <= 8) or
            (features['spacing_ratio'] > 2 and features['size_ratio'] > 1.1)):
            return 'H1'
        
        # H2 detection
        if (features['has_subsection_number'] or
            features['size_ratio'] > 1.15 or
            (features['is_bold'] and features['size_ratio'] > 1.05) or
            (features['spacing_ratio'] > 1.5 and features['size_ratio'] > 1.02)):
            return 'H2'
        
        # H3 detection
        if (features['is_bold'] and features['size_ratio'] >= 0.98 or
            features['size_ratio'] > 1.08 or
            (features['title_case_ratio'] > 0.7 and features['size_ratio'] > 1.02) or
            (features['has_number_prefix'] and features['size_ratio'] > 1.0)):
            return 'H3'
        
        return None
    
    def extract_title_advanced(self, pdf) -> str:
        """Advanced title extraction using layout analysis"""
        candidates = []
        
        # Check first two pages
        for page_num, page in enumerate(pdf.pages[:2]):
            lines = self.extract_line_objects(page)
            
            for line in lines:
                if not line or not line.get('text'):
                    continue
                
                text = line['text']
                if len(text) < 5 or len(text) > 150:
                    continue
                
                features = self.calculate_line_features(
                    line, page.width, page.height, 0
                )
                
                # Title scoring
                size_score = features['size_ratio'] * 3
                position_score = (1 - features['position_y_norm']) * 2
                length_score = max(0, 2 - abs(features['word_count'] - 8) * 0.2)
                bold_score = 1 if features['is_bold'] else 0
                caps_score = features['uppercase_ratio']
                
                total_score = (size_score + position_score + length_score + 
                              bold_score + caps_score)
                
                candidates.append({
                    'text': text,
                    'score': total_score,
                    'page': page_num,
                    'features': features
                })
        
        if candidates:
            # Filter candidates by minimum score threshold
            good_candidates = [c for c in candidates if c['score'] >= 4.0]
            if good_candidates:
                best = max(good_candidates, key=lambda x: x['score'])
                return best['text']
            else:
                # Fallback to highest scoring
                best = max(candidates, key=lambda x: x['score'])
                return best['text']
        
        return "Untitled Document"
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Main processing function"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Analyze font characteristics
                self.font_stats = self.analyze_font_characteristics(pdf)
                self.base_font_size = self.font_stats['base_font_size']
                self.base_font_name = self.font_stats['base_font_name']
                
                print(f"Base font: {self.base_font_name}, size: {self.base_font_size}")
                
                # Extract title
                self.title = self.extract_title_advanced(pdf)
                
                # Extract headings
                self.outline = []
                
                for page_num, page in enumerate(pdf.pages):
                    lines = self.extract_line_objects(page)
                    prev_y = 0
                    
                    for line in lines:
                        if not line or not line.get('text'):
                            continue
                        
                        text = line['text']
                        features = self.calculate_line_features(
                            line, page.width, page.height, prev_y
                        )
                        
                        # Classify the line
                        classification = self.classify_heading_advanced(features, text)
                        
                        if classification and classification != 'TITLE':
                            self.outline.append({
                                "level": classification,
                                "text": text,
                                "page": page_num + 1
                            })
                        
                        prev_y = line['top']
                
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
    output_dir = project_root / "output3"
    
    output_dir.mkdir(exist_ok=True)
    
    for pdf_file in input_dir.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        
        extractor = AdvancedLayoutExtractor()
        result = extractor.process_pdf(str(pdf_file))
        
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {output_file.name}")
        print(f"Title: {result['title']}")
        print(f"Headings found: {len(result['outline'])}")

if __name__ == "__main__":
    main()
