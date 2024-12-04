from flask import Flask, render_template, request, jsonify
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
import re
import os

class AITextDetector:
    def __init__(self):
        self.ai_markers = {
            'repetitive_phrases': [
                'it is important to note',
                'it should be mentioned',
                'it is worth noting',
                'generally speaking',
                'in conclusion',
                'first and foremost',
                'last but not least',
                'it goes without saying',
                'it is crucial to understand',
                'it is essential to consider',
                'it is worth mentioning',
                'it should be noted that',
                'as mentioned earlier',
                'as discussed above',
                'it can be observed that'
            ],
            'academic_hedging': [
                'it could be argued',
                'it seems that',
                'it appears that',
                'arguably',
                'presumably',
                'potentially',
                'conceivably',
                'one might consider',
                'this suggests that',
                'this indicates that',
                'this implies that',
                'in most cases',
                'generally speaking',
                'typically',
                'tends to be'
            ],
            'formulaic_transitions': [
                'furthermore',
                'additionally',
                'moreover',
                'consequently',
                'therefore',
                'thus',
                'hence',
                'in addition to this',
                'as a result',
                'for this reason',
                'with this in mind',
                'in light of',
                'on the other hand',
                'in contrast to this',
                'similarly'
            ],
            'ai_certainty_patterns': [
                'there are several',
                'there are many',
                'there are various',
                'can be categorized',
                'can be classified',
                'plays a crucial role',
                'plays an important role',
                'is a key factor',
                'is an essential aspect',
                'has become increasingly',
                'it is clear that',
                'it is evident that',
                'it is obvious that'
            ],
            'list_introductions': [
                'some examples include',
                'these include',
                'such as',
                'for example',
                'for instance',
                'namely',
                'specifically',
                'to illustrate',
                'in particular'
            ]
        }
        
        self.context_settings = {
            'email': {
                'weights': {
                    'marker_density': 0.2,
                    'sentence_uniformity': 0.15,
                    'vocabulary_diversity': 0.3,
                    'flow_naturalness': 0.2,
                    'complexity_variance': 0.15
                },
                'thresholds': {
                    'human': 0.50,
                    'assisted': 0.75
                }
            },
            'poem': {
                'weights': {
                    'marker_density': 0.1,
                    'sentence_uniformity': 0.3,
                    'vocabulary_diversity': 0.3,
                    'flow_naturalness': 0.2,
                    'complexity_variance': 0.1
                },
                'thresholds': {
                    'human': 0.50,
                    'assisted': 0.75
                }
            },
            'general': {
                'weights': {
                    'marker_density': 0.25,
                    'sentence_uniformity': 0.2,
                    'vocabulary_diversity': 0.2,
                    'flow_naturalness': 0.15,
                    'complexity_variance': 0.2
                },
                'thresholds': {
                    'human': 0.50,
                    'assisted': 0.75
                }
            }
        }
        
        # Define short text weights
        self.short_text_weights = {
            'marker_density': 0.2,
            'sentence_uniformity': 0.0,
            'vocabulary_diversity': 0.4,
            'flow_naturalness': 0.4,
            'complexity_variance': 0.0
        }
        
    def analyze(self, text: str, context_type: str = 'general') -> Dict:
        context = self.context_settings.get(context_type, self.context_settings['general']).copy()
        
        cleaned_text = text.lower().strip()
        sentences = [s.strip() for s in re.split(r'[.!?]+', cleaned_text) if s.strip()]
        words = re.findall(r'\b\w+\b', cleaned_text)
        
        # Calculate length factors
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Define minimum threshold for reliable analysis
        MIN_WORDS = 100
        
        # Adjust weights for short texts
        if word_count < MIN_WORDS:
            context['weights'] = self.short_text_weights.copy()
            warning = f"Analysis may be less reliable due to short text length ({word_count} words)"
        else:
            warning = None
            
        metrics = {
            'marker_density': self._calculate_marker_density(cleaned_text),
            'sentence_uniformity': self._analyze_sentence_uniformity(sentences),
            'vocabulary_diversity': self._calculate_vocabulary_diversity(words),
            'flow_naturalness': self._analyze_flow(sentences),
            'complexity_variance': self._analyze_complexity(sentences)
        }
        
        # Calculate scores without length penalty
        scores = self._calculate_scores(metrics, context['weights'])
        
        # Build response with all metrics
        response = {
            'classification': self._classify_text(scores['total'], context['thresholds']),
            'confidence': scores['confidence'],
            'scores': scores,
            'details': metrics,
            'context': context_type,
            'text_stats': {
                'words': word_count,
                'sentences': sentence_count
            },
            'warning': warning,
            'weights': context['weights']  # Include weights for transparency
        }
        
        return response

    def _calculate_marker_density(self, text: str) -> float:
        total_markers = 0
        for category in self.ai_markers.values():
            for phrase in category:
                total_markers += text.count(phrase)
        words = len(text.split())
        return min(1.0, total_markers / max(1, words/50))

    def _analyze_sentence_uniformity(self, sentences: List[str]) -> float:
        if not sentences:
            return 0.0
        lengths = [len(s.split()) for s in sentences]
        variance = np.var(lengths) if len(lengths) > 1 else 0
        return 1 - min(1.0, variance / 100)

    def _calculate_vocabulary_diversity(self, words: List[str]) -> float:
        if not words:
            return 0.0
        unique_words = len(set(words))
        total_words = len(words)
        return 1 - (unique_words / total_words)

    def _analyze_flow(self, sentences: List[str]) -> float:
        if len(sentences) < 2:
            return 0.0
        starts = [s.split()[0] if s.split() else '' for s in sentences]
        start_repetition = len(starts) - len(set(starts))
        return min(1.0, start_repetition / max(1, len(sentences)))

    def _analyze_complexity(self, sentences: List[str]) -> float:
        if not sentences:
            return 0.0
        complexities = []
        for sentence in sentences:
            words = sentence.split()
            avg_word_length = sum(len(word) for word in words) / max(1, len(words))
            complexities.append(avg_word_length)
            
        if len(complexities) > 1:
            mean = np.mean(complexities)
            std = np.std(complexities)
            cv = std / mean if mean > 0 else 0
            return 1 - min(1.0, cv)  # Normalized by mean
        return 0.0

    def _calculate_scores(self, metrics: Dict, weights: Dict) -> Dict:
        # Calculate raw total score
        raw_total = sum(metrics[key] * weights[key] for key in weights)
        
        # Transform to 0-10 scale
        scaled_total = (raw_total / 0.1) * 2
        
        # Round to nearest 0.5
        rounded_total = round(scaled_total * 2) / 2
        
        # Cap at 10
        final_total = min(10, rounded_total)
        
        # Calculate confidence based on metric consistency
        mean_metric = sum(metrics.values()) / len(metrics)
        variations = sum((x - mean_metric) ** 2 for x in metrics.values())
        metric_variance = variations / len(metrics)
        confidence = 1 - min(1.0, metric_variance * 2)
        
        return {
            'total': final_total / 10,  # Keep as 0-1 for internal use
            'raw_total': final_total,   # 0-10 scale for display
            'confidence': confidence
        }

    def _classify_text(self, score: float, thresholds: Dict) -> str:
        # Convert score to 0-10 scale for threshold comparison
        score_on_ten = score * 10
        
        if score_on_ten < 5.0:
            return "MOSTLY HUMAN"
        elif score_on_ten < 7.5:
            return "AI-ASSISTED"
        else:
            return "AI-WRITTEN"

app = Flask(__name__)
detector = AITextDetector()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text', '')
    context_type = request.json.get('context_type', 'general')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    results = detector.analyze(text, context_type)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)