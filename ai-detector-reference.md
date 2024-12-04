# AI Text Detector Project Reference

## Core Concept
A web application that analyzes text to determine likelihood of AI generation, providing a score out of 10 and detailed metrics.

## Key Features
1. Score range 0-10 with thresholds:
   - <5.0: Mostly Human
   - <7.5: AI-Assisted 
   - ≥7.5: AI-Written

2. Five Key Metrics:
   - Marker Density: Presence of common AI writing patterns/phrases
   - Sentence Uniformity: Similarity in sentence lengths
   - Vocabulary Diversity: Variety in word choice
   - Flow Naturalness: Sentence transitions and beginnings
   - Complexity Variance: Variation in sentence complexity (using CV)

3. Context-Specific Analysis:
   - General Text
   - Email
   - Poetry

## Important Design Decisions

### Short Text Handling (< 100 words)
- Modified weights for more reliable analysis:
  - Marker Density: 20%
  - Vocabulary Diversity: 40%
  - Flow Naturalness: 40%
  - Others: 0%
- Warning displayed to user
- Maintains analysis but with adjusted confidence

### Complexity Variance Calculation
- Uses Coefficient of Variation (CV) instead of raw variance
- CV = standard deviation / mean
- Normalizes for different baseline complexity levels
- Score = 1 - min(1.0, CV)

### Weight System
Default weights for general text:
- Marker Density: 25%
- Sentence Uniformity: 20%
- Vocabulary Diversity: 20%
- Flow Naturalness: 15%
- Complexity Variance: 20%

### AI Markers Categories
1. Repetitive Phrases
2. Academic Hedging
3. Formulaic Transitions
4. AI Certainty Patterns
5. List Introductions

## Development Notes
- Score calculation includes rounding to nearest 0.5
- Frontend displays real-time metric values and weights
- Confidence score reflects consistency across metrics
- Example texts used in development: emails, poetry, academic writing

## Future Enhancement Possibilities
1. Additional context types (academic, technical, creative)
2. Machine learning integration
3. Enhanced pattern detection
4. Style-specific analysis
5. Batch processing capability
6. Historical analysis and trending
7. API integration options

## Known Limitations
1. Minimum text length for reliable analysis (100 words)
2. Language limited to English
3. Context-specific patterns may need refinement
4. Current system uses predefined patterns rather than ML