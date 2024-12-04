import unittest
from app import AITextDetector

class TestAITextDetector(unittest.TestCase):
    def setUp(self):
        """Initialize the detector before each test"""
        self.detector = AITextDetector()
        
        # Sample texts for testing
        self.short_text = "This is a very short text that should trigger the short text warning."
        
        self.human_text = """
            I wrote this email quickly to let you know I'll be late. Traffic is terrible 
            today! Can't believe how backed up the highway is. See you soon, hopefully 
            within the next 30 minutes or so. Sorry about this!
        """
        
        self.ai_text = """
            It is important to note that artificial intelligence has become increasingly 
            prevalent in modern society. Furthermore, there are several key factors that 
            contribute to this trend. Additionally, it should be mentioned that the impact 
            on various sectors has been significant. Moreover, it is worth noting that 
            technological advancement continues to accelerate. It is crucial to understand 
            that this transformation is ongoing. Generally speaking, there are many aspects 
            to consider. First and foremost, it can be observed that AI plays a crucial role. 
            Last but not least, it goes without saying that future developments will be significant.
            In conclusion, it should be noted that these patterns demonstrate clear AI markers.
        """
        
        self.mixed_text = """
            Hey there! I wanted to share some thoughts about artificial intelligence. 
            It is important to note that the technology is advancing quickly. Machine 
            learning and neural networks have become increasingly sophisticated.
            I'm really excited about it, but also a bit worried sometimes. What do you 
            think about all this stuff? Furthermore, it should be mentioned that we need 
            to consider the ethical implications. Let me know your thoughts when you get a chance!
        """
        
        # Long sample text for context testing (>100 words)
        self.long_text = """
            The use of artificial intelligence in modern technology has revolutionized how we approach problem-solving 
            and data analysis. Machine learning algorithms have become increasingly sophisticated, enabling systems 
            to recognize patterns and make decisions with remarkable accuracy. Natural language processing has 
            advanced significantly, allowing computers to understand and generate human-like text. Computer vision 
            systems can now identify objects and faces with exceptional precision. The integration of AI into 
            everyday applications has transformed user experiences and streamlined many processes. However, 
            these developments also raise important ethical considerations about privacy, bias, and the role of 
            AI in decision-making. As we continue to advance in this field, it becomes crucial to balance innovation 
            with responsible development practices. The future of AI holds both exciting possibilities and 
            significant challenges that we must carefully navigate.
        """
    
    def test_short_text_handling(self):
        """Test the handling of texts shorter than minimum length"""
        result = self.detector.analyze(self.short_text)
        
        # Check if warning is present
        self.assertIsNotNone(result['warning'])
        self.assertTrue('short text' in result['warning'].lower())
        
        # Compare with detector's short text weights
        self.assertEqual(result['weights'], self.detector.short_text_weights)
    
    def test_human_text_detection(self):
        """Test detection of likely human-written text"""
        result = self.detector.analyze(self.human_text)
        
        # Score should be in the human range (<5.0)
        self.assertLess(result['scores']['raw_total'], 5.0)
        self.assertEqual(result['classification'], 'MOSTLY HUMAN')
        
        # Check specific metrics
        metrics = result['details']
        self.assertLess(metrics['marker_density'], 0.3)  # Few AI markers
        self.assertGreater(metrics['complexity_variance'], 0.3)  # More variance
    
    def test_ai_text_detection(self):
        """Test detection of likely AI-written text"""
        result = self.detector.analyze(self.ai_text)
        
        # Debug output
        print(f"\nAI Text Analysis Scores:")
        print(f"Raw Total: {result['scores']['raw_total']}")
        print(f"Classification: {result['classification']}")
        print(f"Details: {result['details']}")
        
        # Score should indicate AI text
        self.assertEqual(result['classification'], 'AI-WRITTEN')
        
        # Check specific metrics
        metrics = result['details']
        self.assertGreater(metrics['marker_density'], 0.4)  # Many AI markers
        self.assertGreater(metrics['sentence_uniformity'], 0.4)  # More uniform
    
    def test_mixed_text_detection(self):
        """Test detection of likely AI-assisted text"""
        result = self.detector.analyze(self.mixed_text)
        
        # Debug output
        print(f"\nMixed Text Analysis Scores:")
        print(f"Raw Total: {result['scores']['raw_total']}")
        print(f"Classification: {result['classification']}")
        print(f"Details: {result['details']}")
        
        # Score should be in the assisted range
        self.assertEqual(result['classification'], 'AI-ASSISTED')
    
    def test_context_specific_weights(self):
        """Test different context-specific weight configurations"""
        # Print word count for debugging
        print(f"\nWord count in test text: {len(self.long_text.split())}")
        
        # Test poem context
        poem_result = self.detector.analyze(self.long_text, context_type='poem')
        print(f"Poem context weights: {poem_result['weights']}")
        self.assertEqual(poem_result['weights'], self.detector.context_settings['poem']['weights'])
        
        # Test email context
        email_result = self.detector.analyze(self.long_text, context_type='email')
        print(f"Email context weights: {email_result['weights']}")
        self.assertEqual(email_result['weights'], self.detector.context_settings['email']['weights'])
        
        # Test general context
        general_result = self.detector.analyze(self.long_text)
        print(f"General context weights: {general_result['weights']}")
        self.assertEqual(general_result['weights'], self.detector.context_settings['general']['weights'])
    
    def test_marker_detection(self):
        """Test detection of specific AI markers"""
        test_text = "It is important to note that this text has markers. Furthermore, it should be mentioned that there are several key factors."
        result = self.detector.analyze(test_text)
        
        # Should detect multiple markers
        self.assertGreater(result['details']['marker_density'], 0.5)
    
    def test_score_normalization(self):
        """Test that scores are properly normalized"""
        result = self.detector.analyze(self.long_text)
        
        # Check that all metrics are between 0 and 1
        for metric_value in result['details'].values():
            self.assertGreaterEqual(metric_value, 0.0)
            self.assertLessEqual(metric_value, 1.0)
        
        # Check total score is between 0 and 10
        self.assertGreaterEqual(result['scores']['raw_total'], 0.0)
        self.assertLessEqual(result['scores']['raw_total'], 10.0)
    
    def test_empty_input(self):
        """Test handling of empty input"""
        result = self.detector.analyze("")
        
        # All metrics should be zero
        for metric_value in result['details'].values():
            self.assertEqual(metric_value, 0.0)
        
        # Should have warning
        self.assertIsNotNone(result['warning'])
    
    def test_invalid_context(self):
        """Test handling of invalid context type"""
        result = self.detector.analyze(self.long_text, context_type='invalid_type')
        
        # Should use general weights for invalid context
        print(f"\nInvalid context test results:")
        print(f"Context: {result['context']}")
        print(f"Weights: {result['weights']}")
        print(f"Expected weights: {self.detector.context_settings['general']['weights']}")
        
        # Check weights match general context
        self.assertEqual(result['weights'], self.detector.context_settings['general']['weights'])
        
        # Context type should remain as provided
        self.assertEqual(result['context'], 'invalid_type')

if __name__ == '__main__':
    unittest.main(verbosity=2)