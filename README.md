# MITRA – Emotion-Aware Conversational AI for Student Mental Well-Being

MITRA is a Deep Learning–based emotion-aware conversational AI framework designed to detect and interpret student emotional states from textual input. The system integrates contextual and local feature extraction using a hybrid BiLSTM–CNN architecture to enhance emotional understanding beyond keyword-based approaches.
MITRA aims to support academic environments by enabling emotionally intelligent chatbot systems that can assess, monitor, and respond to student mental well-being.

# Problem Statement

Traditional educational chatbots operate using rule-based or keyword-driven systems that lack emotional awareness. However, students frequently express stress, anxiety, frustration, and mixed emotions in subtle and context-dependent ways.
MITRA addresses this limitation by incorporating deep learning–based emotion recognition to:
1. Detect emotional intent in student messages
2. Classify emotional states accurately
3. Enable supportive and context-aware responses

# Dataset Information

The system was evaluated using publicly available Kaggle datasets named Emotions Dataset for NLP and Emotion Dataset for Emotion Recognition:
* Dataset Size: ~32,000 text samples
* Emotion Classes (6):
1. Joy
2. Sadness
3. Anger
4. Fear
5. Stress
6. Love

The dataset represents diverse emotional expressions commonly observed in student communication.

# Model Architecture
MITRA uses a hybrid deep learning pipeline:
1. Text Input
2. Tokenization and Padding
3. Word Embedding Layer
4. BiLSTM Layer
* Captures contextual dependencies
6. CNN Layer
* Extracts local semantic features
6. Dense + Softmax Layer
7. Emotion Classification Output

This architecture enables both sequential context modeling and localized feature learning for improved emotion detection.

# Performance Metrics
Best performing model: BiLSTM–CNN
* Accuracy: 94.7%
* Precision: 89.8%
* Recall: 75.75%
* Baseline Model: Linear SVM

Results demonstrate that deep learning–based sequence modeling significantly outperforms conventional machine learning approaches for emotion recognition tasks.

# Sample Inference
Example predictions:

| Input Text                       | Predicted Emotion          |
| -------------------------------- | -------------------------- |
| "I’m really happy about my GPA." | Joy                        |
| "I’m so stressed about exams."   | Stress                     |
| "I feel completely exhausted."   | Stress (context-dependent) |

# Limitations
* Single-label classification (does not capture multi-emotion expressions)
* Short or ambiguous text may lead to misclassification
* Limited conversational memory (message-level classification only)

# Future Scope
* Multi-label emotion classification
* Context-aware multi-turn conversation modeling
* Reinforcement learning for adaptive emotional response
* Multilingual and code-mixed emotion detection
* Real-time chatbot deployment in academic systems

# Ethical Considerations
MITRA is designed as a supportive first-layer emotional assistance system. It is not intended to replace professional counseling services. User privacy and ethical AI principles are considered fundamental to the system design.

# Author
```
Lokesh Manepalli
Deep Learning Researcher | AI for Mental Health | Emotion-Aware Systems
```
