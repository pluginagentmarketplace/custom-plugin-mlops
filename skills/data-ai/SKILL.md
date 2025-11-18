---
name: ai-machine-learning
description: Master AI, machine learning, and data science. Learn Python, deep learning, LLMs, MLOps, data pipelines, and production AI systems. Build generative AI applications and deploy scalable ML models.
---

# Data Science & AI/ML Skills

## Quick Start

Data science and AI development involves building machine learning models, analyzing data, and creating intelligent applications powered by AI and LLMs.

### Python ML Basics
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
```

### Deep Learning with PyTorch
```python
import torch
import torch.nn as nn
from torch.optim import Adam

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = NeuralNetwork()
optimizer = Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()
```

### LLM with LangChain
```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short summary about {topic}"
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="Machine Learning")
print(result)
```

## Core Topics

### 1. Python & Data Science Fundamentals
- **NumPy**: Arrays, vectorization, mathematical operations
- **Pandas**: DataFrames, data manipulation, EDA
- **Matplotlib/Seaborn**: Data visualization
- **SQL**: Database queries, data retrieval
- **Statistics**: Distributions, hypothesis testing, correlation

### 2. Machine Learning
- **Supervised Learning**: Classification, regression
- **Algorithms**: Linear regression, decision trees, random forests, SVM
- **Unsupervised Learning**: Clustering, dimensionality reduction
- **Evaluation Metrics**: Accuracy, precision, recall, F1, AUC-ROC
- **Cross-validation**: Model selection and validation

### 3. Deep Learning
- **Neural Networks**: Perceptrons, backpropagation, activation functions
- **Convolutional Neural Networks (CNNs)**: Image classification, object detection
- **Recurrent Neural Networks (RNNs)**: Sequence modeling, LSTM, GRU
- **Transformers**: Attention mechanism, BERT, GPT
- **Frameworks**: PyTorch, TensorFlow, Keras

### 4. Natural Language Processing (NLP)
- **Text Preprocessing**: Tokenization, stemming, lemmatization
- **Word Embeddings**: Word2Vec, GloVe, FastText
- **Language Models**: n-grams, probabilistic models
- **Transformers**: BERT, GPT, T5
- **Applications**: Sentiment analysis, machine translation, Q&A

### 5. Large Language Models (LLMs)
- **LLM Fundamentals**: How transformers work, training objectives
- **Prompt Engineering**: Writing effective prompts, few-shot learning
- **LLM APIs**: OpenAI, Anthropic Claude, Google Gemini
- **Retrieval-Augmented Generation (RAG)**: Vector databases, embeddings
- **Fine-tuning**: LoRA, QLoRA, parameter-efficient adaptation

### 6. MLOps & Production
- **Experiment Tracking**: MLflow, Weights & Biases
- **Model Versioning**: DVC, model registries
- **CI/CD for ML**: Automated testing, deployment pipelines
- **Model Serving**: TensorFlow Serving, TorchServe, FastAPI
- **Monitoring**: Model drift detection, performance tracking

### 7. Data Pipelines
- **ETL/ELT**: Data extraction, transformation, loading
- **Orchestration**: Apache Airflow, Prefect, Dagster
- **Data Quality**: Great Expectations, validation
- **Feature Engineering**: Feature stores, feature extraction
- **Scalability**: Spark, Dask, distributed processing

### 8. Cloud ML Services
- **AWS**: SageMaker, Lambda, S3, EC2
- **Google Cloud**: Vertex AI, BigQuery, Cloud Storage
- **Azure**: Azure ML, Databricks, Cosmos DB

### 9. Data Visualization & Communication
- **Visualization**: Matplotlib, Seaborn, Plotly, Tableau
- **Business Intelligence**: Power BI, Looker, Tableau
- **Storytelling**: Communicating findings effectively
- **Dashboards**: Real-time monitoring, KPI tracking

### 10. AI Ethics & Responsible AI
- **Bias Detection**: Identifying and mitigating bias
- **Fairness**: Ensuring equitable model behavior
- **Transparency**: Explainability, interpretability
- **Privacy**: Differential privacy, data protection
- **Governance**: Model governance, compliance

## Learning Path

### Month 1-2: Fundamentals
- Python programming
- NumPy and Pandas basics
- Data visualization
- SQL fundamentals

### Month 3-4: Machine Learning
- Supervised learning algorithms
- Model evaluation and validation
- Scikit-learn mastery
- Feature engineering basics

### Month 5-6: Deep Learning
- Neural network fundamentals
- CNN and RNN architectures
- PyTorch or TensorFlow
- Real-world image/text projects

### Month 7-8: Generative AI & LLMs
- Prompt engineering techniques
- RAG systems and vector databases
- Fine-tuning and adaptation
- Building LLM-powered applications

### Month 9-10: MLOps
- Experiment tracking
- Model deployment and serving
- Monitoring and drift detection
- CI/CD for machine learning

### Month 11+: Advanced Topics
- Advanced NLP and transformers
- Multi-agent AI systems
- Large-scale ML systems
- Research and cutting-edge techniques

## Tools & Technologies

### Languages & Frameworks
- **Python**: Primary language for ML/AI
- **Deep Learning**: PyTorch, TensorFlow, Keras
- **NLP**: Hugging Face Transformers, spaCy
- **ML**: scikit-learn, XGBoost, LightGBM
- **LLMs**: LangChain, LlamaIndex, Hugging Face

### Data & Databases
- **Data Processing**: Pandas, Polars, Apache Spark
- **SQL**: PostgreSQL, MySQL, BigQuery
- **Vector Databases**: Pinecone, Weaviate, Milvus, FAISS
- **Data Warehouses**: Snowflake, BigQuery, Redshift

### MLOps & DevOps
- **Experiment Tracking**: MLflow, Weights & Biases, Neptune
- **Model Serving**: TensorFlow Serving, TorchServe, BentoML
- **Orchestration**: Apache Airflow, Prefect, Kubeflow
- **Containerization**: Docker, Kubernetes
- **Cloud Platforms**: AWS, GCP, Azure

### Development Tools
- **Notebooks**: Jupyter, JupyterLab, Google Colab
- **IDEs**: VS Code, PyCharm
- **Version Control**: Git, GitHub, DVC
- **Monitoring**: Prometheus, Grafana, Datadog

## Advanced Topics

### Advanced Deep Learning
- **Transfer Learning**: Fine-tuning pre-trained models, domain adaptation
- **Model Compression**: Quantization, pruning, knowledge distillation
- **Multi-GPU Training**: Distributed training, data parallelism, model parallelism
- **Custom Layers**: Implementing custom PyTorch/TensorFlow layers
- **Neural Architecture Search (NAS)**: Automated architecture optimization
- **Adversarial Training**: Robustness against adversarial examples
- **Meta-Learning**: Learning to learn, few-shot learning

### Production ML Systems
- **Feature Stores**: Feast, Tecton for managing features at scale
- **Model Serving**: TensorFlow Serving, TorchServe, Seldon Core, KServe
- **A/B Testing**: Statistical testing for model improvements
- **Model Monitoring**: Data drift, model drift, performance degradation detection
- **Retraining Pipelines**: Automated retraining triggers, continuous training
- **Model Registry**: MLflow Model Registry, Azure ML, SageMaker Model Registry
- **Explainability**: SHAP, LIME for model interpretability

### Advanced LLM & Generative AI
- **Prompt Optimization**: Automated prompt engineering, prompt injection prevention
- **Multi-LLM Orchestration**: Fallback strategies, load balancing
- **Context Window Management**: Efficient context utilization, summarization
- **Streaming LLMs**: Real-time token generation
- **Function Calling**: LLMs calling external APIs
- **Retrieval Optimization**: Hybrid search, semantic ranking, reranking
- **LLM Fine-tuning at Scale**: LoRA, QLoRA, Adapter tuning

### Data Engineering at Scale
- **Data Lakehouse Architecture**: Delta Lake, Apache Iceberg, Apache Hudi
- **Stream Processing**: Apache Kafka, Flink, Spark Streaming
- **Data Quality Frameworks**: Great Expectations, Soda, Deequ
- **Data Lineage Tracking**: OpenLineage, data catalogs
- **Real-time Feature Engineering**: Feature stores with streaming
- **Data Governance**: Metadata management, data lineage, compliance

### Responsible AI
- **Bias Detection**: Identifying and mitigating bias in datasets
- **Fairness Metrics**: Demographic parity, equalized odds
- **Explainability Methods**: Feature importance, SHAP values, attention mechanisms
- **Privacy-Preserving ML**: Differential privacy, federated learning
- **Model Governance**: Model documentation, risk assessment, compliance
- **Ethical Guidelines**: Following responsible AI principles
- **Adversarial Robustness**: Testing model robustness to attacks

## Common Pitfalls & Gotchas

1. **Data Leakage**: Information from test set leaks into training
   - **Fix**: Ensure proper train/test split, scale before split
   - **Example**: Normalize entire dataset BEFORE splitting

2. **Imbalanced Classes**: Model predicts majority class for everything
   - **Fix**: Use SMOTE, class weights, stratified splitting
   - **Example**: `class_weight='balanced'` in scikit-learn

3. **Overfitting to Test Set**: Tuning hyperparameters on test data
   - **Fix**: Use validation set, cross-validation, separate test set
   - **Example**: Train/validation/test split (70/15/15)

4. **Feature Scaling Forgotten**: Some algorithms require scaled input
   - **Fix**: Always scale features (StandardScaler, MinMaxScaler)
   - **Models**: KNN, SVM, neural networks require scaling

5. **Ignoring Baseline**: Complex model no better than simple baseline
   - **Fix**: Always establish baseline (dummy classifier, simple model)
   - **Example**: Logistic Regression before Deep Learning

6. **P-Hacking**: Multiple tests increase false positive probability
   - **Fix**: Adjust significance level, pre-register experiments
   - **Lesson**: Proper hypothesis testing, multiple comparison correction

7. **Model Not Reproducible**: Same code produces different results
   - **Fix**: Set random seeds, document data processing
   - **Example**: `random_state=42`, `torch.manual_seed(42)`

8. **Ignoring Class Imbalance**: Accuracy misleading when classes imbalanced
   - **Fix**: Use F1-score, precision-recall, ROC-AUC
   - **Example**: If 99% negative, accuracy can be 99% with all predictions negative

9. **Not Monitoring Drift**: Model degrades over time in production
   - **Fix**: Monitor data drift, model drift, performance metrics
   - **Tools**: Evidently, Whylabs, Arize

10. **Poor Prompt Engineering**: LLM returns irrelevant results
    - **Fix**: Use system prompts, chain-of-thought, few-shot examples
    - **Example**: "Explain your reasoning step by step."

## Production Deployment Checklist

- [ ] Data pipeline validated and monitored
- [ ] Model training reproducible (seeds, code, data versions)
- [ ] Feature store set up (if applicable)
- [ ] Model evaluation metrics tracked
- [ ] Cross-validation performed (not just train/test split)
- [ ] Baseline established and documented
- [ ] Model interpretability analyzed
- [ ] Bias and fairness assessed
- [ ] Model serving infrastructure ready
- [ ] Monitoring for data/model drift configured
- [ ] A/B testing framework set up
- [ ] Retraining pipeline automated
- [ ] Documentation complete (methodology, limitations, data)
- [ ] Model versioning and registry configured
- [ ] Compliance and privacy requirements met

## Best Practices

1. **Data Quality First** - Garbage in, garbage out
2. **Baseline Matters** - Know your baseline before complex models
3. **Reproducibility** - Track all experiments, document processes
4. **Proper Validation** - Train/validation/test split, cross-validation
5. **Monitor Everything** - Data drift, model drift, performance
6. **Ethics & Bias** - Fairness, privacy, accountability
7. **Testing** - Unit tests, integration tests, model tests
8. **Documentation** - Methodology, limitations, deployment info
9. **Scalability** - Plan for production scale from the start
10. **Continuous Improvement** - A/B testing, feedback loops, retraining

## ML Architecture Patterns

### Feature Engineering Pipeline
```
Raw Data → Cleaning → Feature Engineering → Scaling → Training
                ↑                                              ↓
          Monitoring ← Feature Store ← Versioning ← Model Registry
```

### MLOps Pipeline
```
Data Collection → Validation → Feature Engineering → Model Training
                                                           ↓
                                                    Model Evaluation
                                                           ↓
                                                    Model Registry
                                                           ↓
                                            Model Serving (Production)
                                                           ↓
                                         Monitoring & Retraining Triggers
```

## Performance Optimization Checklist

- [ ] Data preprocessing optimized
- [ ] Feature selection/engineering completed
- [ ] Hyperparameter tuning performed
- [ ] Model size acceptable for deployment
- [ ] Inference latency measured (< 100ms goal)
- [ ] Batch processing vs. real-time considered
- [ ] GPU/TPU utilization optimized
- [ ] Memory usage profiled
- [ ] Model quantization/compression applied if needed

## Testing Best Practices

```python
# Good test - tests model behavior
def test_model_predictions_reasonable():
    model = load_model()
    predictions = model.predict(test_data)

    assert predictions.shape[0] == test_data.shape[0]
    assert predictions.min() >= 0 and predictions.max() <= 1  # Probabilities
    assert not np.isnan(predictions).any()

# Bad test - too specific
def test_model_accuracy_exact():
    model = load_model()
    acc = model.score(test_data, test_labels)
    assert acc == 0.95  # Too strict, will fail due to randomness

# Better test - with tolerance
def test_model_accuracy_reasonable():
    model = load_model()
    acc = model.score(test_data, test_labels)
    assert acc >= 0.85  # Acceptable range
```

## Resources & Learning

### Documentation
- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Docs](https://pytorch.org/docs)
- [TensorFlow/Keras](https://keras.io)
- [Scikit-learn Guide](https://scikit-learn.org)
- [MLflow Docs](https://mlflow.org/docs)

### Learning Platforms
- [Fast.ai](https://fast.ai) - Practical deep learning
- [Kaggle Learn](https://kaggle.com/learn) - Quick courses
- [DeepLearning.AI](https://deeplearning.ai) - Structured courses
- [Papers With Code](https://paperswithcode.com) - Research to code

### Staying Current
- [ArXiv.org](https://arxiv.org) - Latest research papers
- [Papers Weekly](https://papersweekly.site)
- [The Batch](https://batch.deeplearning.ai) - AI newsletter

## Interview Preparation

### Common ML Interview Topics
1. **Supervised Learning**: Regression, classification, evaluation metrics
2. **Unsupervised Learning**: Clustering, dimensionality reduction
3. **Feature Engineering**: Domain knowledge, feature selection
4. **Model Selection**: Bias-variance tradeoff, regularization
5. **Time Series**: ARIMA, seasonality, forecasting
6. **NLP Basics**: Tokenization, embeddings, word2vec
7. **Deep Learning**: Neural networks, CNNs, RNNs, Transformers
8. **Production ML**: Monitoring, A/B testing, deployment

### System Design Interview
- **Design a Recommendation System**: Collaborative filtering, content-based
- **Design a Ranking System**: Learning-to-rank, online learning
- **Design a Fraud Detection System**: Imbalanced classification, real-time
- **Design an ML Pipeline**: Data → Training → Serving → Monitoring

## Next Steps

1. **Master Python and fundamentals** - NumPy, Pandas, Scikit-learn
2. **Build end-to-end projects** - Kaggle competitions, real datasets
3. **Learn deep learning** - Neural networks, PyTorch/TensorFlow
4. **Explore LLMs** - Prompt engineering, fine-tuning, RAG systems
5. **Study MLOps** - Production pipelines, monitoring, deployment
6. **Focus on areas of weakness** - If weak on math, review linear algebra and calculus
7. **Stay research-focused** - Read papers, understand SOTA techniques
