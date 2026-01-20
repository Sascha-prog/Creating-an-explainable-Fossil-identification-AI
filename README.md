
# LegaSea: A Multi-Agent System for Explainable Fossil Identification

![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)


![alt text](https://img.shields.io/badge/python-3.9+-blue.svg)


![alt text](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)


![alt text](https://img.shields.io/badge/Affiliation-Naturalis%20Biodiversity%20Center-green)

Precise fossil identification is a fundamental prerequisite for reconstructing Ice Age biomes and documenting historical biodiversity patterns. The LegaSea project, coordinated by the Naturalis Biodiversity Center, aims to reconstruct Dutch Ice Age biomes through an AI-assisted citizen science approach. This project part focuses on developing and evaluating AI models for fossil identification, improving explainability methods, and enhancing the interaction between users and AI systems. The project seeks to make AI predictions intelligible, trustworthy, and useful for citizen scientists
 

## 🏛 Project Context

Coordinated by the Naturalis Biodiversity Center, this project leverages a Design Science Research (DSR) methodology to transform crowdsourced data from oervondstchecker.nl into scientifically validated reconstructions. This repository represents the fulfillment of Level 4 Software Layer competencies (Analyse, Design, Realize, Advice).
## 🏗 System Architecture

The framework moves away from monolithic linear pipelines in favor of a Decoupled Multi-Agent Architecture. By separating cognitive tasks, the system ensures technical reproducibility and prevents "Clever Hans" effects—where models rely on environmental noise (like scale bars) rather than biological morphology.
### Agent processing Tier

   - Identification Agent: Uses a CLIP (ViT-B-32) transformer backbone pretrained on the laion2b_s34b_b79k dataset and fine-tuned on 5,000 expert-verified specimens.

   - Reasoning Agent: A fine-tuned BLIP model that generates linguistic justifications for classifications, providing educational feedback to citizen scientists.

   - Inspector Agent (XAI): Provides dual-track visual validation via Grad-CAM (spatial focus) and LIME (texture/super-pixel relevance).

   - Orchestrator: Manages asynchronous communication via the AgentMessage protocol, ensuring consistent data payloads (tensors, labels, and heatmaps).

## 🚀 Getting Started
### Prerequisites

   - Python 3.9 or higher

   - CUDA-compatible GPU (strongly recommended for BLIP inference)

### Installation

```Bash
# Clone the repository
git clone https://github.com/Sascha-prog/Creating-an-explainable-Fossil-identification-AI.git

# Navigate to Scripts and open FinalPrototype.ipynb
# Run the code blocks from top to bottom to replicate each step
```
### 📂 Data Acquisition

The project utilizes a dataset of 5,000 specimens mapping to Naturalis reviewer notes. To synchronize the local image repository with the remote dataset, use the provided scraper utility:
```Bash
# Ensure your mapping CSV is located at ../Dataset/images_mapping_newest.csv
python scripts/scraper.py
```
The scraper performs stream-based acquisition and includes idempotency checks to resume interrupted downloads.

## 📊 Scientific Validation

The system’s reliability is measured using the Drop-in-Confidence (DIC) metric. By masking the diagnostic regions identified by the Inspector Agent, we measure the resulting confidence decay in the classification.

   - Mean Residual Confidence: 0.6562

   - Significance: This quantitative audit ensures the model's logic is grounded in diagnostic morphological features rather than spurious background correlations.

## 🔮 Future Roadmap

   - Conversational AI: Re-introducing high-parameter models (e.g., BLIP-FLAN) fine-tuned on specialized paleontological jargon for interactive user querying.

   - Preprocessing ROI: Implementing automated bounding-box localization to isolate fossil specimens from sediment and scale-bar noise.

   - On-Device Deployment: Quantizing weights for real-time mobile inference at the point of discovery.

   - Standardized Expert Feedback: Enhancing datasets with descriptive expert annotations rather than binary labels.

## 📚 Reference & Citation

If you use this artifact in your research or wish to reference the portfolio deliverables, please use the following citation format:
code Bibtex
```Bibtex
@software{Ingemey2026LegaSea,
  author = {Ingemey, Sascha},
  title = {The LegaSea Project: A Multi-Agent System for Explainable Fossil Identification},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Sascha-prog/Creating-an-explainable-Fossil-identification-AI}}
}
```
## 📄 License

Distributed under the MIT License. See LICENSE for more information.
