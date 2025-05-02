# AI Artwork Enhancer

AI Artwork Enhancer is a web-based tool that provides technical analysis and automatic enhancements for uploaded artworks. It uses LLaVA through Ollama to deliver detailed image feedback and applies suggested improvements. Premium features include color palette extraction and stylistic variations.

## Features

- User authentication: sign up, login, logout, and account upgrades.
- AI-based image feedback using LLaVA via Ollama.
- Automated parsing of feedback into actionable image adjustments.
- Color adjustments: contrast, brightness, saturation, sharpness, and color balance.
- Hue shifting and enhancement based on parsed instructions.
- Premium-only:
  - Extract dominant color palettes using KMeans clustering.
  - Apply predefined artistic styles (Vintage, Monochrome, Watercolor).

## Technologies Used

- Python
- Gradio (for the user interface)
- Pillow (PIL) for image processing
- NumPy for numerical operations
- scikit-learn for KMeans clustering
- Ollama + LLaVA for AI-powered image analysis
