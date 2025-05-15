# Relation Extraction from Texts in Biodiversity Domain

This project focuses on extracting semantic relationships from textual data within the biodiversity domain. It aims to identify and categorize relationships between entities such as species, habitats, and ecological interactions using natural language processing (NLP) techniques.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)

## Overview

The primary goal of this project is to develop a system capable of extracting meaningful relationships from biodiversity-related texts. This involves:

- Processing textual data to identify entities relevant to biodiversity.
- Extracting relationships between these entities.
- Storing and analyzing the extracted relationships for further research and application.

## Project Structure

The repository is organized as follows:

- `main.py`: The main script that orchestrates the relation extraction process.
- `utils/`: Contains utility functions and helper scripts used across the project.
- `data/`: Directory for storing input datasets and intermediate files.
- `prompts.txt`: Contains prompt templates used for guiding the NLP model during extraction.
- `relation_examples2x.csv`: A CSV file with example relationships used for training or evaluation.
- `predicted_relations.csv`: Output file where the extracted relationships are stored.
- `requirements.txt`: Lists the Python dependencies required to run the project.

## Installation

To set up the project environment, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MilVosk/Thesis_.git
   cd Thesis_
2. **Install the required dependencies**
    ```bash
    pip install -r requirements.txt

3. **Run the relation extrection**
    ```bash
    python main.py
