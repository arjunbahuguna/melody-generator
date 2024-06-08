# Melody Generator

Generate melodies with transformers!

## Overview

This project implements a melody generation pipeline using transformer models. It includes scripts for preprocessing MIDI data, training a transformer model, and generating new melodies based on a given starting sequence. The implementation leverages PyTorch for the model, Mido for MIDI file processing, and DVC for data and model versioning.

## Features

- **Data and Model Versioning**: Utilizes DVC to track datasets and models.
- **Transformer Model**: Custom transformer architecture tailored for sequence-to-sequence melody generation.
- **Inference Pipeline**: Scripts to generate melodies and save them as MIDI files.

## Upcoming Features

- BentoML inference server for serving models in production.
- Containerization using Docker for consistent deployments.
- GitHub Action workflows for seamless CI/CD integration with AWS and Docker.
