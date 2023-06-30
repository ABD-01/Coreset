# Coreset Extraction using Greedy Selection

## About

This repository proposes a greedy algorithm for coreset selection based on approximating the mean gradient. The goal is to select a subset of data (coreset) that can be used to train a model with performance comparable to training on the full dataset. This approach leverages gradient information to identify the most impactful samples for the learning process.

This work was done during my Research Intern under the guidance of [Prof. Dr. Konda Mopuri](https://krmopuri.github.io/) at IIT Guwahati.

## How It Works

The algorithm iteratively selects data points that approximate the mean gradient of the full dataset. This ensures that the coreset has a similar gradient distribution to the original dataset, making it representative and effective for training.

## Experiments

Detailed experiments and results can be found [**here**: muhammed-abd.notion.site](https://muhammed-abd.notion.site/3183643bc62d47af8264ed5862f8d84f?v=45060469a6d349d59801dfaceeec2d90).

