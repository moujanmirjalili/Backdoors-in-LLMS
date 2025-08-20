# Backdoors in LLMs – SAE Feature Analysis

This repository contains code and experiments for the NLP38 Selected Topics in Data Science project Backdoors in LLMs, supervised by Dominik Meier and Jan Philip Wahle.

## Project Overview
Backdoors are hidden triggers in prompts that cause LLMs to respond in adversary-controlled ways. The aim of this project is to use Sparse Autoencoders (SAEs) to analyze internal activations of language models and identify features that spike when triggers are present.

## The aim of this project is to:
- Apply pretrained SAEs to model activations.
- Compare activations from clean vs. triggered prompts.
- Identify features that spike when the trigger is present.
- Analyze patterns across prompts.

## Approach
1. **Initial Exploration with Gemma Notebook** - Gemma.ipynb
   - I used this notebook to get an overall overview to understand (Sparse AutoEncoders) SAEs on Gemma-2B model and How they work.
   - This notebooke helped me get familiar with hooks, activations, and Neuronpedia dashboard.

2. **SAE Analysis** - SAE.ipynb 
   - In this notebook I used pretrained SAEs on GPT-2 to explore how internal model activations can be decomposed into interpretable features.
   - GPT-2 + SAEs was limited in interpretability, therefore in **later notebooks I switched to gemma-2B** not instruction tuned model.
   - *Note: Further explainations about model choices is in Model section*
   - Main components:
       - SAE Lens Setup: Loading a pretrained SAE with SAE Lens and understanding its class and configuration.
       - Feature Exploration: Inspecting SAE features through dashboards, linking them to Neuronpedia, and using autointerp explanations.
       - Feature Inference: Using HookedSAETransformer to decompose GPT-2 activations into sparse features and comparing them across related prompts.
       - Visualization Dashboards: Generating max activating examples, activation histograms, and logit weight distributions. Including reproducing results from “Not all language model features are linear.”
       - Prompt Comparisons: Testing prompt pairs (clean vs. altered) to see which features spike under different inputs.
       - Analysis Methods: Applying SAE-based interpretability tools:
         - Steering model generation with selected features.
         - Ablating features to test their causal role in predictions.

3. **Prompt Comparison** - Prompts comparison.ipynb
   - Ran clean vs. triggered prompts through Gemma-2B model.(Not instruction-tuned)
   - Prompts detail: I used *contrast pairs trick* (two similar prompts with/without trigger) to pinpoint which features drive behavior changes. But how do the prompts look like?
      - Here I have 3 categories of toxic prompts:
        1. Prefix Injection
        2. Context Ignoring
        3. Refusal Suppression (more detail about these categories can be found in the notebook.)
      - For each category I have 4 sets of results:
        1. triggered prompts + triggered answers
        2. triggered prompts + clean answers
        3. clean prompts + triggered answers
        4. clean prompts + clean answers
   - I then tried to compared feature activations and identified features that spike when a trigger is present across my outputs.
   - Analyzed patterns across prompts. (Results are in the Findings section of the notebook)

4. **Pattern Extraction Across Features** - Feature activation comparison.ipynb
   - Compared features activation patterns across different clean vs. triggered prompts.
   - Plotted per-feature activation values using Neuronpedia and visualization tools to interpret the semantics of activated features.
   - Large differences observed in some features and found out that triggered prompts prime the model for helpful technical output. (Interpretations are in the Findings section of the notebook)

## Model

## Important Notes
- Some of the outputs such as Neuronpedia interactive dashboards cannot be displayed on notebooks in github, therefore you need to run the notebooks yourself or use the neuronpedia website, or email me moujanmirjalili@gmail.com to get the notebooks with complete results

## Main Libraries:
  - [`sae_lens`](https://github.com/jbloomAus/SAELens) – loading & running SAEs
  - [`transformer_lens`](https://github.com/neelnanda-io/TransformerLens) – model control & tokenization
  - [`Neuronpedia`](https://www.neuronpedia.org) – interactive feature interpretation

## References
- Bloom, J., Tigges, C., Duong, A., & Chanin, D. (2024). SAELens. 
- Olah et al. (2023). Monosemantic Features. transformer-circuits.pub
- Neel Nanda. Mechanistic Interpretability. neelnanda.io
- Alignment Forum. SAEs and Backdoor Detection. alignmentforum.org
- Evertz, J. (2025). Whispers in the Machine: Confidentiality in LLM-Integrated Systems. https://arxiv.org/pdf/2402.06922
