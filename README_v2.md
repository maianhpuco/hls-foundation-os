```
conda activate hls_fresh 
``` 

pip install -e .

Input Image (multi-band, optional multi-frame)
     ↓
TemporalViTEncoder (tokenization + transformer)
     ↓
ConvTransformerTokensToEmbeddingNeck (reshape into feature map)
     ↓
┌───────────────────────┐
│   Decode Head (FCN)    │───► Main segmentation output
└───────────────────────┘
┌───────────────────────┐
│ Auxiliary Head (FCN)   │───► Auxiliary output (training only)
└───────────────────────┘

------------- 
1. Backbone: TemporalViTEncoder
Input: Images with multiple channels (len(bands) = 6) and multiple frames (num_frames = 1 here, but the model is flexible).

Key points:
- Patchify input into small patches (size $16 \times 16$) → create tokens.
- Apply a Transformer Encoder with:

$12$ layers (depth).
$12$ attention heads.
$768$-dimensional embeddings.
Can handle sequences over time (if num_frames > 1).

Loads pretrained weights (optional).
→ Output: a set of patch embeddings representing spatial-temporal features. 



2. Neck: ConvTransformerTokensToEmbeddingNeck

Purpose: Convert flat transformer tokens into a feature map format ($C \times H \times W$).
Key points: 
- It reshapes and aggregates transformer tokens spatially.
- Drops the class token (if any).
- Embedding dimension after aggregation is still $768$.

→ Output: a $768 \times 14 \times 14$ feature map.

3. Decode Head: FCNHead

\textbf{Purpose:} Perform semantic segmentation from the feature map.
\textbf{Key points:}
- Simple Fully Convolutional Network (1 convolution layer).
- Reduces channels from $768$ → $256$ internally → output 2-class prediction (flood vs no flood).
- Loss function: Weighted Cross-Entropy Loss.

Weights: [0.3, 0.7] → puts more weight on the flooded class.

→ Output: Pixel-wise prediction of classes.

4. Auxiliary Head: another FCNHead
\textbf{Purpose:} Add extra supervision to intermediate features (stabilizes training).

Same structure as the main decode head. 

 

Modification later: 
- Short prompts vs. long prompts
Compare num_prompts=4, 8, 16, 32
Small number = quick, lightweight adaptation
Larger number = model can learn richer adaptation.

- Mid-layer prompt injection (optional, harder)
At Block 4 or Block 8, inject new prompts.
Need modifying the forward() of Transformer.
-> Insert small new prompt tokens at intermediate stages. 
(If you want I can help you code a basic example if you're interested.)

- Prompt initialization from pretrained parameters
Instead of random, initialize prompts from position embeddings or CLS token.

Gives a better starting point → faster convergence.