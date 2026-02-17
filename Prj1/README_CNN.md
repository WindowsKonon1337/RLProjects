```
Input:
  [Grid (4ch)] + [Coords (2ch)]  --->  (Batch, 6, 7, 7)
       |
       v
  [Conv1] (32 filters, 3x3)      --->  (Batch, 32, 7, 7)
       |
       v
  [Conv2] (64 filters, 3x3)      --->  (Batch, 64, 7, 7)
       |
       v
  [Conv3] (64 filters, 3x3)      --->  (Batch, 64, 7, 7)
       |
       +-------------------------------------+
       |                                     |
[Branch A: Spatial Map]             [Branch B: Global Stats]
       |                                     |
   Flatten (3136)                    Global Avg Pool (1x1)
       |                                     |
   Linear(256) [Projection]             Flatten (64)
       |                                     |
       v                                     v
   [Vec: 256]                            [Vec: 64]
       |                                     |
       +------------------+------------------+
                          |
                  [Branch C: Safety]
             Input: 8 neighbors (Vec: 8)
                          |
                    Linear(32)
                          |
                          v
                      [Vec: 32]
                          |
--------------------------+---------------------------
                          |
               FUSION: [256 + 64 + 32] = 352
                          |
               Linear(352 -> 512) + ReLU
                          |
               Linear(512 -> 4)   [Raw Logits]
                          |
               * LOGIT_SCALE (0.05) [Scaled Logits]
                          |
               Softmax -> Probabilities
```