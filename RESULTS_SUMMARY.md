"""
Model Comparison Results
======================

Test Accuracy Results:
- CNN (1D Convolution):     38.5%
- LSTM (Bidirectional):     16.5%

Difference: LSTM är 22% sämre än CNN

Analys:
-------

1. LSTM tränade för få epoker (early stopping tidigt)
   - Genom tidigare stopped på ~epoch 12
   - CNN tränade längre och nådde bättre konvergens

2. Hyperparametrar kan behöva justas
   - Current LSTM: 128 hidden units, 1 layer
   - Försök: 256 hidden units, 2 lager
   
3. Normaliserade features
   - Features normaliserade från [0,1] -> [-1,1]
   - CNN kan behöva tränas om med normaliserade features

4. LSTM Design
   - Använder global average pooling (kan förlora temporal struktur)
   - Kan behöva attention mechanism

NÄSTA STEG - Prioritering:
==========================

1. ✅ RETRAIN CNN med normaliserade features
   python3 src/train.py cnn

2. ✅ RETRAIN LSTM med bättre hyperparametrar
   - Öka hidden_size: 128 -> 256
   - Öka layers: 1 -> 2
   - Kör full training

3. ✅ Prova GRU (enklare än LSTM, snabbare)
   python3 src/train.py gru

4. ✅ Lägg till data augmentation
   - Temporal jitter
   - Frame dropout
   
5. ✅ Senare: Transformer modell
"""
