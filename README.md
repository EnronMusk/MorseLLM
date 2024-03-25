### **Morse LLM** ###

I created the transformer architecture from scratch with all the supported functions and trained a model on some shakespear plays (500k chars). I created a nice notebook to go along with training analysis and tests.

**Results**

LLMs can easily pick up on the patterns present in morse code naturally. In particular, LLMs can adapt to the morse code increasing the average token length by a factor of **3.37** with a proportional increase in context length. In particular, with a proportional increase in ONLY the context length we could maintain very consistent results with the original model. This is despite each token indivudally having much less meaning compared to a model on the english characters, so the embeddings, attention heads and feedforwards are unaffected as long as the context_length is context (in pure character terms).
