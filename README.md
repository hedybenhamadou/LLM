[README.md](https://github.com/user-attachments/files/27281457/README.md)
# GPT-2 from Scratch — Implémentation PyTorch

Une implémentation complète du modèle de langage GPT-2 en PyTorch, construite brique par brique, avec chargement des poids préentraînés d'OpenAI et génération de texte.

---

## Présentation

Ce notebook guide à travers la construction d'un modèle Transformer compatible GPT-2 entièrement from scratch, puis le chargement des poids officiels préentraînés d'OpenAI pour faire de l'inférence. Il est conçu comme une ressource pédagogique pour comprendre les rouages internes des grands modèles de langage.

---

## Architecture

Le modèle est construit composant par composant :

- **SelfAttention** attention par produit scalaire mis à l'échelle, version de base
- **CausalAttention**  ajoute un masque causal et du dropout pour la génération autoregressive
- **MultiHeadAttention**  répartit l'attention sur plusieurs têtes avec une projection de sortie
- **LayerNorm**  normalisation de couche personnalisée avec paramètres d'échelle et de décalage apprenables
- **GELU**  fonction d'activation Gaussian Error Linear Unit (approximation par tanh)
- **FeedForward**  MLP à deux couches avec expansion 4× et activation GELU
- **TransformerBlock**  combine attention multi-têtes et feed-forward avec connexions résiduelles et pré-normalisation
- **GPTModel**  modèle complet avec embeddings de tokens, embeddings positionnels, blocs Transformer empilés et tête de modèle de langage

---

## Configuration du modèle (GPT-2 Small, 124M)

Le modèle de référence utilisé dans ce notebook est le GPT-2 Small avec 124 millions de paramètres. Il utilise un vocabulaire de 50 257 tokens, une longueur de contexte de 1 024 tokens, une dimension d'embedding de 768, 12 têtes d'attention, 12 couches Transformer et un taux de dropout de 0.1.

Le notebook définit également les configurations pour les variantes GPT-2 Medium (355M), Large (774M) et XL (1558M).

---

## Fonctionnalités

- **Chargement des poids préentraînés** — télécharge les checkpoints GPT-2 depuis le stockage public d'OpenAI et mappe les poids TensorFlow dans le modèle PyTorch
- **Génération de texte** — supporte le décodage greedy, l'échantillonnage top-k et la mise à l'échelle par température
- **Arrêt anticipé** — la génération peut s'interrompre sur un token de fin de séquence configurable
- **Sauvegarde et restauration** — sauvegarde et recharge l'état du modèle et de l'optimiseur

---

## Prérequis

```
torch
tiktoken
tensorflow >= 2.15.0
tqdm >= 4.66
numpy
requests
```

Installation :

```bash
pip install torch tiktoken tensorflow>=2.15.0 tqdm numpy requests
```

---

## Utilisation

### 1. Construire et tester avec des poids aléatoires

```python
model = GPTModel(GPT_CONFIG_124M)
model.eval()

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
```

### 2. Charger les poids préentraînés d'OpenAI

```python
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
load_weights_into_gpt(gpt, params)
```

### 3. Générer du texte

```python
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Bonjour, je suis", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.4
)
print(token_ids_to_text(token_ids, tokenizer))
```

### 4. Sauvegarder et restaurer un checkpoint

```python
# Sauvegarde
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "model_and_optimizer.pth")

# Chargement
checkpoint = torch.load("model_and_optimizer.pth")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

---

## Structure du projet

```
gbt2_model.ipynb            # Notebook principal — architecture, chargement des poids, génération
gpt2/                       # Poids préentraînés téléchargés (créé à l'exécution)
model.pth                   # Poids du modèle sauvegardés (créé à l'exécution)
model_and_optimizer.pth     # Checkpoint complet (créé à l'exécution)
```

---

## Références

- [GPT-2 — Language Models are Unsupervised Multitask Learners](https://openai.com/research/language-unsupervised)
- [Poids publics GPT-2 d'OpenAI](https://openaipublic.blob.core.windows.net/gpt-2/models)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
