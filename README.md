# Minimax Multi-Agent : Pong compétitif et Ball Sort coopératif
Ce projet explore l'**apprentissage par renforcement multi-agent (MARL)** à travers deux environnements interactifs développés en Python et rendus en temps réel avec **Pygame** :

1. **Pong compétitif (Minimax Q-Learning)** : Deux raquettes s'affrontent dans un jeu à somme nulle. L'agent utilise l'algorithme Minimax pour anticiper et contrer le meilleur coup de son adversaire, maximisant son gain minimal dans un cadre théorique issu de la théorie des jeux (Littman, 1994).
2. **Ball Sort coopératif (Independent Q-Learning - IQL)** : Deux agents doivent collaborer sans communication directe pour trier des balles colorées dans des tubes. Chaque agent apprend de manière indépendante avec sa propre table Q, illustrant les dynamiques d'émergence de la coordination ainsi que les défis de non-stationnarité de l'environnement (Tan, 1993).

Le projet propose une visualisation dynamique des entraînements, le contrôle de la vitesse de simulation en temps réel et une analyse pas à pas de la convergence des algorithmes.


| Jeu | Type d'interaction | Algorithme | Fichier |
|-----|--------------------|------------|---------|
| **Pong** | Compétitif (somme nulle) | Minimax Q-Learning | `minimax_pong_v3.py` |
| **Ball Sort** | Coopératif | Independent Q-Learning (IQL) | `ball_sort_iql.py` |

> 📸 **Aperçu** :![alt text](<images/Capture d’écran 2026-09-28 à 14.00.50.png>) .
>
> `![Pong](images/pong.gif)` · `![Ball Sort](images/ball_sort.gif)`
![alt text](<images/Capture d’écran 2026-09-28 à 14.01.26.png>)
---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

Dépendances : `pygame` (affichage) et `numpy` (calcul).

## Lancement

```bash
python3 minimax_pong_v3.py         # Pong : deux agents qui s'affrontent
python3 ball_sort_iql.py           # Ball Sort : deux agents qui coopèrent
```

### Contrôles

| Jeu | Touche | Action |
|-----|--------|--------|
| Pong | `1` à `9` | Vitesse de simulation (1 = normal, 9 = 100x) |
| Pong | `ESPACE` | Pause |
| Pong | `R` | Réinitialiser |
| Pong | `ÉCHAP` | Quitter |
| Ball Sort | `ESPACE` | Pause / reprise de l'entraînement |
| Ball Sort | `F` | Mode rapide |
| Ball Sort | `ÉCHAP` | Quitter |

---

## 1. Pong : Minimax Q-Learning

Deux raquettes s'affrontent. Ce que gagne un agent, l'autre le perd : c'est un **jeu à somme nulle**.

### Fonctionnement

- **État** (du point de vue de chaque agent) : `(balle_vers_moi, position_relative)` où la position relative vaut `ABOVE`, `AT` ou `BELOW`.
- **Actions** : `UP`, `STAY`, `DOWN`.
- **Récompense** : `+1` quand l'agent 1 marque, `-1` quand il encaisse, `0` sinon.
- **Table Q conjointe** : `Q(s, a1, a2)`, la valeur de l'état quand les deux agents jouent leurs actions.
- **Valeur d'un état (règle minimax)** :

  ```
  V(s) = max_a1  min_a2  Q(s, a1, a2)
  ```

- **Mise à jour** :

  ```
  Q(s, a1, a2) ← Q(s, a1, a2) + α · [ r + γ · V(s') − Q(s, a1, a2) ]
  ```

### Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| α (taux d'apprentissage) | 0.5 |
| γ (facteur d'actualisation) | 0.95 |
| ε initial | 0.3 |
| ε minimum | 0.02 |
| Décroissance de ε | 0.998 |

### Ce qu'on observe

Le compteur d'échanges (rally) augmente au fil de l'apprentissage : les agents apprennent à suivre la balle au lieu de jouer au hasard.

---

## 2. Ball Sort : Independent Q-Learning

Deux agents doivent **coopérer** pour trier des balles colorées dans des tubes. Ils partagent la même récompense mais ne communiquent pas : chacun a sa propre table Q.

### Fonctionnement

- **Environnement** : 4 tubes de capacité 4, 3 couleurs (un tube reste vide), 2 agents.
- **État** : `(couleurs_du_dessus, tailles_des_tubes)`.
- **Actions** : `(tube_source, tube_destination)` ou `STAY`. Chaque agent peut déplacer une balle depuis n'importe quel tube.
- **Mise à jour** (pour chaque agent, indépendamment) :

  ```
  Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]
  ```

### Récompenses

| Événement | Récompense |
|-----------|------------|
| Puzzle résolu | +100 |
| Balle mieux triée | +2 par balle gagnée |
| Mouvement invalide | −1 par mouvement |
| `STAY` alors que des mouvements existent | −0.3 |
| Coût par étape | −0.1 |

### Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| α | 0.15 |
| γ | 0.95 |
| ε initial | 1.0 |
| ε minimum | 0.02 |
| Décroissance de ε | 0.998 |

### Phases d'entraînement

| Phase | Épisodes | Ce qu'on observe |
|-------|----------|------------------|
| 1 | 1 – 30 | Exploration quasi aléatoire |
| 2 | 30 – 100 | Premiers motifs qui émergent |
| 3 | 100 – 300 | Coordination entre agents |
| 4 | 300+ | Exploitation et maîtrise |

### Limite connue : la non-stationnarité

Chaque agent apprend en considérant l'autre comme faisant partie de l'environnement. Or l'autre agent change en permanence, donc l'environnement « bouge » du point de vue de chacun. C'est la principale limite de l'IQL et elle est expliquée dans la documentation.

---
## Documentation

- [Minimax Q-Learning expliqué](./docs/MINIMAX_Q_LEARNING_EXPLICATION.md)
- [Ball Sort IQL : guide complet](./docs/BALL_SORT_IQL_EXPLICATION_FR.md)


## Références

- Littman, M. (1994). *Markov games as a framework for multi-agent reinforcement learning.*
- Tan, M. (1993). *Multi-agent reinforcement learning: Independent vs. cooperative agents.*
- Sutton, R. & Barto, A. *Reinforcement Learning: An Introduction.*