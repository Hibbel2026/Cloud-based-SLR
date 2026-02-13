# GitHub Authentication for Colab

## Problem
`fatal: could not read Username for 'https://github.com': No such device or address`

## Solution

Du behöver autentisera med GitHub från Colab. Här är två metoder:

### Metod 1: Personal Access Token (ENKLAST) ⭐

1. **Skapa Personal Access Token på GitHub:**
   - Gå till: https://github.com/settings/tokens
   - Klicka "Generate new token" → "Generate new token (classic)"
   - Ge det ett namn, t.ex. "Colab-Training"
   - Välj scope: `repo` (full control of private repositories)
   - Klicka "Generate token"
   - **KOPIERA och SPARA token någonstans säker!**

2. **I Colab, använd denna kommando för att clona:**
```python
!git clone https://YOUR_USERNAME:YOUR_TOKEN@github.com/Hibbel2026/Cloud-based-SLR.git
%cd Cloud-based-SLR
```

3. **Eller för push efter träning:**
```python
!git remote set-url origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/Hibbel2026/Cloud-based-SLR.git
!git push origin Baseline-ML
```

### Metod 2: SSH Key (SÄKRARE men mer komplex)

```python
# I Colab:
!ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N ""
!cat /root/.ssh/id_ed25519.pub
```

Kopiera output, gå till https://github.com/settings/keys, lägg till SSH key.

Sedan:
```python
!git clone git@github.com:Hibbel2026/Cloud-based-SLR.git
```

---

## Rekommenderad Setup för Colab

```python
# Cell 1: Clone med Personal Access Token
YOUR_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
YOUR_USERNAME = "Hibbel2026"

!git clone https://{YOUR_USERNAME}:{YOUR_TOKEN}@github.com/Hibbel2026/Cloud-based-SLR.git
%cd Cloud-based-SLR

# Cell 2: Install dependencies
!pip install opencv-python torch torchvision tqdm -q

# Cell 3-5: Rest of training...

# Cell N (Push): 
!git config --global user.email "din-email@example.com"
!git config --global user.name "Ditt Namn"
!git add checkpoints/
!git commit -m "Add 3D CNN trained model"
!git push origin Baseline-ML
```

---

## Security Notes

⚠️ **OBS:** Lagra inte tokens i koden! 

Bättre sätt:
- Lagra token i Colab Secrets (längst upp i sidebar)
- Använd `from google.colab import userdata`
- Hämta token: `token = userdata.get('GITHUB_TOKEN')`

Exempel:
```python
from google.colab import userdata

token = userdata.get('GITHUB_TOKEN')
username = "Hibbel2026"

!git clone https://{username}:{token}@github.com/Hibbel2026/Cloud-based-SLR.git
```

---

Välj Metod 1 med Personal Access Token - det är enklast! 🚀
