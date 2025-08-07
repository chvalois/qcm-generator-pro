# Playwright Tests - Interface Comparison

Ce dossier contient les tests Playwright pour comparer les interfaces Streamlit et React du QCM Generator Pro.

## 🎯 Objectif

Les tests Playwright permettent de :
- **Validation fonctionnelle** : Vérifier que toutes les fonctionnalités marchent
- **Comparaison visuelle** : Capturer des screenshots pour comparer Streamlit vs React
- **Tests de régression** : Garantir qu'aucune fonctionnalité n'est cassée pendant la migration
- **Baseline de performance** : Mesurer les performances pour optimisation

## 🏗️ Structure

```
tests/playwright/
├── conftest.py                    # Configuration et fixtures
├── test_interface_comparison.py   # Tests de comparaison entre interfaces
├── test_streamlit_baseline.py     # Tests baseline Streamlit
├── pytest.ini                     # Configuration pytest
├── requirements.txt              # Dépendances Playwright
├── screenshots/                  # Screenshots de comparaison
├── videos/                       # Enregistrements vidéo des tests
└── reports/                      # Rapports JSON des résultats
```

## 🚀 Installation

1. **Installer les dépendances** :
```bash
pip install -r tests/playwright/requirements.txt
```

2. **Installer Playwright browsers** :
```bash
playwright install chromium
```

## ▶️ Exécution des Tests

### Tests Baseline Streamlit
```bash
# Tests baseline complets
pytest tests/playwright/test_streamlit_baseline.py -v

# Test spécifique
pytest tests/playwright/test_streamlit_baseline.py::TestStreamlitBaseline::test_navigation_structure -v
```

### Tests de Comparaison (Future)
```bash
# Quand React sera disponible
pytest tests/playwright/test_interface_comparison.py -v
```

### Tous les tests
```bash
pytest tests/playwright/ -v --html=tests/playwright/reports/report.html
```

## 📊 Résultats et Rapports

### Screenshots
Les captures d'écran sont sauvées dans `tests/playwright/screenshots/` avec la nomenclature :
- `{test_name}_{interface}_{step}_{timestamp}.png`
- Exemple : `upload_page_streamlit_2025-01-07_123456.png`

### Rapports JSON
Les métriques sont sauvées dans `tests/playwright/reports/` :
- `streamlit_navigation_baseline.json` : Structure navigation
- `streamlit_performance_baseline.json` : Métriques performance
- `streamlit_upload_baseline.json` : Fonctionnalités upload
- etc.

### Vidéos
Les enregistrements vidéo des tests sont dans `tests/playwright/videos/`

## 🔧 Configuration

### Services Requis
Les tests attendent que ces services soient démarrés :
- **FastAPI Backend** : `http://localhost:8000`
- **Streamlit Interface** : `http://localhost:8501`
- **React Interface** : `http://localhost:3000` (futur)

### Démarrage des Services
```bash
# Démarrer tous les services
make run-app

# Ou via Docker
make docker-run
```

## 🧪 Types de Tests

### 1. Tests Baseline (Actuels)
- Navigation et structure UI
- Workflow upload de documents
- Workflow génération QCM
- Fonctionnalités export
- Monitoring système
- Comportement responsive
- Métriques de performance

### 2. Tests de Comparaison (Futurs)
- Parité fonctionnelle Streamlit vs React
- Comparaisons visuelles
- Tests de performance comparative
- Workflows end-to-end identiques

### 3. Tests de Régression
- Validation que les nouvelles implémentations ne cassent rien
- Vérification de la compatibilité API
- Tests des migrations de données

## 🎨 Captures d'Écran

### Viewports Testés
- **Desktop Large** : 1920x1080
- **Desktop Medium** : 1366x768
- **Tablet Landscape** : 1024x768
- **Tablet Portrait** : 768x1024
- **Mobile** : 375x667

### Éléments Capturés
- Pages complètes (`full_page=True`)
- Éléments spécifiques lors de tests focused
- États d'erreur et de chargement
- Workflows complets en séquence

## ⚡ Performance

### Métriques Mesurées
- **Temps de chargement initial**
- **Temps de navigation** entre sections
- **Temps de réponse** des interactions
- **Utilisation mémoire** (approximation)
- **Temps de rendu** des composants

### Seuils de Performance
- Chargement initial : < 30s
- Navigation : < 3s par section
- Interactions : < 1s réponse utilisateur

## 🐛 Debugging

### Mode Debug
```bash
# Tests avec browser visible
pytest tests/playwright/ -v -s --headed

# Un seul test avec logs détaillés
pytest tests/playwright/test_streamlit_baseline.py::TestStreamlitBaseline::test_navigation_structure -v -s --log-cli-level=DEBUG
```

### Logs
Les logs sont configurés pour afficher :
- Actions Playwright
- Navigation et interactions
- Erreurs et warnings
- Métriques de performance

### Screenshots Debug
En cas d'échec, les screenshots automatiques permettent de voir l'état de la page au moment de l'erreur.

## 🔄 Intégration CI/CD

### GitHub Actions (Futur)
```yaml
- name: Run Playwright Tests
  run: |
    make run-app &
    sleep 30  # Wait for services
    pytest tests/playwright/ --html=report.html
    
- name: Upload Screenshots
  uses: actions/upload-artifact@v3
  with:
    name: playwright-screenshots
    path: tests/playwright/screenshots/
```

## 📋 Maintenance

### Mise à jour des Tests
1. **Nouveaux éléments UI** → Ajouter locators dans conftest.py
2. **Nouvelles pages** → Créer fixtures spécifiques
3. **Changements API** → Mettre à jour les attentes de réponse

### Nettoyage
```bash
# Supprimer anciens screenshots/vidéos
rm -rf tests/playwright/screenshots/*
rm -rf tests/playwright/videos/*
rm -rf tests/playwright/reports/*
```

## 🎯 Roadmap

### Phase 1 (Actuel) ✅
- Tests baseline Streamlit complets
- Infrastructure de comparaison
- Screenshots et métriques

### Phase 2 (Migration)
- Tests React dès interface disponible
- Comparaisons visuelles automatisées
- Validation parité fonctionnelle

### Phase 3 (Post-Migration)
- Tests de régression continus
- Optimisation performances
- Tests end-to-end complexes

Ce système de tests garantit une migration React sans perte de fonctionnalité et avec amélioration mesurable de l'expérience utilisateur.