#!/usr/bin/env python3
"""
Create a simple test PDF for testing the upload functionality.
"""

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

def create_test_pdf():
    """Create a simple test PDF."""
    if not REPORTLAB_AVAILABLE:
        print("❌ ReportLab not available. Install with: pip install reportlab")
        return False
    
    filename = "test_document.pdf"
    
    # Create PDF
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Add content
    c.setFont("Helvetica", 16)
    c.drawString(50, height - 50, "QCM Generator Test Document")
    
    c.setFont("Helvetica", 12)
    
    # Add some content about programming
    content = [
        "",
        "Introduction à Python",
        "",
        "Python est un langage de programmation interprété, multi-paradigme et",
        "multiplateformes. Il favorise la programmation impérative structurée,",
        "fonctionnelle et orientée objet.",
        "",
        "Caractéristiques principales:",
        "• Syntaxe claire et lisible",
        "• Typage dynamique fort",
        "• Gestion automatique de la mémoire",
        "• Vaste bibliothèque standard",
        "",
        "Structures de données:",
        "Les listes, tuples, dictionnaires et ensembles sont les structures",
        "de données de base en Python. Elles permettent de stocker et organiser",
        "les informations de manière efficace.",
        "",
        "Programmation orientée objet:",
        "Python supporte la programmation orientée objet avec des classes,",
        "l'héritage, l'encapsulation et le polymorphisme.",
    ]
    
    y = height - 100
    for line in content:
        c.drawString(50, y, line)
        y -= 20
        if y < 50:  # New page if needed
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50
    
    # Add second page with more content
    c.showPage()
    c.setFont("Helvetica", 16)
    c.drawString(50, height - 50, "Concepts Avancés")
    
    c.setFont("Helvetica", 12)
    
    advanced_content = [
        "",
        "Gestion des exceptions:",
        "Python utilise le mécanisme try/except pour gérer les erreurs",
        "et exceptions qui peuvent survenir lors de l'exécution.",
        "",
        "Modules et packages:",
        "Les modules permettent d'organiser le code en fichiers séparés,",
        "tandis que les packages regroupent plusieurs modules.",
        "",
        "Tests unitaires:",
        "Python fournit le module unittest pour créer et exécuter",
        "des tests automatisés du code.",
        "",
        "Bonnes pratiques:",
        "• Suivre les conventions PEP 8",
        "• Documenter le code avec des docstrings",
        "• Utiliser des noms de variables explicites",
        "• Éviter la duplication de code",
    ]
    
    y = height - 100
    for line in advanced_content:
        c.drawString(50, y, line)
        y -= 20
    
    c.save()
    
    print(f"✅ Test PDF created: {filename}")
    return True

if __name__ == "__main__":
    create_test_pdf()