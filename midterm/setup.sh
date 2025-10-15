#!/bin/bash
# Script de setup automatizado para el proyecto Midterm
# Uso: bash setup.sh

set -e  # Exit on error

echo "=================================================="
echo "Setup: GNN-based Recommender Systems - Midterm"
echo "=================================================="

# Check Python version
echo ""
echo "[1/5] Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 no encontrado"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "✓ $PYTHON_VERSION encontrado"

# Create virtual environment
echo ""
echo "[2/5] Creando virtual environment..."
if [ -d "venv" ]; then
    echo "⚠ venv ya existe, saltando..."
else
    python3 -m venv venv
    echo "✓ Virtual environment creado"
fi

# Activate venv
echo ""
echo "[3/5] Activando venv..."
source venv/bin/activate
echo "✓ venv activado"

# Upgrade pip
echo ""
echo "[4/5] Actualizando pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip actualizado"

# Install dependencies
echo ""
echo "[5/5] Instalando dependencias..."
echo "Esto puede tomar varios minutos..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "=================================================="
echo "Verificando instalación..."
echo "=================================================="

python3 << 'EOF'
import sys
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")

    import torch_geometric
    print(f"✓ PyTorch Geometric {torch_geometric.__version__}")

    import numpy
    print(f"✓ NumPy {numpy.__version__}")

    import pandas
    print(f"✓ Pandas {pandas.__version__}")

    import scipy
    print(f"✓ SciPy {scipy.__version__}")

    import sentence_transformers
    print(f"✓ Sentence Transformers {sentence_transformers.__version__}")

    print("\n✅ Todas las dependencias instaladas correctamente!")

except ImportError as e:
    print(f"\n❌ Error al importar: {e}")
    sys.exit(1)
EOF

# Final instructions
echo ""
echo "=================================================="
echo "✅ Setup completado exitosamente!"
echo "=================================================="
echo ""
echo "Próximos pasos:"
echo ""
echo "1. Activar el venv (cada vez que trabajes):"
echo "   source venv/bin/activate"
echo ""
echo "2. Ejecutar tests de verificación:"
echo "   python test_pipeline.py"
echo ""
echo "3. Construir grafos (si no existen):"
echo "   python build_graphs.py"
echo ""
echo "4. Probar demos interactivas:"
echo "   python demo_ltm.py"
echo ""
echo "Ver README.md para más información."
echo ""
