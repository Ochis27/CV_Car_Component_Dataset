#!/bin/bash

# Project setup script for CV_Project
# Creates folder structure and virtual environment

echo "🚀 Setting up CV_Project..."

# Create main folders
mkdir -p Component_Images
mkdir -p src
mkdir -p datasets/first5_multi
mkdir -p outputs

# Create Python virtual environment
python3 -m venv .venv
echo "✅ Virtual environment created"

# Activate and install dependencies
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python numpy pillow

echo "✅ Dependencies installed"

# Create .gitignore
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
.venv/
venv/

# macOS
.DS_Store

# Generated outputs
outputs/
datasets/

# Keep locally, don't upload
Component_Images/
Component_Images.zip
EOF

echo "✅ .gitignore created"

# Create requirements.txt
cat > requirements.txt <<'EOF'
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
EOF

echo "✅ requirements.txt created"

echo ""
echo "📁 Project structure created:"
echo "   Component_Images/  ← Place your 1.jpeg...5.jpeg here"
echo "   src/               ← Python scripts"
echo "   datasets/          ← Generated crops"
echo "   outputs/           ← Final outputs"
echo ""
echo "🎯 Next steps:"
echo "   1. Place images 1.jpeg...5.jpeg in Component_Images/"
echo "   2. Activate venv: source .venv/bin/activate"
echo "   3. Run: python3 src/extract_first5_components.py"