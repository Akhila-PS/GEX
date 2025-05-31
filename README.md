# GeX (Gene Expression Explorer)

## 🧬 Description
**GeX** is a bioinformatics web application for gene expression analysis and visualization. It allows users to input gene sequences and visualize expression differences between healthy and diseased samples using an interactive dashboard.

A lightweight AI model is integrated to predict whether the input gene sequence is diseased or not.

---

## 🧪 Problem Statement

- Public genomics datasets are difficult to interpret for non-experts.
- There’s a lack of simple tools to compare gene expression between healthy and diseased samples.

---

## 💡 Proposed Solution

- A user-friendly web tool that allows easy search, analysis, and visualization of differentially expressed genes from **GEO datasets**.
- Built for non-bioinformatics users with an intuitive interface and real-time visual feedback.

---

## ⚙️ Tech Stack

- **Frontend**: HTML, CSS,JS
- **Backend**: Python (Fastapi)
- **Libraries**: GEOparse, Pandas, SciPy
- **Visualization**: Plotly, Chart.js
- **Data Source**: NCBI GEO

---

## 🌟 Key Features

1. **Gene Query & Search**: View gene expression across healthy and diseased samples.
2. **Differential Expression Analysis**: Identify up/downregulated genes using fold change and p-value.
3. **Interactive Charts**: Visualize data using box plots, heatmaps, and scatter plots.
4. **Expression Filters**: Filter results by expression threshold.
5. **AI-Based Prediction**: Predict whether an input gene is healthy or diseased.
6. **Export Results** *(Planned)*: CSV/PDF download support.
7. **Multiple Dataset Comparison** *(Planned)*: Upload and compare more than one dataset.
8. **Gene Metadata Integration** *(Planned)*: Link genes with databases like Ensembl or UniProt.

---

## 🧭 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Akhila-PS/GEX
   cd Hack-4-mini
  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt

3. Run the backend
   ```bash
   python main.py
4. Open the Dashboard
   ```bash
   open dash.html

Hack-4-mini/
│
├── main.py              # Backend logic
├── dash.html            # Dashboard interface
├── css/                 # Styling files
├── js/                  # JavaScript files
├── setup.py             # For dependency installation
└── README.md



![Screenshot 2025-05-31 130316](https://github.com/user-attachments/assets/5b437425-b454-45cf-b098-9e380c2e007f)
![Screenshot 2025-05-31 130426](https://github.com/user-attachments/assets/40e1f0ef-d096-4993-abb8-30cad6f60d7f)
![Screenshot 2025-05-31 130332](https://github.com/user-attachments/assets/719c6041-fbad-48a6-919d-56284b05ff0d)
![Screenshot 2025-05-31 101833](https://github.com/user-attachments/assets/2d1f952a-470e-43bf-8d38-720194489ac5)
