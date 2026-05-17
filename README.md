# ai-cyber-security-platform
🛡️ AI-Powered Cyber Security Platform
A multi-module AI-based cyber security dashboard built with Python and Streamlit. It detects threats, classifies malicious URLs, analyses QR codes, detects phishing emails, and performs digital forensics — all in one web application.

📌 Features

🔴 **Threat Detection**
Detects anomalies in network traffic using Isolation Forest.

---

🔍 **Digital Forensics**
Analyses incident logs — timeline, users, IPs, failed logins.

---

🌐 **URL / IP Classifier**
Classifies malicious URLs and IPs using a CNN model.

---

📷 **QR Code Detector**
Scans QR code payloads for hidden malicious links.

---

📧 **Phishing Detector**
Detects phishing emails using TF-IDF + Logistic Regression.

---

📄 **PDF Report**
Generates downloadable incident reports via ReportLab.

---

🗄️ **Alert Storage**
Saves all threat results to a local SQLite database.

🧰 Tech Stack

Frontend — Streamlit (cyberpunk CSS theme)
ML Models — Scikit-learn, TensorFlow / Keras
Data — Pandas, NumPy
Storage — SQLite3
PDF — ReportLab
Language — Python 3.x
