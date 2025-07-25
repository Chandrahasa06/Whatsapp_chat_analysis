## 📊 Chat Insights – WhatsApp Chat Analyzer

Unlock the secrets hidden in your conversations with Chat Insights – a user-friendly and powerful Streamlit-based app that helps you analyze WhatsApp chat data. Visualize user contributions, word clouds, emoji usage, sentiment dynamics, and more.

---

### 🚀 Features

* 📤 Upload WhatsApp chat files
* 🡭‍♂️ User-wise message breakdown
* ⏰ Activity timelines and heatmaps
* 📈 Comparative analysis between users

---

### 🛠️ Technologies Used

| Tech                        | Description                                  |
| --------------------------- | -------------------------------------------- |
| **Python**                  | Core programming language                    |
| **Streamlit**               | Web UI framework for the app                 |
| **Pandas**                  | Data processing and analysis                 |
| **Matplotlib & Seaborn**    | Data visualizations                          |
| **Tailwind CSS (via HTML)** | Landing page styling                         |
| **Custom CSS**              | Component styling (dropdown, headings, etc.) |

---

### 📁 Project Structure

```
chat-insights/
│
├── app.py                # Main Streamlit app
├── preprocessor.py       # Data preprocessing 
├── helper.py             # Data preprocessing and plotting functions
├── style.css             # Custom CSS styles
├── README.md             # You’re reading it!
├── requirements.txt      # Python dependencies
```

---

### ⚙️ Setup Instructions

#### 1. **Clone the repository**

```bash
git clone https://github.com/your-username/chat-insights.git
cd chat-insights
```

#### 2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

#### 3. **Install dependencies**

```bash
pip install -r requirements.txt
```

> 📌 If you don’t have a `requirements.txt`, you can create it like:

```bash
pip freeze > requirements.txt
```

#### 4. **Run the Streamlit app**

```bash
streamlit run app.py
```

The app will launch in your browser at: [http://localhost:8501](http://localhost:8501)

---

### 📷 Screenshots

![Screenshot 2025-06-30 184441](https://github.com/user-attachments/assets/6c93ca33-f54e-45cb-bb13-4c1bd05f8383)

![image](https://github.com/user-attachments/assets/620b8d18-8c70-45e3-8a1a-ee359ceed8d7)

Click on any of the options and start analysing

![Screenshot 2025-06-30 184505](https://github.com/user-attachments/assets/e666b735-a288-4642-8637-d886be47af19)


