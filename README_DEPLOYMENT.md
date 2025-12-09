# 🚀 Deployment Guide - Free Dashboard Hosting

This guide shows you how to deploy your Customer Behavior Prediction dashboard for **FREE** on various platforms.

## 🎯 Best Free Options for Data Dashboards

### 1. **Streamlit Cloud** ⭐ (Recommended)
**Best for:** Streamlit dashboards, easiest setup

**Why Streamlit Cloud?**
- ✅ 100% Free forever
- ✅ Auto-deploy from GitHub
- ✅ No credit card required
- ✅ Automatic HTTPS
- ✅ Public and private apps
- ✅ Easy sharing with URL

**Deployment Steps:**
1. Push your code to GitHub (already done!)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository: `iNSRawat/customer-behavior-prediction`
6. Select branch: `main`
7. Main file path: `dashboard.py`
8. Click "Deploy"
9. Wait 2-3 minutes
10. Your dashboard is live! 🎉

**Required Files:**
- ✅ `dashboard.py` (already created)
- ✅ `requirements.txt` (with streamlit added)
- ✅ Data files in `data/` folder
- ✅ Model file in `models/` (after running training)

---

### 2. **Hugging Face Spaces** ⭐
**Best for:** ML demos, supports Streamlit and Gradio

**Why Hugging Face?**
- ✅ Free GPU/CPU resources
- ✅ Great for ML projects
- ✅ Auto-deploy from Git
- ✅ Community features

**Deployment Steps:**
1. Create account at [huggingface.co](https://huggingface.co)
2. Create a new Space
3. Choose SDK: **Streamlit**
4. Connect your GitHub repo
5. Set `dashboard.py` as main file
6. Deploy!

**File Structure Needed:**
```
app.py (rename dashboard.py to app.py)
requirements.txt
data/
models/
```

---

### 3. **Render**
**Best for:** Full-stack apps, auto-scaling

**Why Render?**
- ✅ Free tier available
- ✅ Auto-deploy from GitHub
- ✅ Custom domains
- ⚠️ Free tier has limitations (spins down after inactivity)

**Deployment Steps:**
1. Sign up at [render.com](https://render.com)
2. Click "New Web Service"
3. Connect GitHub repo
4. Service type: **Web Service**
5. Build command: `pip install -r requirements.txt`
6. Start command: `streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0`
7. Deploy!

---

### 4. **PythonAnywhere**
**Best for:** Simple hosting, Jupyter notebooks

**Why PythonAnywhere?**
- ✅ Free beginner account
- ✅ Easy setup
- ✅ Supports Flask/Streamlit
- ⚠️ Limited resources on free tier

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure:

- [x] All dependencies in `requirements.txt`
- [x] `dashboard.py` is in root directory
- [x] Data files are committed to repo (or use external storage)
- [x] Model file exists after training
- [x] All paths are relative (not absolute)

## 🔧 Quick Setup for Streamlit Cloud

1. **Update requirements.txt:**
```bash
# Already done - streamlit added!
```

2. **Create streamlit config (optional):**
Create `.streamlit/config.toml`:
```toml
[server]
headless = true
port = 8501

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

3. **Test locally first:**
```bash
pip install streamlit
streamlit run dashboard.py
```

4. **Deploy to Streamlit Cloud:**
   - Go to share.streamlit.io
   - Connect GitHub
   - Select repo and file
   - Deploy!

## 🎨 Dashboard Features

The `dashboard.py` includes:
- 📈 Overview with key metrics
- 🔍 Interactive data exploration
- 🤖 Model performance visualization
- 📊 Prediction interface
- 📉 Feature importance charts

## 🔗 Quick Links

- **Streamlit Cloud:** [share.streamlit.io](https://share.streamlit.io)
- **Hugging Face:** [huggingface.co/spaces](https://huggingface.co/spaces)
- **Render:** [render.com](https://render.com)
- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)

## 💡 Tips

1. **For Streamlit Cloud:** Your app URL will be like:
   `https://your-app-name.streamlit.app`

2. **Data Size:** Keep data files < 100MB for free tiers

3. **Model Size:** Consider using `.joblib` compression for large models

4. **Secrets:** Use Streamlit secrets for API keys (if needed)

5. **Performance:** Use `@st.cache_data` and `@st.cache_resource` decorators

## 🆘 Troubleshooting

**Issue:** Import errors in deployed app
- **Solution:** Ensure all dependencies in `requirements.txt`

**Issue:** Data file not found
- **Solution:** Check file paths are relative, files are in repo

**Issue:** Model file not found
- **Solution:** Run training script first, commit model file

**Issue:** App is slow
- **Solution:** Use caching decorators, optimize data loading

---

## 🎉 Recommended: Streamlit Cloud

**Easiest and fastest option!**

1. Your code is already on GitHub ✅
2. Just connect at share.streamlit.io
3. Deploy in 2 minutes
4. Share your dashboard URL!

**Your dashboard will be live at:**
`https://customer-behavior-prediction.streamlit.app`

(URL format: `https://your-repo-name.streamlit.app`)

