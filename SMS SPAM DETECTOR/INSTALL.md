# Installation Guide for SMS Spam Detection

## Quick Install (Recommended)

Open a **new Command Prompt or Terminal** (not in Jupyter) and run:

```bash
pip install wordcloud
```

If that doesn't work, try:

```bash
python -m pip install wordcloud
```

## Install All Dependencies at Once

To install all required packages:

```bash
pip install -r requirements.txt
```

Or install them individually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk jupyter
```

## Troubleshooting

### If pip is not recognized:
1. Make sure Python is added to your PATH
2. Try using `python -m pip` instead of `pip`

### If you get permission errors:
Add `--user` flag:
```bash
pip install wordcloud --user
```

### If installation is slow:
The wordcloud package may take a few minutes to install as it has compiled components.

## After Installation

1. **Restart Jupyter Notebook kernel**:
   - In Jupyter: Click "Kernel" → "Restart"
   
2. **Run the first cell again** to import all libraries

3. **Continue with the rest of the notebook**

## Alternative: Run Without Word Clouds

If you want to skip the word cloud visualizations, you can comment out those sections in the notebook:
- Comment out the `from wordcloud import WordCloud` import
- Comment out or skip the word cloud visualization cells

The rest of the analysis and models will work perfectly fine without word clouds!
