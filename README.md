app website: https://sentimental-analysis-on-amazon-by-harsh.streamlit.app

🖱️ How to Use the App

Single Review Analysis

1. Go to the “✍️ Single Review Analysis” tab.
2. Paste or type an Amazon review in the text area.
3. Click “Analyze 🔎”.
4. The app will show:
◦  Positive or Negative sentiment.
◦  A visual badge (green for positive, red for negative).
◦  Confidence bar (if probability is available).

Batch Review Analysis (CSV)

1. Go to the “📄 Batch Review Analysis” tab.
2. (Optional) Download the CSV template to see the expected format.
3. Upload your own CSV file:
◦  File should contain at least one column with review text.
4. Select the review text column from the dropdown.
5. Click “▶️ Run Batch Analysis”:
◦  The app will output:
▪  Prediction for each row in a new column: “Predicted Sentiment”.
▪  Summary metrics: total, number of positive, number of negative.
▪  A pie chart of sentiment distribution.
▪  Option to download the predictions CSV.



⚙️ Model & Training Details

High‑level training pipeline (see sentimental_analysis(2).ipynb for full code):

•  Libraries: pandas, numpy, nltk, scikit-learn, seaborn, etc.
•  Dataset:
◦  Amazon product reviews with columns like:
▪  Review – review text.
▪  Sentiment – integer 1–5.
◦  Label mapping:
▪  Sentiment <= 3 → 0 (Negative).
▪  Sentiment > 3 → 1 (Positive).
•  Text preprocessing:
◦  Lowercasing, punctuation removal, basic stopword filtering (via NLTK).
◦  Character length analysis for extremely short and extremely long reviews.
•  Vectorization & model:
◦  CountVectorizer for bag‑of‑words representation.
◦  MultinomialNB (or similar) classifier.
•  Evaluation:
◦  Train/test split.
◦  Metrics: accuracy, precision, recall, F1, confusion matrix.
•  Export:
◦  Save trained vectorizer as vectorizer.pkl.
◦  Save trained model as classifier.pkl.



🧭 Roadmap / Ideas

Based on things to do.txt and potential improvements:

•  Explore when to use which model (e.g. Naive Bayes vs. Logistic Regression vs. Transformers).
•  Add documentation on each step of the ML lifecycle in this project:
◦  Data collection → EDA → preprocessing → feature engineering → modeling → evaluation → deployment.
•  Compare advantages & limitations of different models and document them in the repo.
•  Add tests for the preprocessing pipeline and prediction functions.
•  Add CI/CD for deployment to Streamlit Cloud or other platforms.
•  Improve short‑text handling & emoji / non‑English support.



📝 License

Specify your license here, for example:

> This project is licensed under the MIT License – see the LICENSE file for details.

(If you haven’t chosen a license yet, you can remove this section or add one later.)



If you tell me your GitHub repo name and preferred license, I can tailor the top section and the clone URL to match exactly.
