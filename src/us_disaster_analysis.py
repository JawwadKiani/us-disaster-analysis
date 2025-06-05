# Full Advanced Disaster Analytics Script

import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.tsa.seasonal import STL
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob  # New import for sentiment analysis

# 1. Download and load the dataset
path = kagglehub.dataset_download("headsortails/us-natural-disaster-declarations")
for file in os.listdir(path):
    if file.endswith('.csv'):
        data_path = os.path.join(path, file)
        df = pd.read_csv(data_path)
        break

# 2. Basic Exploration
print(df.head())
print(df.info())
print(df.describe())
print(df['incident_type'].value_counts())

# 3. Time Series Forecasting
print("\n=== Time Series Analysis ===")
df['date'] = pd.to_datetime(df['declaration_date'])
df.set_index('date', inplace=True)
monthly = df.resample('M').size()

plt.figure(figsize=(10, 6))
monthly.plot(title='Monthly Disaster Declarations')
plt.ylabel("Number of Declarations")
plt.grid(True)
plt.tight_layout()
plt.savefig("monthly_disaster_declarations.png")
plt.show()

model = ARIMA(monthly, order=(1, 1, 1))
results = model.fit()
forecast = results.forecast(steps=12)

plt.figure(figsize=(10, 6))
monthly.plot(label='Historical')
forecast.plot(label='Forecast', legend=True)
plt.title('Forecast of Disaster Frequency')
plt.xlabel('Date')
plt.ylabel('Number of Declarations')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("disaster_frequency_forecast.png")
plt.show()

# STL Decomposition
stl = STL(monthly)
res = stl.fit()
res.plot()
plt.title("STL Decomposition of Disaster Frequency")
plt.savefig("stl_decomposition.png")
plt.show()

# 4. Duration Analysis (Survival)
print("\n=== Survival Analysis ===")
df['incident_begin_date'] = pd.to_datetime(df['incident_begin_date'])
df['incident_end_date'] = pd.to_datetime(df['incident_end_date'])
df['duration_days'] = (df['incident_end_date'] - df['incident_begin_date']).dt.days

df['duration_days'] = df['duration_days'].fillna(0)
df['duration_days'] = df['duration_days'].apply(lambda x: x if x >= 0 else 0)

plt.figure(figsize=(10, 6))
df['duration_days'].hist(bins=50)
plt.title('Distribution of Disaster Durations')
plt.xlabel('Duration (Days)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig("disaster_duration_distribution.png")
plt.show()

# 5. Incident Type Trend Over Time
print("\n=== Incident Type Trends ===")
df['incident_type'] = df['incident_type'].astype(str)
type_monthly = df.groupby([pd.Grouper(freq='M'), 'incident_type']).size().unstack().fillna(0)

plt.figure(figsize=(15, 8))
type_monthly.plot()
plt.title('Monthly Trends of Disaster Types')
plt.ylabel('Number of Declarations')
plt.grid(True)
plt.tight_layout()
plt.savefig("incident_type_trends.png")
plt.show()

# 6. Program Declarations Analysis
print("\n=== Program Declarations Analysis ===")
program_cols = ['ih_program_declared', 'ia_program_declared', 'pa_program_declared', 'hm_program_declared']
program_sum = df[program_cols].sum()

plt.figure(figsize=(8, 6))
program_sum.plot(kind='bar')
plt.title('Frequency of Disaster Program Declarations')
plt.ylabel('Number of Declarations')
plt.grid(True)
plt.tight_layout()
plt.savefig("program_declarations_bar.png")
plt.show()

# 7. Top States by Declarations
print("\n=== Top 10 States with Most Disasters ===")
top_states = df['state'].value_counts().head(10)

plt.figure(figsize=(8, 6))
top_states.plot(kind='bar')
plt.title('Top 10 States by Disaster Declarations')
plt.ylabel('Number of Declarations')
plt.grid(True)
plt.tight_layout()
plt.savefig("top_states_disasters.png")
plt.show()

# 8. Correlation Matrix (for numeric fields)
print("\n=== Correlation Matrix ===")
plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Numerical Columns')
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

# 9. Clustering Analysis
print("\n=== Clustering Disaster Events ===")
features = df[['duration_days'] + program_cols]
features_scaled = StandardScaler().fit_transform(features)
kmeans = KMeans(n_clusters=4, random_state=42).fit(features_scaled)
df['cluster'] = kmeans.labels_

sns.countplot(data=df, x='cluster')
plt.title('Disaster Clusters')
plt.tight_layout()
plt.savefig("disaster_clusters.png")
plt.show()

# 10. Predicting Severity of Disasters
print("\n=== Severity Prediction ===")
df['severity'] = (df['duration_days'] > df['duration_days'].median()).astype(int)
X = df[program_cols]
y = df['severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = RandomForestClassifier().fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 11. Geospatial Mapping
print("\n=== Disaster Mapping ===")
state_counts = df['state'].value_counts().reset_index()
state_counts.columns = ['state', 'count']
fig = px.choropleth(state_counts,
                    locations='state',
                    locationmode="USA-states",
                    color='count',
                    scope="usa",
                    title="Disaster Declarations by State")
fig.write_html("us_disaster_map.html")
fig.show()

# 12. Natural Language Summary
max_state = df['state'].value_counts().idxmax()
longest_disaster = df[df['duration_days'] == df['duration_days'].max()]
print(f"The state with most declarations is {max_state}.")
print(f"The longest disaster lasted {longest_disaster['duration_days'].values[0]} days in {longest_disaster['state'].values[0]}.")

# 13. Monthly Variance & Anomalies
monthly_var = monthly.rolling(window=3).std()
plt.figure(figsize=(10, 5))
monthly_var.plot()
plt.title("Rolling Standard Deviation of Disaster Counts")
plt.grid(True)
plt.tight_layout()
plt.savefig("monthly_variance.png")
plt.show()

# 14. Program Declaration Combinations
df['program_combo'] = df[program_cols].astype(str).agg('-'.join, axis=1)
combo_counts = df['program_combo'].value_counts().head(10)
plt.figure(figsize=(10, 6))
combo_counts.plot(kind='bar')
plt.title('Top 10 Program Declaration Combinations')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("top_program_combinations.png")
plt.show()

# 15. Temporal Shift in Disaster Type Prevalence
early = df[df.index.year < 2010]['incident_type'].value_counts(normalize=True)
later = df[df.index.year >= 2010]['incident_type'].value_counts(normalize=True)
combined = pd.DataFrame({'Before 2010': early, '2010 and After': later}).fillna(0)
combined.plot(kind='bar', figsize=(14, 6), title="Disaster Type Distribution Before and After 2010")
plt.tight_layout()
plt.savefig("disaster_type_shift.png")
plt.show()

# 16. Economic Impact Regression (if available)
if 'total_damage' in df.columns:
    df['total_damage'] = pd.to_numeric(df['total_damage'], errors='coerce')
    df.dropna(subset=['total_damage'], inplace=True)
    reg = LinearRegression()
    reg.fit(df[['duration_days']], df['total_damage'])
    print("Damage Prediction Slope:", reg.coef_[0])
    print("Damage Prediction Intercept:", reg.intercept_)

# 17. Topic Modeling from Notes (if available)
if 'declaration_title' in df.columns:
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df['declaration_title'].fillna(""))
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(doc_term_matrix)
    for i, topic in enumerate(lda.components_):
        top_words = [vectorizer.get_feature_names_out()[j] for j in topic.argsort()[-10:]]
        print(f"Topic {i+1}:", ", ".join(top_words))

# 18. Sentiment Analysis of Declaration Titles
if 'declaration_title' in df.columns:
    df['title_sentiment'] = df['declaration_title'].fillna('').apply(lambda x: TextBlob(x).sentiment.polarity)
    plt.figure(figsize=(10, 6))
    df['title_sentiment'].hist(bins=50)
    plt.title('Sentiment Polarity of Disaster Declaration Titles')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("title_sentiment_distribution.png")
    plt.show()
