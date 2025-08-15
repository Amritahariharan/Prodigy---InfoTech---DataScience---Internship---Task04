from textblob import TextBlob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

reviews = [
    "I love this product! It's amazing.",
    "Worst service ever. Totally disappointed.",
    "It's okay, not bad but not great.",
    "Fantastic quality and great value.",
    "Terrible experience. Will not return.",
    "The delivery was super fast and well packaged.",
    "Customer support was rude and unhelpful.",
    "This is the best purchase I've made this year.",
    "Mediocre performance for the price.",
    "Excellent durability and beautiful design."
]

# Create DataFrame
df = pd.DataFrame(reviews, columns=['review'])

# Analyze sentiment
df['polarity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df['sentiment'] = df['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Display results
print(df)

# Sentiment Count Plot
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df, palette='Set2')
plt.title("Sentiment Distribution")
plt.show()

# Polarity Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['polarity'], bins=10, kde=True, color='orange')
plt.title("Polarity Score Distribution")
plt.xlabel("Polarity")
plt.ylabel("Frequency")
plt.show()
