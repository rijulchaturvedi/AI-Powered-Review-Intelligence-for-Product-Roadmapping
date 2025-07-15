# priority_matrix.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load analyzed review data
with open("data/analyzed_reviews.pkl", "rb") as f:
    data = pickle.load(f)
    df = data['df']
    topic_labels = data.get('topic_labels')

# Remove outlier/noise topics like -1
df = df[df['topic'] != -1]

# Aggregate
topic_summary = df.groupby('topic').agg({
    'num_reviews': 'count',
    'sentiment': 'mean'
}).reset_index()


# Assign labels from topic_labels if available
if topic_labels:
    topic_summary['Label'] = topic_summary['topic'].map(lambda t: topic_labels[t] if isinstance(topic_labels[t], str) else ', '.join(topic_labels[t][:3]))
else:
    topic_summary['Label'] = topic_summary['topic'].astype(str)

# Generate color palette
unique_labels = topic_summary['Label'].unique()
palette = dict(zip(unique_labels, sns.color_palette('hsv', len(unique_labels))))

plt.figure(figsize=(12, 7))
scatter = sns.scatterplot(
    data=topic_summary,
    x='num_reviews',
    y='sentiment',
    size='num_reviews',
    sizes=(40, 600),
    hue='Label',
    palette=palette,
    alpha=0.8,
    edgecolor='black',
    linewidth=0.5
)

from matplotlib.lines import Line2D

top_labels = topic_summary.nlargest(10, 'num_reviews')
handles = [Line2D([0], [0], marker='o', color='w',
                  label=str(row['topic']),
                  markerfacecolor=palette[row['Label']],
                  markersize=10) for _, row in top_labels.iterrows()]
plt.legend([], [], frameon=False)  # Remove legend clutter

# Annotate each point with its label
for _, row in top_labels.iterrows():
    plt.text(row['num_reviews'] + 8, row['sentiment'] + 0.01, row['Label'], fontsize=9, weight='bold')

x_mean = topic_summary['num_reviews'].mean()
y_mean = topic_summary['sentiment'].mean()

top_recommendations = topic_summary[
    (topic_summary['num_reviews'] > x_mean) & (topic_summary['sentiment'] > y_mean)
].sort_values(by=['num_reviews', 'sentiment'], ascending=False).head(5)

# Annotate just top 5 high priority points directly with topic number
for _, row in top_recommendations.iterrows():
    plt.text(row['num_reviews'] + 10, row['sentiment'] + 0.01,
             f"Topic {int(row['topic'])}",
             fontsize=10, weight='bold', color='black', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

plt.title("Prioritization Matrix: Feature Frequency vs Sentiment")
plt.xlabel("Number of Reviews (Feature Frequency)")
plt.ylabel("Average Sentiment")
plt.grid(True)
plt.axhline(y_mean, color='gray', linestyle='--', linewidth=1)
plt.axvline(x_mean, color='gray', linestyle='--', linewidth=1)

plt.text(x_mean + 50, y_mean + 0.05, '🔥 High Frequency, High Sentiment', fontsize=10, fontweight='bold')
plt.text(x_mean - 200, y_mean + 0.05, '💡 Low Frequency, High Sentiment', fontsize=10, fontweight='bold')
plt.text(x_mean + 50, y_mean - 0.15, '⚠️ High Frequency, Low Sentiment', fontsize=10, fontweight='bold')
plt.text(x_mean - 200, y_mean - 0.15, '🧊 Low Frequency, Low Sentiment', fontsize=10, fontweight='bold')

plt.tight_layout()
output_path = "prioritization_matrix.png"
plt.savefig(output_path)
print(f"Saved prioritization matrix to: {output_path}")
topic_summary.to_csv("topic_summary.csv", index=False)
print("Saved topic summary to: topic_summary.csv")

print("\nTop 5 High Priority Topics (🔥):")
for _, row in top_recommendations.iterrows():
    label = row['topic']
    print(f"- Topic {label} (Reviews: {row['num_reviews']}, Sentiment: {row['sentiment']:.2f})")
plt.show()