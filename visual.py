from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head() #loading dataset
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 10))
corr = df.corr()

# Plot heatmap with annotations and a color bar
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, cbar=True)

plt.title('Correlation Heatmap of Features', fontsize=16)
plt.show()