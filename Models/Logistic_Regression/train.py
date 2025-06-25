from sklearn.metrics import accuracy_score,classification_report
import pandas as pd
from LogisticRegression import LogisticRegression

df=pd.read_csv("./Data/Removed_null_personality_dataset.csv")
df.drop("Unnamed: 0",axis=1,inplace=True)
y=df["Personality"]
X=df.drop("Personality",axis=1)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=42)

model=LogisticRegression(lr=0.001,n_iter=1000)
model.fit(X_train,y_train)

y_pred=model.pridect(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))