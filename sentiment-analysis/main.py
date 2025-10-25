import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D, Input
from tensorflow.keras.regularizers import l2

df = pd.read_csv("data.csv")
df = df[["target", "cleaned_text"]]
df["text"] = df["cleaned_text"]
df = df.drop(columns=["cleaned_text"], axis=1)
df.head()

df.shape

df.columns

df["target"].unique()

df["target"].value_counts()

# Check for Nan check
print(f"Target contains NaN {df["target"].isnull().sum()}")
print(f"Text contains NaN {df["text"].isnull().sum()}")


# target counts for null values
df[df.isnull().any(axis =1)]["target"].value_counts()

df_cleaned = df.dropna()

df_cleaned.shape

X, y = df_cleaned["text"], df_cleaned["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size : {len(X_train)} and test size {len(X_test)}")

def create_fcnn(input_shape):
    model = Sequential()
    
    model.add(Input(shape=input_shape))
    
    model.add(Dense(
        128, 
        activation='relu',
        kernel_regularizer=l2(0.001)  # L2 Regularization
    ))
    model.add(Dropout(0.5))  # Dropout
    
    model.add(Dense(
        64, 
        activation='relu',
        kernel_regularizer=l2(0.001)  # L2 Regularization
    ))
    model.add(Dropout(0.5))  # Dropout
    
    model.add(Dense(1, activation='softmax'))
    
    return model


from sklearn.feature_extraction.text import CountVectorizer

vocab_size_bow = 10000
bow_vectorizer = CountVectorizer(max_features=vocab_size_bow)

X_train_bow = bow_vectorizer.fit_transform(X_train).toarray()

X_test_bow = bow_vectorizer.transform(X_test).toarray()

bow_model = create_fcnn(input_shape=(vocab_size_bow,))
bow_model.summary()

bow_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_bow = bow_model.fit(
    X_train_bow, y_train,
    epochs=1,
    batch_size=256,
    validation_data=(X_test, y_test),
    verbose=1
)




print("-----------------    End of Program ------------------------")