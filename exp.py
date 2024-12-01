import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

data = pd.read_csv("annclassification/Churn_Modelling.csv")
print(data.head())

## Data preprocessing
# Drop irrelebant columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
print(data.head())

## Encode Categorical Variables
label_encoder_gender = LabelEncoder()
data["Gender"] = label_encoder_gender.fit_transform(data["Gender"])
print(data)

## Onehot encode Geography
onehot_encoder_geo = OneHotEncoder()
geo_encoder = onehot_encoder_geo.fit_transform(data[["Geography"]])
print(geo_encoder)
print(onehot_encoder_geo.get_feature_names_out(['Geography']))

geo_df = pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
print(geo_df)

## Combain OHE columns Geo
data = pd.concat([data.drop("Geography", axis=1), geo_df], axis=1)
print(data.head())


# ## Save Encoding
# with open('label_encoder_gender.pkl', 'wb') as file:
#     pickle.dump(label_encoder_gender, file)

# with open('onehot_encoder_geo.pkl', 'wb') as file:
#     pickle.dump(onehot_encoder_geo, file)

Y = data["Exited"]
X = data.drop("Exited", axis=1)

## Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

## Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print(X_train)

# ## Save scalar in pickle
# with open('scaler.pkl', 'wb') as file:
#     pickle.dump(scaler, file)

##### ANN #####
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, TensorBoard
import datetime
import keras

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), ## HL1 -> input layer
    Dense(32, activation='relu'),   ## HL2
    Dense(1, activation='sigmoid')  ## Output layer
])

print(model.summary())

opt = keras.optimizers.Adam(learning_rate=0.01)
# loss = keras.losses.BinaryCrossentropy()
## Compile the Model
model.compile(
    optimizer=opt, loss="binary_crossentropy", 
    metrics=['accuracy']
)

log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

## Earlystopping
early_stopping_callback = EarlyStopping(
    monitor='val_loss', patience=5, 
    restore_best_weights=True
)

### Model Training
history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=100,
    callbacks=[tensorflow_callback, early_stopping_callback]
)