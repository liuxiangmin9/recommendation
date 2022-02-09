import tensorflow as tf
import pandas as pd

# tf dataset方式读取数据
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        num_epochs=1,
        ignore_errors=True)
    return dataset

# 数据将"income_bracket"预先转为"label"
training_samples_file_path = tf.keras.utils.get_file("census_train.csv", "file:///Census/train1.csv")
test_samples_file_path = tf.keras.utils.get_file("census_test.csv", "file:///Census/test1.csv")

train_dataset = get_dataset(training_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)

train_data = pd.read_csv("./Census/train.csv")

# 输入
inputs = {'age': tf.keras.layers.Input(name='age', shape=(), dtype='int32'),
          'education_num': tf.keras.layers.Input(name='education_num', shape=(), dtype='int32'),
          'capital_gain': tf.keras.layers.Input(name='capital_gain', shape=(), dtype='int32'),
          'capital_loss': tf.keras.layers.Input(name='capital_loss', shape=(), dtype='int32'),
          'hours_per_week': tf.keras.layers.Input(name='hours_per_week', shape=(), dtype='int32'),
          'workclass': tf.keras.layers.Input(name='workclass', shape=(), dtype='string'),
          'education': tf.keras.layers.Input(name='education', shape=(), dtype='string'),
          'marital_status': tf.keras.layers.Input(name='marital_status', shape=(), dtype='string'),
          'occupation': tf.keras.layers.Input(name='occupation', shape=(), dtype='string'),
          'relationship': tf.keras.layers.Input(name='relationship', shape=(), dtype='string'),
          'race': tf.keras.layers.Input(name='race', shape=(), dtype='string'),
          'gender': tf.keras.layers.Input(name='gender', shape=(), dtype='string'),
          'native_country': tf.keras.layers.Input(name='native_country', shape=(), dtype='string')}

# 类别特征列和连续型特征列
categorical_col_names = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "gender", "native_country"]

categorical_columns = []
categorical_columns_raw = []
for col in categorical_col_names:
    vocab = list(set(train_data[col]))
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=col, vocabulary_list=vocab)
    categorical_columns_raw.append(cat_col)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

continuous_columns = [tf.feature_column.bucketized_column(tf.feature_column.numeric_column('age'), [20, 30, 40, 50]),
                      tf.feature_column.numeric_column('education_num'),
                      tf.feature_column.numeric_column('capital_gain'),
                      tf.feature_column.numeric_column('capital_loss'),
                      tf.feature_column.numeric_column('hours_per_week')]


# 特征交叉
crossed_feature = tf.feature_column.indicator_column(tf.feature_column.crossed_column(categorical_columns_raw, 1000))

# wide and deep模型
# 构造deep部分
deep = tf.keras.layers.DenseFeatures(continuous_columns + categorical_columns)(inputs)
deep = tf.keras.layers.Dense(512, activation='relu')(deep)
deep = tf.keras.layers.Dense(256, activation='relu')(deep)
deep = tf.keras.layers.Dense(128, activation='relu')(deep)

# 构造wide部分
wide = tf.keras.layers.DenseFeatures(crossed_feature)(inputs)

# wide + deep
both = tf.keras.layers.concatenate([deep, wide])
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(both)
model = tf.keras.Model(inputs, output_layer)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, validation_data=test_dataset, epochs=10)
