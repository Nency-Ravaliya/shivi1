import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

class KANLayer(layers.Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super(KANLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='w'
        )
        self.attention = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='attention'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='b'
        )
        
    def call(self, inputs):
        # Enhanced attention mechanism
        attention_scores = tf.nn.softmax(tf.matmul(inputs, self.attention))
        attended_inputs = inputs * tf.matmul(attention_scores, tf.transpose(self.w))
        output = tf.matmul(attended_inputs, self.w) + self.b
        return self.activation(output)

def create_enhanced_kan_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    
    # Initial feature extraction
    x = layers.Dense(1024, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # First KAN block
    x1 = KANLayer(768, activation='selu')(x)
    x1 = layers.Dropout(0.4)(x1)
    x1 = layers.BatchNormalization()(x1)
    
    # Second KAN block
    x2 = KANLayer(512, activation='selu')(x1)
    x2 = layers.Dropout(0.4)(x2)
    x2 = layers.BatchNormalization()(x2)
    
    # Third KAN block with skip connection
    x3 = KANLayer(256, activation='selu')(x2)
    x3 = layers.Dropout(0.4)(x3)
    x3 = layers.BatchNormalization()(x3)
    
    # Merge features with skip connections
    x = layers.Concatenate()([x1, x2, x3])
    
    # Deep feature extraction
    x = layers.Dense(512, activation='selu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='selu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='selu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def load_and_merge_data():
    print("Loading datasets...")
    train_main = pd.read_csv('dataset/Train-1542865627584.csv')
    train_beneficiary = pd.read_csv('dataset/Train_Beneficiarydata-1542865627584.csv')
    train_inpatient = pd.read_csv('dataset/Train_Inpatientdata-1542865627584.csv')
    train_outpatient = pd.read_csv('dataset/Train_Outpatientdata-1542865627584.csv')
    
    print("Processing inpatient data...")
    inpatient_agg = train_inpatient.groupby('Provider').agg({
        'InscClaimAmtReimbursed': ['sum', 'mean', 'std', 'count'],
        'DeductibleAmtPaid': ['sum', 'mean', 'std'],
        'ClmDiagnosisCode_1': 'nunique',
        'ClmProcedureCode_1': 'nunique',
        'AdmissionDt': 'count',
        'DischargeDt': 'count'
    }).reset_index()
    
    inpatient_agg.columns = ['Provider', 'TotalInpatientReimbursed', 'AvgInpatientReimbursed', 
                            'StdInpatientReimbursed', 'InpatientClaimCount', 
                            'TotalInpatientDeductible', 'AvgInpatientDeductible',
                            'StdInpatientDeductible', 'UniqueInpatientDiagnosis', 
                            'UniqueInpatientProcedures', 'AdmissionCount', 'DischargeCount']
    
    print("Processing outpatient data...")
    outpatient_agg = train_outpatient.groupby('Provider').agg({
        'InscClaimAmtReimbursed': ['sum', 'mean', 'std', 'count'],
        'DeductibleAmtPaid': ['sum', 'mean', 'std'],
        'ClmDiagnosisCode_1': 'nunique',
        'ClmProcedureCode_1': 'nunique'
    }).reset_index()
    
    outpatient_agg.columns = ['Provider', 'TotalOutpatientReimbursed', 'AvgOutpatientReimbursed',
                             'StdOutpatientReimbursed', 'OutpatientClaimCount',
                             'TotalOutpatientDeductible', 'AvgOutpatientDeductible',
                             'StdOutpatientDeductible', 'UniqueOutpatientDiagnosis',
                             'UniqueOutpatientProcedures']
    
    print("Processing beneficiary data...")
    # First, get all unique BeneID-Provider pairs from claims
    inpatient_bene = train_inpatient[['BeneID', 'Provider']].drop_duplicates()
    outpatient_bene = train_outpatient[['BeneID', 'Provider']].drop_duplicates()
    bene_provider = pd.concat([inpatient_bene, outpatient_bene]).drop_duplicates()
    
    # Merge beneficiary data with provider information
    bene_provider_data = bene_provider.merge(train_beneficiary, on='BeneID', how='left')
    
    # Now aggregate beneficiary features by provider
    beneficiary_agg = bene_provider_data.groupby('Provider').agg({
        'NoOfMonths_PartACov': ['mean', 'std'],
        'NoOfMonths_PartBCov': ['mean', 'std'],
        'ChronicCond_Alzheimer': 'mean',
        'ChronicCond_Heartfailure': 'mean',
        'ChronicCond_KidneyDisease': 'mean',
        'ChronicCond_Cancer': 'mean',
        'ChronicCond_ObstrPulmonary': 'mean',
        'ChronicCond_Depression': 'mean',
        'ChronicCond_Diabetes': 'mean',
        'ChronicCond_IschemicHeart': 'mean',
        'ChronicCond_Osteoporasis': 'mean',
        'ChronicCond_rheumatoidarthritis': 'mean',
        'ChronicCond_stroke': 'mean',
        'IPAnnualReimbursementAmt': ['sum', 'mean', 'std'],
        'IPAnnualDeductibleAmt': ['sum', 'mean', 'std'],
        'OPAnnualReimbursementAmt': ['sum', 'mean', 'std'],
        'OPAnnualDeductibleAmt': ['sum', 'mean', 'std']
    }).reset_index()
    
    # Flatten column names
    beneficiary_agg.columns = [
        'Provider' if col[0] == 'Provider' else f"{col[0]}_{col[1]}" 
        for col in beneficiary_agg.columns
    ]
    
    print("Merging datasets...")
    merged_train = train_main.merge(inpatient_agg, on='Provider', how='left')
    merged_train = merged_train.merge(outpatient_agg, on='Provider', how='left')
    merged_train = merged_train.merge(beneficiary_agg, on='Provider', how='left')
    
    merged_train = merged_train.fillna(0)
    
    print(f"Final dataset shape: {merged_train.shape}")
    return merged_train

# ... [Rest of the code remains the same] ...


def preprocess_data(df):
    print("Creating advanced features...")
    # Convert PotentialFraud to numeric
    df['PotentialFraud'] = df['PotentialFraud'].map({'Yes': 1, 'No': 0})
    
    # Advanced feature engineering
    df['ReimbursementRatio'] = df['TotalOutpatientReimbursed'] / (df['TotalInpatientReimbursed'] + 1)
    df['DeductibleRatio'] = df['TotalOutpatientDeductible'] / (df['TotalInpatientDeductible'] + 1)
    df['ClaimDensity'] = (df['OutpatientClaimCount'] + df['InpatientClaimCount']) / (df['TotalOutpatientReimbursed'] + df['TotalInpatientReimbursed'] + 1)
    df['AvgReimbursementPerClaim'] = (df['TotalOutpatientReimbursed'] + df['TotalInpatientReimbursed']) / (df['OutpatientClaimCount'] + df['InpatientClaimCount'] + 1)
    df['DiagnosisComplexity'] = df['UniqueInpatientDiagnosis'] + df['UniqueOutpatientDiagnosis']
    df['ProcedureComplexity'] = df['UniqueInpatientProcedures'] + df['UniqueOutpatientProcedures']
    
    # Risk indicators
    df['ReimbursementVariability'] = df['StdInpatientReimbursed'] / (df['AvgInpatientReimbursed'] + 1)
    df['DeductibleVariability'] = df['StdInpatientDeductible'] / (df['AvgInpatientDeductible'] + 1)
    df['ClaimFrequency'] = (df['InpatientClaimCount'] + df['OutpatientClaimCount']) / (df['AdmissionCount'] + 1)
    
    # Chronic condition risk score
    chronic_columns = [col for col in df.columns if 'ChronicCond_' in col]
    df['ChronicConditionScore'] = df[chronic_columns].mean(axis=1)
    
    print("Preprocessing features...")
    # Drop non-numeric columns
    X = df.drop(['Provider', 'PotentialFraud'], axis=1)
    y = df['PotentialFraud']
    
    # Convert to float32
    X = X.astype('float32')
    
    return X, y

def train_model():
    print("\nStarting model training process...")
    print("=" * 50)
    
    # Load and preprocess data
    df = load_and_merge_data()
    X, y = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    print("\nScaling features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE-Tomek for balanced sampling
    print("Applying SMOTE-Tomek for balanced sampling...")
    smote_tomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_scaled, y_train)
    
    # Create and compile model
    print("\nCreating enhanced KAN model...")
    model = create_enhanced_kan_model(X_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Train the model
    print("\nTraining model...")
    class_weights = {0: 1.0, 1: 4.0}  # Increased weight for fraud cases
    history = model.fit(
        X_train_balanced, y_train_balanced,
        epochs=48,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=50,
                restore_best_weights=True,
                min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=20,
                min_lr=0.00001
            )
        ],
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model performance...")
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nModel Performance Metrics:")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    print("\nGenerating visualizations...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("\nTraining completed successfully!")
    print("=" * 50)
    
    return model, scaler

if __name__ == "__main__":
    train_model()