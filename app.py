import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt

# app title
# note: this must be the very first Streamlit command in the script
st.set_page_config(
    page_title="Titanic Survivor Predictor",
    page_icon="🛳️",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("Titanic Survivor Predictor with XAI")

col1, col2 = st.columns([3, 4])


def load_and_preprocess_data():
    url = (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    df = pd.read_csv(url)

    df.dropna(subset=["Survived"], inplace=True)

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    df_features = df[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]]

    return df_features


training_data_df = load_and_preprocess_data()
training_data_np = training_data_df.values


# opening the model
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)


with col1:
    st.header("passenger details")
    pclass = st.selectbox("passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    sex = st.radio("sex", ["male", "female"])
    age = st.slider("age", 0, 80, 25)
    fare = st.slider("fare", 0, 500, 50)
    sibsp = st.number_input(
        "Siblings/Spouses Aboard", min_value=0, max_value=8, value=0
    )
    parch = st.number_input(
        "Parents/Children Aboard", min_value=0, max_value=6, value=0
    )
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

with col2:

    if sex == "male":
        sex = 0
    else:
        sex = 1

    embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

    input_data = pd.DataFrame(
        [[pclass, sex, age, fare, sibsp, parch, embarked]],
        columns=["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"],
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.subheader(f"Prediction: {'Survived' if prediction else 'Did NOT Survive'}")
    st.write(f"Probability of Survival: **{probability*100:.2f}%**")

    # SHAP explanation
    st.subheader("SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    fig, axis = plt.subplots(figsize=(8, 4))
    shap.summary_plot(
        shap_values, input_data, plot_type="bar", show=False
    )  # shap summary plot on this figure
    st.pyplot(fig)  # pass the fig object explicitly to Streamlit

    # LIME explanation
    st.subheader("LIME Explanation")
    lime_explanation = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data_np,
        feature_names=["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"],
        class_names=["Not Survived", "Survived"],
        mode="classification",
    )

    lime_exp_result = lime_explanation.explain_instance(
        input_data.iloc[0].values, model.predict_proba, num_features=4
    )

    lime_html = lime_exp_result.as_html()

    styled_html = f"""
  <div style="color: white; background-color: #2c3e50; padding: 10px; border-radius: 8px;">
      {lime_html}
  </div>
  """

    st.components.v1.html(styled_html, height=400)

    st.subheader("feature importance")
    imp = model.feature_importances_
    feature_names = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": imp}).sort_values(
        by="Importance"
    )

    fig, ax = plt.subplots()
    ax.barh(imp_df["Feature"], imp_df["Importance"])
    ax.set_title("Feature Importance")
    st.pyplot(fig)
