# Smart Event Attendance Predictor - Simple Web App (1 file)
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Smart Event Attendance Predictor", page_icon="🎯")


# ---------- Data + Models (sirf ek baar chalta hai) ----------
@st.cache_resource
def train_everything():
    # 1. Dataset banao (500 participants)
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "past_attendance": np.random.rand(n).round(2),
        "reminder_sent": np.random.randint(0, 2, n),
        "distance_km": np.random.randint(1, 31, n).astype(float),
        "weather": np.random.choice(["Sunny", "Rainy"], n),
    })
    score = (-1 + 3 * df["past_attendance"] + 1 * df["reminder_sent"]
             - 0.05 * df["distance_km"] - 1 * (df["weather"] == "Rainy"))
    df["attended"] = np.random.binomial(1, 1 / (1 + np.exp(-score)))
    df.loc[np.random.choice(n, 25, replace=False), "distance_km"] = np.nan
    raw = df.copy()

    # 2. Preprocessing (Lab 1)
    df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())
    clean = df.copy()
    df = pd.get_dummies(df, columns=["weather"], drop_first=True)
    X, y = df.drop("attended", axis=1), df["attended"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 3. Logistic Regression (Lab 3)
    lr = LogisticRegression().fit(X_train_sc, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_sc))

    # 4. Random Forest (Lab 5)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    # 5. K-Means (Lab 10): 3 groups
    cols = ["past_attendance", "distance_km"]
    cs = StandardScaler()
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    clean["cluster"] = km.fit_predict(cs.fit_transform(clean[cols]))
    order = clean.groupby("cluster")["attended"].mean().sort_values().index
    names = dict(zip(order, ["Rare attender", "Occasional attender", "Regular attender"]))

    return dict(raw=raw, X=X, scaler=scaler, lr=lr, rf=rf, lr_acc=lr_acc, rf_acc=rf_acc,
                km=km, cs=cs, cols=cols, names=names)


m = train_everything()

# ---------- Screen ----------
st.title("🎯 Smart Event Attendance Predictor")
st.write("Participant ki details daalo aur dekho woh event mein aayega ya nahi.")

st.sidebar.header("Participant Details")
past = st.sidebar.slider("Past attendance rate (0 = kabhi nahi, 1 = hamesha)", 0.0, 1.0, 0.5)
reminder = st.sidebar.checkbox("Reminder bheja gaya?", value=True)
distance = st.sidebar.slider("Venue ka distance (km)", 1, 30, 10)
weather = st.sidebar.selectbox("Weather", ["Sunny", "Rainy"])

if st.sidebar.button("Predict", type="primary"):
    new = pd.DataFrame([[past, int(reminder), float(distance), int(weather == "Sunny")]],
                       columns=m["X"].columns)
    rf_chance = m["rf"].predict_proba(new)[0][1] * 100
    lr_chance = m["lr"].predict_proba(m["scaler"].transform(new))[0][1] * 100
    group = m["names"][m["km"].predict(m["cs"].transform(
        pd.DataFrame([[past, float(distance)]], columns=m["cols"])))[0]]

    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.metric("Random Forest chance", f"{rf_chance:.0f}%")
    c2.metric("Logistic Regression chance", f"{lr_chance:.0f}%")
    st.progress(int(rf_chance))
    if rf_chance >= 50:
        st.success("✅ Yeh participant event mein AAYEGA")
    else:
        st.error("❌ Yeh participant event mein NAHI aayega")
    st.info(f"Participant group (K-Means): **{group}**")
else:
    st.info("👈 Left side mein details daalo aur **Predict** dabao.")

# ---------- Project details ----------
st.divider()
tab1, tab2, tab3 = st.tabs(["📊 Model Results", "🔍 Important Factors", "🗂 Dataset"])

with tab1:
    st.write("Test data par accuracy:")
    st.bar_chart(pd.DataFrame({"Accuracy": [m["lr_acc"], m["rf_acc"]]},
                              index=["Logistic Regression", "Random Forest"]))
with tab2:
    st.write("Kaunsa factor attendance par sabse zyada asar karta hai (Random Forest):")
    st.bar_chart(pd.Series(m["rf"].feature_importances_, index=m["X"].columns))
with tab3:
    st.write("Synthetic dataset (500 rows). NaN = missing value jo preprocessing mein bhari gayi.")
    st.dataframe(m["raw"].head(20))
