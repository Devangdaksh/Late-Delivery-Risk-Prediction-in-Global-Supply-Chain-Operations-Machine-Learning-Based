import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Page config + theme
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="APL Logistics | Late Delivery Risk",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#14213D"
STEEL = "#3A5A78"
AMBER = "#E8871E"
SAFE = "#2E8B57"
WARN = "#E8871E"
DANGER = "#C1440E"
BG = "#F6F7F9"

RISK_COLORS = {"Low Risk": SAFE, "Medium Risk": AMBER, "High Risk": DANGER}


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "clean"
MODAL_DIR = BASE_DIR / "modal"

DATA_PATH = DATA_DIR / "APL_Logistics_Cleaned.csv"
MODEL_PATH = MODAL_DIR / "late_delivery_model.joblib"
REGION_MAP_PATH = MODAL_DIR / "region_rate_map.joblib"
ARTIFACT_PATH = MODAL_DIR / "model_artifacts.json"

EXPRESS_MODES = {"Same Day", "First Class"}


# --------------------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    region_map = joblib.load(REGION_MAP_PATH)
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    return model, region_map, artifact, explainer


def engineer_features(df: pd.DataFrame, region_map: pd.Series, global_rate: float) -> pd.DataFrame:
    df = df.copy()
    df["shipping_pressure_index"] = df["Order Item Quantity"] / (df["Days for shipment (scheduled)"] + 1)
    df["express_mode_flag"] = df["Shipping Mode"].isin(EXPRESS_MODES).astype(int)
    df["order_complexity_score"] = df["Order Item Quantity"] * (1 + df["Order Item Discount Rate"])
    df["tight_schedule_flag"] = (df["Days for shipment (scheduled)"] <= 1).astype(int)
    df["discount_to_price_ratio"] = (
        df["Order Item Discount"] / df["Order Item Product Price"].replace(0, np.nan)
    ).fillna(0)
    df["regional_congestion_index"] = df["Order Region"].map(region_map).fillna(global_rate)
    return df


def risk_category(p):
    if p < 0.35:
        return "Low Risk"
    elif p < 0.65:
        return "Medium Risk"
    return "High Risk"


@st.cache_data(show_spinner="Scoring 180K+ orders for late-delivery risk...")
def load_and_score():
    model, region_map, artifact, _ = load_model_artifacts()
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df, region_map, artifact["global_late_rate"])
    X = df[artifact["feature_cols"]]
    proba = model.predict_proba(X)[:, 1]
    df["predicted_risk_proba"] = proba
    df["predicted_risk_category"] = df["predicted_risk_proba"].apply(risk_category)
    df["order_ref"] = (
        "ORD-" + df.index.astype(str).str.zfill(6)
    )
    return df


model, region_map, artifact, explainer = load_model_artifacts()
df_all = load_and_score()

CAT_COLS = artifact["categorical_cols"]
NUM_COLS = artifact["numeric_cols"]
FEATURE_COLS = artifact["feature_cols"]

# --------------------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------------------
st.sidebar.markdown("## 🚚 APL Logistics")
st.sidebar.markdown("**Late Delivery Risk Intelligence**")
st.sidebar.markdown("---")

markets = sorted(df_all["Market"].unique())
sel_markets = st.sidebar.multiselect("Market", markets, default=markets)

regions = sorted(df_all[df_all["Market"].isin(sel_markets)]["Order Region"].unique())
sel_regions = st.sidebar.multiselect("Order Region", regions, default=regions)

modes = sorted(df_all["Shipping Mode"].unique())
sel_modes = st.sidebar.multiselect("Shipping Mode", modes, default=modes)

segments = sorted(df_all["Customer Segment"].unique())
sel_segments = st.sidebar.multiselect("Customer Segment", segments, default=segments)

st.sidebar.markdown("### Risk threshold")
risk_threshold = st.sidebar.slider(
    "Flag orders at or above this probability as actionable risk",
    min_value=0.0, max_value=1.0, value=0.65, step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<span class="subtle">Model: **{artifact["best_model_name"]}** · '
    f'Test ROC-AUC: **{artifact["test_metrics"]["roc_auc"]:.3f}**</span>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<span class="subtle">Predicts risk from order-time signals only — '
    'no post-shipment fields are used.</span>',
    unsafe_allow_html=True,
)

mask = (
    df_all["Market"].isin(sel_markets)
    & df_all["Order Region"].isin(sel_regions)
    & df_all["Shipping Mode"].isin(sel_modes)
    & df_all["Customer Segment"].isin(sel_segments)
)
df = df_all[mask]

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.title("Late Delivery Risk Prediction")
st.markdown(
    '<span class="subtle">Machine learning–based early-warning system for '
    'APL Logistics (KWE Group) — flags high-risk orders before they ship, '
    'so operations can reroute, reprioritize, or communicate proactively.</span>',
    unsafe_allow_html=True,
)
st.markdown("")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Delay Risk Overview",
    "🔍 Order-Level Risk Prediction",
    "🗺️ Region & Mode Risk Analysis",
    "🚨 Operations Action Panel",
])

# ========================================================================================
# TAB 1 — Delay Risk Overview
# ========================================================================================
with tab1:
    if df.empty:
        st.warning("No orders match the current filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        n_total = len(df)
        n_high = (df["predicted_risk_category"] == "High Risk").sum()
        n_actionable = (df["predicted_risk_proba"] >= risk_threshold).sum()
        avg_risk = df["predicted_risk_proba"].mean()

        c1.metric("Orders in view", f"{n_total:,}")
        c2.metric("High-risk orders", f"{n_high:,}", f"{n_high/n_total:.1%} of orders")
        c3.metric(f"At/above {risk_threshold:.0%} threshold", f"{n_actionable:,}")
        c4.metric("Average predicted risk", f"{avg_risk:.1%}")

        st.markdown("---")
        col1, col2 = st.columns([1, 1.3])

        with col1:
            st.subheader("Risk category distribution")
            dist = df["predicted_risk_category"].value_counts().reindex(
                ["Low Risk", "Medium Risk", "High Risk"]
            ).fillna(0)
            fig = go.Figure(go.Pie(
                labels=dist.index, values=dist.values, hole=0.55,
                marker=dict(colors=[RISK_COLORS[c] for c in dist.index]),
                textinfo="label+percent",
            ))
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=340)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Predicted risk probability distribution")
            fig = px.histogram(
                df, x="predicted_risk_proba", nbins=40,
                color_discrete_sequence=[STEEL],
            )
            fig.add_vline(x=risk_threshold, line_dash="dash", line_color=DANGER,
                           annotation_text="threshold", annotation_position="top")
            fig.update_layout(
                xaxis_title="Predicted late-delivery probability", yaxis_title="Orders",
                margin=dict(t=10, b=10, l=10, r=10), height=340,
            )
            st.plotly_chart(fig, width="stretch")

        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Risk by shipping mode")
            m = df.groupby("Shipping Mode")["predicted_risk_proba"].mean().sort_values(ascending=False)
            fig = px.bar(m, orientation="h", color=m.values, color_continuous_scale=["#2E8B57", "#E8871E", "#C1440E"])
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                               xaxis_title="Avg predicted risk", yaxis_title="",
                               margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig, width="stretch")
        with col4:
            st.subheader("Risk by customer segment")
            s = df.groupby("Customer Segment")["predicted_risk_proba"].mean().sort_values(ascending=False)
            fig = px.bar(s, orientation="h", color=s.values, color_continuous_scale=["#2E8B57", "#E8871E", "#C1440E"])
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                               xaxis_title="Avg predicted risk", yaxis_title="",
                               margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig, width="stretch")

        with st.expander("Model performance detail (held-out test set)"):
            tm = artifact["test_metrics"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
            mc2.metric("Precision", f"{tm['precision']:.3f}")
            mc3.metric("Recall", f"{tm['recall']:.3f}")
            mc4.metric("F1 Score", f"{tm['f1']:.3f}")
            cm = np.array(tm["confusion_matrix"])
            fig = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                x=["Pred: On-time", "Pred: Late"], y=["Actual: On-time", "Actual: Late"],
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Note: 'Delivery Status' and 'Days for shipping (real)' were excluded from "
                "training — both are only known after shipment and would leak the answer. "
                "All predictions here use only information available at order time."
            )

# ========================================================================================
# TAB 2 — Order-Level Risk Prediction
# ========================================================================================
with tab2:
    st.subheader("Look up an individual order")
    st.caption("Pick an order from the filtered set to see its risk score and the factors driving it.")

    if df.empty:
        st.warning("No orders match the current filters.")
    else:
        lookup_df = df.sort_values("predicted_risk_proba", ascending=False).head(2000)
        order_choice = st.selectbox(
            "Order reference (showing top 2,000 by risk within current filters)",
            lookup_df["order_ref"].tolist(),
        )
        row = df[df["order_ref"] == order_choice].iloc[0]

        proba = row["predicted_risk_proba"]
        cat = row["predicted_risk_category"]

        colA, colB = st.columns([1, 2])
        with colA:
            st.markdown(f"### {order_choice}")
            st.markdown(
                f'<span class="risk-pill" style="background-color:{RISK_COLORS[cat]}">{cat}</span>',
                unsafe_allow_html=True,
            )
            st.metric("Late-delivery probability", f"{proba:.1%}")
            st.write(f"**Shipping mode:** {row['Shipping Mode']}")
            st.write(f"**Scheduled transit:** {row['Days for shipment (scheduled)']} days")
            st.write(f"**Order region:** {row['Order Region']} ({row['Market']})")
            st.write(f"**Customer segment:** {row['Customer Segment']}")
            st.write(f"**Order status:** {row['Order Status']}")

        with colB:
            st.markdown("**Key contributing factors (SHAP local explanation)**")
            X_row = row[FEATURE_COLS].to_frame().T
            for c in NUM_COLS:
                X_row[c] = pd.to_numeric(X_row[c])
            X_trans = model.named_steps["prep"].transform(X_row)
            sv = explainer.shap_values(X_trans)
            sv = np.array(sv).reshape(-1)

            ohe = model.named_steps["prep"].named_transformers_["cat"]
            cat_names = list(ohe.get_feature_names_out(CAT_COLS))
            all_names = NUM_COLS + cat_names

            contrib = pd.Series(sv, index=all_names)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
            top_df = pd.DataFrame({
                "factor": top.index,
                "impact": top.values,
                "direction": np.where(top.values > 0, "↑ increases risk", "↓ decreases risk"),
            })
            fig = px.bar(
                top_df, x="impact", y="factor", orientation="h", color="impact",
                color_continuous_scale=["#2E8B57", "#F6F7F9", "#C1440E"],
                color_continuous_midpoint=0,
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                xaxis_title="SHAP contribution (push toward late-risk →)",
                margin=dict(t=10, b=10, l=10, r=10), height=380,
            )
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("What-if simulator")
    st.caption("Estimate risk for a hypothetical new order before it's placed.")

    with st.form("whatif"):
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            wi_type = st.selectbox("Payment type", sorted(df_all["Type"].unique()))
            wi_mode = st.selectbox("Shipping mode", sorted(df_all["Shipping Mode"].unique()))
            wi_sched = st.slider("Scheduled shipping days", 0, 6, 4)
            wi_qty = st.slider("Order item quantity", 1, 10, 2)
        with wc2:
            wi_segment = st.selectbox("Customer segment", sorted(df_all["Customer Segment"].unique()))
            wi_region = st.selectbox("Order region", sorted(df_all["Order Region"].unique()))
            wi_market = st.selectbox("Market", sorted(df_all["Market"].unique()))
            wi_status = st.selectbox("Order status", sorted(df_all["Order Status"].unique()))
        with wc3:
            wi_price = st.number_input("Product price ($)", 5.0, 2000.0, 100.0, step=5.0)
            wi_discount_rate = st.slider("Discount rate", 0.0, 0.5, 0.1, step=0.01)
            wi_profit_ratio = st.slider("Order item profit ratio", -0.5, 0.6, 0.2, step=0.01)
            wi_country = st.selectbox("Customer country", sorted(df_all["Customer Country"].unique()))
        submitted = st.form_submit_button("Predict risk")

    if submitted:
        sim = pd.DataFrame([{
            "Type": wi_type,
            "Days for shipment (scheduled)": wi_sched,
            "Benefit per order": wi_price * wi_profit_ratio,
            "Sales per customer": wi_price * wi_qty,
            "Category Name": df_all["Category Name"].mode()[0],
            "Customer Country": wi_country,
            "Customer Segment": wi_segment,
            "Department Name": df_all["Department Name"].mode()[0],
            "Market": wi_market,
            "Order Country": df_all["Order Country"].mode()[0],
            "Order Region": wi_region,
            "Order Status": wi_status,
            "Shipping Mode": wi_mode,
            "Order Item Discount": wi_price * wi_discount_rate,
            "Order Item Discount Rate": wi_discount_rate,
            "Order Item Product Price": wi_price,
            "Order Item Profit Ratio": wi_profit_ratio,
            "Order Item Quantity": wi_qty,
            "Sales": wi_price * wi_qty,
            "Order Item Total": wi_price * wi_qty * (1 - wi_discount_rate),
            "Order Profit Per Order": wi_price * wi_profit_ratio,
            "Product Price": wi_price,
        }])
        sim = engineer_features(sim, region_map, artifact["global_late_rate"])
        sim_proba = model.predict_proba(sim[FEATURE_COLS])[:, 1][0]
        sim_cat = risk_category(sim_proba)
        st.markdown(
            f'<span class="risk-pill" style="background-color:{RISK_COLORS[sim_cat]}">{sim_cat}</span> '
            f'&nbsp; **{sim_proba:.1%}** predicted late-delivery probability',
            unsafe_allow_html=True,
        )

# ========================================================================================
# TAB 3 — Region & Mode Risk Analysis
# ========================================================================================
with tab3:
    if df.empty:
        st.warning("No orders match the current filters.")
    else:
        st.subheader("Risk heatmap — Region × Shipping Mode")
        pivot = df.pivot_table(
            index="Order Region", columns="Shipping Mode",
            values="predicted_risk_proba", aggfunc="mean",
        )
        fig = px.imshow(
            pivot, color_continuous_scale=["#2E8B57", "#F4D35E", "#C1440E"],
            aspect="auto", labels=dict(color="Avg predicted risk"),
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=520)
        st.plotly_chart(fig, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Shipping mode risk comparison")
            m = df.groupby("Shipping Mode").agg(
                avg_risk=("predicted_risk_proba", "mean"),
                orders=("predicted_risk_proba", "size"),
                high_risk_count=("predicted_risk_category", lambda s: (s == "High Risk").sum()),
            ).sort_values("avg_risk", ascending=False)
            st.dataframe(
                m.style.format({"avg_risk": "{:.1%}", "orders": "{:,}", "high_risk_count": "{:,}"})
                .background_gradient(subset=["avg_risk"], cmap="OrRd"),
                width="stretch",
            )
        with col2:
            st.subheader("Top 10 highest-risk regions")
            r = df.groupby("Order Region")["predicted_risk_proba"].mean().sort_values(ascending=False).head(10)
            fig = px.bar(r, orientation="h", color=r.values, color_continuous_scale=["#F4D35E", "#C1440E"])
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                               yaxis=dict(autorange="reversed"),
                               xaxis_title="Avg predicted risk", yaxis_title="",
                               margin=dict(t=10, b=10, l=10, r=10), height=380)
            st.plotly_chart(fig, width="stretch")

        st.subheader("Market-level summary")
        mk = df.groupby("Market").agg(
            orders=("predicted_risk_proba", "size"),
            avg_risk=("predicted_risk_proba", "mean"),
            high_risk_share=("predicted_risk_category", lambda s: (s == "High Risk").mean()),
        ).sort_values("avg_risk", ascending=False)
        st.dataframe(
            mk.style.format({"orders": "{:,}", "avg_risk": "{:.1%}", "high_risk_share": "{:.1%}"})
            .background_gradient(subset=["avg_risk"], cmap="OrRd"),
            width="stretch",
        )

# ========================================================================================
# TAB 4 — Operations Action Panel
# ========================================================================================
with tab4:
    st.subheader("Orders requiring immediate attention")
    st.caption(
        f"Orders at or above the {risk_threshold:.0%} risk threshold, ranked by predicted "
        "probability — treat as a prioritized action queue."
    )

    if df.empty:
        st.warning("No orders match the current filters.")
    else:
        action_df = df[df["predicted_risk_proba"] >= risk_threshold].sort_values(
            "predicted_risk_proba", ascending=False
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Orders flagged", f"{len(action_df):,}")
        c2.metric("Share of filtered orders", f"{len(action_df)/max(len(df),1):.1%}")
        c3.metric("Sales value at risk", f"${action_df['Sales'].sum():,.0f}")

        display_cols = [
            "order_ref", "predicted_risk_proba", "predicted_risk_category",
            "Shipping Mode", "Order Region", "Market", "Customer Segment",
            "Days for shipment (scheduled)", "Order Status", "Sales",
        ]
        show_df = action_df[display_cols].rename(columns={
            "order_ref": "Order",
            "predicted_risk_proba": "Risk %",
            "predicted_risk_category": "Category",
        })
        show_df["Risk %"] = (show_df["Risk %"] * 100).round(1)

        st.dataframe(
            show_df.head(500).style.format({"Sales": "${:,.2f}", "Risk %": "{:.1f}%"})
            .background_gradient(subset=["Risk %"], cmap="OrRd"),
            width="stretch", height=420,
        )
        st.caption(f"Showing top 500 of {len(action_df):,} flagged orders, sorted by risk.")

        csv = action_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download full action queue (CSV)",
            data=csv, file_name="high_risk_orders_action_queue.csv", mime="text/csv",
        )

        st.markdown("---")
        st.subheader("Where the flagged orders concentrate")
        col1, col2 = st.columns(2)
        with col1:
            g = action_df["Order Region"].value_counts().head(10)
            fig = px.bar(g, orientation="h", color_discrete_sequence=[DANGER])
            fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"),
                               xaxis_title="Flagged orders", yaxis_title="",
                               margin=dict(t=10, b=10, l=10, r=10), height=360)
            st.plotly_chart(fig, width="stretch")
        with col2:
            g = action_df["Shipping Mode"].value_counts()
            fig = px.bar(g, orientation="h", color_discrete_sequence=[AMBER])
            fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"),
                               xaxis_title="Flagged orders", yaxis_title="",
                               margin=dict(t=10, b=10, l=10, r=10), height=360)
            st.plotly_chart(fig, width="stretch")