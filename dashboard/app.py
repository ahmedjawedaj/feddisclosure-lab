"""Optional Streamlit interface for trusted local research viewing.

It reads *only* scalar reviewed evidence. Never loads attacker tensors or raw
per-site records. Run via: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.evidence import make_snapshot
import streamlit as st
import pandas as pd

st.set_page_config(page_title='FedDisclosure Lab', layout='wide')
st.title('FedDisclosure Lab · Research Explorer')
st.caption('Approved public/synthetic data experiments · Read-only scalar evidence')
s = make_snapshot()
section = st.sidebar.selectbox('Experiment', ['Federated training', 'Aggregate leakage', 'Membership inference', 'Verification'])
if section == 'Federated training':
    st.subheader('Equal-work, five-seed partition study')
    df = pd.DataFrame(s['heterogeneity']).set_index('name')
    st.bar_chart(df[['global_accuracy', 'worst_client_accuracy']])
    st.dataframe(df)
elif section == 'Aggregate leakage':
    st.subheader('Five-seed idealized arithmetic aggregation')
    df = pd.DataFrame(s['collusion']).set_index('name')
    st.bar_chart(df[['mean_mse']])
    st.subheader('Joint reconstruction, all contributor labels known')
    st.dataframe(pd.DataFrame(s['joint_oracle']))
    st.warning('Joint attack is label-privileged; no SecAgg+ native run or site attribution.')
elif section == 'Membership inference':
    df = pd.DataFrame(s['membership']).set_index('name')
    st.bar_chart(df[['auc_mean']])
    st.dataframe(df)
else:
    st.json(s['native_protocol_verified'])
    for limitation in s['limitations']:
        st.write('•', limitation)
