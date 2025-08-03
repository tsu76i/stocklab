import streamlit as st

about_page = st.Page("about.py", title="About", icon=":material/info:")
dashboard_page = st.Page(
    "dashboard.py",
    title="StockLab Dashboard",
    icon=":material/finance_mode:",
    default=True,
)
pg = st.navigation([about_page, dashboard_page])
pg.run()
