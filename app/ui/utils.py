import streamlit as st
from abc import ABC, abstractmethod


class Page(ABC):
    @abstractmethod
    def write(self):
        pass


st.set_page_config(page_title="Coffee Road Trip", layout="wide")


def add_custom_css():
    st.markdown(
        """
        <style>
        </style>
        """,
        unsafe_allow_html=True,
    )
