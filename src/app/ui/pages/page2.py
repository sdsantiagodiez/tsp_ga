import streamlit as st
from ..utils import Page


class Page2(Page):
    def __init__(self, state):
        self.state = state

    def write(self):
        st.title("Map routes")

        st.write(self.state.client_config["slider_value"])
