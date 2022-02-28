import streamlit as st

from ui.utils import add_custom_css
from ui.pages import PAGE_MAP
from ui.state import provide_state

add_custom_css()


@provide_state()
def main(state=None):
    current_page = st.sidebar.radio("Go To", list(PAGE_MAP))
    PAGE_MAP[current_page](state=state).write()


if __name__ == "__main__":
    main()
