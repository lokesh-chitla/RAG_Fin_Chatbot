import streamlit as st
from generator import generate_response

def main():
    st.title("RAG Financial Q&A")

    user_query = st.text_input("Enter your question about TCS financials:")

    if st.button("Submit"):
        if user_query.strip():
            with st.spinner("Generating answer..."):
                answer = generate_response(user_query)
                st.markdown("### Answer")
                st.write(answer)
        else:
            st.warning("Please enter a query first.")

if __name__ == "__main__":
    main()
