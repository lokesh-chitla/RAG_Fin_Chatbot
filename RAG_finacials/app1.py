import streamlit as st
from generator import generate_response

def main():
    st.title("RAG Financial Q&A")

    # Let user pick retrieval approach if you want both Basic & Multi-Stage
    retrieval_mode = st.selectbox("Select Retrieval Mode:", ["basic", "multi-stage"])

    user_query = st.text_input("Enter your query here:")

    if st.button("Submit"):
        if user_query.strip():
            with st.spinner("Generating answer..."):
                # We pass the retrieval_mode to generate_response
                answer = generate_response(user_query, mode=retrieval_mode)
                st.markdown("### Answer")
                st.write(answer)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
