import streamlit as st
from mock_backend import get_recommendations

st.set_page_config(page_title="Shopping Assistant", page_icon="🛍️", layout="centered")

st.title("🛍️ Shopping Product Recommendation Assistant")
st.write("Describe what you're looking for, and we'll recommend the best matches.")

# --- Input ---------------------------------------------------------------
query = st.text_input(
    "What are you shopping for?",
    placeholder="e.g. I want a 32 inch TV with good picture quality, budget $500",
)
search_clicked = st.button("Find products", type="primary")

st.divider()

# --- Output ----------------------------------------------------------------
if search_clicked:
    if not query.strip():
        st.warning("Please describe what you're shopping for first.")
    else:
        with st.spinner("Finding the best matches for you..."):
            results = get_recommendations(query, top_k=3)

        if not results:
            st.info("No matching products found. Try adjusting your request.")
        else:
            st.subheader(f"Top {len(results)} recommendations")
            for product in results:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(product["imgUrl"], use_container_width=True)
                with col2:
                    st.markdown(f"**{product['title']}**")
                    st.write(f"⭐ {product['stars']} &nbsp;&nbsp; 💲{product['price']:.2f}")
                    st.caption(product["explanation"])
                    st.link_button("View product", product["productURL"])
                st.divider()
else:
    st.caption("Enter a request above and click **Find products** to see recommendations.")
