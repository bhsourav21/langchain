import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.title("Travel Guide App")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = PromptTemplate(
    input_variables=["city", "month", "language", "budget_category"],
    template="""Welcome to the {city} travel guide!
    If you are visiting {city} in {month}, here is what you can do:
        1. Must visit attractions
        2. Local cuisines you must try.
        3. helful phrases in {language} to get around.
        4. Tips for travelling on a {budget_category} budget.

    Return the response as a valid JSON object with exactly the following keys:
        - "attractions": [list of attractions],
        - "cuisines": [list of cuisines],
        - "phrases": [list of phrases],
        - "tips": [list of tips]
    Do not include any explanations, markdown, or text outside the JSON.
    """
)

city = st.text_input("Enter the city: ", placeholder="e.g. Paris, London, Tokyo")
month = st.text_input("Enter the month: ", placeholder="e.g. January, February, March")
language = st.text_input("Enter the language: ", placeholder="e.g. English, French, Spanish")
budget_category = st.selectbox("Enter the budget category: ", ["Budget", "Mid-range", "Luxury"])

if city and month and language and budget_category:
    prompt_value = prompt.format(
        city=city,
        month=month,
        language=language,
        budget_category=budget_category,
    )
    response = llm.invoke(prompt_value)
    content = response.content.strip()

    # Strip possible markdown code fences
    if "```" in content:
        parts = content.split("```")
        if len(parts) >= 3:
            content = parts[1].strip()
        else:
            content = parts[-1].strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        st.error("Could not parse the model response as JSON. Showing raw output instead.")
        st.write(response.content)
    else:
        st.subheader("Attractions")
        st.write(data.get("attractions", []))

        st.subheader("Cuisines")
        st.write(data.get("cuisines", []))

        st.subheader("Helpful phrases")
        st.write(data.get("phrases", []))

        st.subheader("Tips")
        st.write(data.get("tips", []))

        st.subheader("Raw JSON")
        st.json(data)
