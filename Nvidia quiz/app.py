import streamlit as st
import json
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

st.set_page_config(
    page_title="AI PDF Quiz Generator",
    layout="wide"
)

st.title("📚 AI PDF Quiz Generator")

# -----------------------------
# PDF PROCESSING
# -----------------------------
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    return docs, chunks

# -----------------------------
# ROBUST ANSWER RESOLUTION
# -----------------------------
def get_correct_index(answer_field, options):
    """
    Figures out which option index is correct, no matter how the LLM
    formatted the 'answer' field (e.g. "A", "a", "A) Paris", "(C)", or
    even the full option text). Returns None if it truly can't resolve.
    """
    answer_field = str(answer_field).strip()

    match = re.search(r"[A-Da-d]", answer_field)
    if match:
        letter = match.group(0).upper()
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(options):
            return idx

    normalized_options = [o.strip().lower() for o in options]
    normalized_answer = answer_field.strip().lower()
    if normalized_answer in normalized_options:
        return normalized_options.index(normalized_answer)

    return None

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:

    pdf_path = uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Generate Quiz"):

        with st.spinner("Reading PDF..."):
            docs, chunks = process_pdf(pdf_path)

        st.write(f"Pages: {len(docs)}")
        st.write(f"Chunks: {len(chunks)}")

        context = "\n\n".join(
            [chunk.page_content for chunk in chunks[:5]]
        )

        llm = ChatOpenAI(
            model="deepseek-ai/deepseek-v4-flash",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY,
            temperature=0.3
        )

        prompt = f"""
Generate 5 multiple-choice questions from the content.

Rules:
- Exactly 4 options
- Only one correct answer
- "answer" must be ONLY a single letter: A, B, C, or D. No parentheses, no text, no punctuation.
- Return ONLY valid JSON, no markdown fences, no commentary.

Format:

[
  {{
    "question":"...",
    "options":[
      "...",
      "...",
      "...",
      "..."
    ],
    "answer":"A"
  }}
]

Content:

{context}
"""

        with st.spinner("Generating Quiz..."):
            response = llm.invoke(prompt)

        try:
            raw_text = response.content

            if "```json" in raw_text:
                raw_text = raw_text.replace("```json", "")
                raw_text = raw_text.replace("```", "")

            start = raw_text.find("[")
            end = raw_text.rfind("]") + 1

            json_text = raw_text[start:end]

            quiz_data = json.loads(json_text)

            # Clean options (strip stray whitespace from LLM formatting)
            for q in quiz_data:
                q["options"] = [str(o).strip() for o in q["options"]]

            st.session_state["quiz_data"] = quiz_data
            # Reset any old radio selections from a previous quiz
            for key in list(st.session_state.keys()):
                if key.startswith("question_"):
                    del st.session_state[key]

            st.success("Quiz Generated Successfully")

        except Exception as e:
            st.error(f"JSON Parsing Error: {e}")
            st.text(response.content)

# -----------------------------
# DISPLAY QUIZ
# -----------------------------
if "quiz_data" in st.session_state:

    quiz_data = st.session_state["quiz_data"]

    st.subheader("Quiz")

    user_answers = []

    for i, q in enumerate(quiz_data):

        answer = st.radio(
            f"Q{i+1}. {q['question']}",
            q["options"],
            index=None,
            key=f"question_{i}"
        )

        user_answers.append(answer)

    if st.button("Submit Quiz"):

        unanswered = [i + 1 for i, a in enumerate(user_answers) if a is None]

        if unanswered:
            st.warning(
                f"Please answer all questions before submitting. "
                f"Missing: Q{', Q'.join(map(str, unanswered))}"
            )

        else:
            score = 0
            results = []

            for q, user_ans in zip(quiz_data, user_answers):

                correct_idx = get_correct_index(q["answer"], q["options"])

                if correct_idx is None:
                    # LLM gave an unresolvable answer — skip grading this one safely
                    results.append((q, user_ans, None))
                    continue

                correct_text = q["options"][correct_idx]
                is_correct = (user_ans == correct_text)

                if is_correct:
                    score += 1

                results.append((q, user_ans, correct_text))

            graded = [r for r in results if r[2] is not None]
            st.success(f"Your Score: {score}/{len(graded)}")

            st.subheader("Results")

            for q, user_ans, correct_text in results:

                if correct_text is None:
                    st.warning(f"⚠ {q['question']} — could not grade (bad answer key from AI)")
                    continue

                if user_ans == correct_text:
                    st.success(f"✔ {q['question']}")
                else:
                    st.error(f"✘ {q['question']}")
                    st.write(f"Your Answer: {user_ans}")
                    st.write(f"Correct Answer: {correct_text}")