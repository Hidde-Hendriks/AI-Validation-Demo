import os
import streamlit as st
import pandas as pd
import re
from dotenv import load_dotenv
from langchain_openai import OpenAI
from PyPDF2 import PdfReader
from io import BytesIO
import plotly.express as px


# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI model
llm = OpenAI(api_key=OPENAI_API_KEY, model_name="azure.gpt-4o-mini")

def load_document(document_file, max_length=10000):
    document_text = ""
    if document_file.type == "application/pdf":
        reader = PdfReader(document_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
                if len(document_text) > max_length:
                    break
    else:
        document_text = document_file.read().decode("utf-8")
    
    return document_text[:max_length].strip()

def simple_query(query: str):
    question = (f"Create exactly 10 different versions of the following question, "
                f"starting with the original unchanged question as number 1, "
                f"followed by 9 distinct variations. Ensure each question is clearly numbered "
                f"from 1 to 10 and is kept as one sentence: {query}")
    response = llm.invoke(question, max_tokens=1000)
    return response

def process_responses(response):
    match = re.search(r'1\..*?10\..*?\?', response, re.DOTALL)
    if match:
        responses = match.group(0)
        answers = [answer.strip() for answer in responses.split('\n')]
    else:
        answers = []
    
    while len(answers) < 10:
        answers.append(None)

    answers = [re.sub(r'^\d+\.\s*', '', answer) for answer in answers]

    return answers

def answer_question(question: str, document: str, llm) -> str:
    prompt_with_document = (
        f"Based on the document, answer the question below:\n\n"
        f"Document: {document}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    response = llm.invoke(prompt_with_document)
    return response

def validate_answer_example(response: str, example_answer: str, llm) -> str:
    prompt = (
        f"Evaluate the answer based on the example answer:\n\n"
        f"Example answer: {example_answer}\n\n"
        f"Answer: {response}\n\n"
        f"Validation answer 'correct' or 'incorrect', followed by a reasoning statement:"
    )
    validation = llm.invoke(prompt)
    return validation

def validate_answer_criteria(response: str, certain_criteria: str, llm) -> str:
    prompt = (
        f"Evaluate the answer based on the given criteria:\n\n"
        f"Criteria: {certain_criteria}\n\n"
        f"Answer: {response}\n\n"
        f"Validation answer 'correct' or 'incorrect', followed by a reasoning statement:"
    )
    validation = llm.invoke(prompt)
    return validation

def evaluate_tone(answer: str, specified_tone: str, llm) -> str:
    prompt = (
        f"Analyze if the tone of the following answer matches the specified tone of voice: {specified_tone}\n\n"
        f"Answer: {answer}\n\n"
        f"Tone match evaluation answer 'correct' or 'incorrect', followed by a reasoning statement:"
    )
    tone_evaluation = llm.invoke(prompt)
    return tone_evaluation

def evaluate_robustness(answer: str, llm) -> str:
    prompt = (
        "Evaluate the robustness of the answer below:\n\n"
        f"Answer: {answer}\n\n"
        "Answer 'correct' or 'incorrect', followed by a reasoning statement:"
    )
    robustness_evaluation = llm.invoke(prompt)
    return robustness_evaluation

def evaluate_accuracy(answer: str, llm) -> str:
    prompt = (
        "Assess the following answer for factual accuracy and determine if it contains any hallucinations (fabricated or non-factual elements):\n\n"
        f"Answer: {answer}\n\n"
        "If any hallucinations are found, return 'incorrect', followed by an explanation of what the hallucinations are and why the answer is inaccurate. "
        "If no hallucinations are present and the information is accurate, return 'correct' along with a reasoning statement."
    )
    accuracy_evaluation = llm.invoke(prompt)
    return accuracy_evaluation

def evaluate_completeness(answer: str, llm) -> str:
    prompt = (
        "Assess if the following answer is complete by verifying inclusion of all necessary details and responses to the question requirements:\n\n"
        f"Answer: {answer}\n\n"
        "Completeness evaluation answer 'correct' if complete or 'incorrect' if incomplete, followed by a reasoning statement:"
    )
    completeness_evaluation = llm.invoke(prompt)
    return completeness_evaluation

def parse_validation_reason(response: str) -> tuple:
    """
    Parses the validation response to extract status and reasoning
    
    Args:
        response (str): The response string from the language model.
    
    Returns:
        tuple: A tuple containing the status and reasoning
    """
    clean_response = response.strip().lower()

    # Determine status
    if 'incorrect' in clean_response:
        status = 'incorrect'
        reasoning = clean_response
    elif 'correct' in clean_response:
        status = 'correct'
        reasoning = clean_response
    else:
        status = None
        reasoning = clean_response

    # Remove '*' characters and normalize spaces
    reasoning = reasoning.replace('*', '')
    reasoning = re.sub(r'\s+', ' ', reasoning).strip()  # Normalize spacing

    return status, reasoning

def calculate_overall_scores(results):
    scores_per_question = []
    scores_per_criteria = {}

    for criteria, result_df, correct_counts in results:
        if not result_df.empty:
            # Group by 'Original Question' to get average score per question
            for question, group in result_df.groupby('Original Question'):
                # Calculate the average score for this question (fraction of 'correct' validations)
                average_score = sum(group['Validation'].str.lower() == 'correct') / len(group['Validation'])
                scores_per_question.append((question, average_score))
                
                # Add this score to the corresponding criteria
                if criteria not in scores_per_criteria:
                    scores_per_criteria[criteria] = []
                scores_per_criteria[criteria].append(average_score)

    # Convert to DataFrame and calculate the overall averages
    question_scores_avg = pd.DataFrame(scores_per_question, columns=['Question', 'Score']).groupby('Question').mean().reset_index()
    criteria_scores_avg = {k: sum(v) / len(v) for k, v in scores_per_criteria.items() if len(v) > 0}

    return question_scores_avg, criteria_scores_avg, scores_per_criteria

def create_plotly_chart(correct_counts):
    fig = px.bar(
        x=list(range(1, len(correct_counts) + 1)),  # Question numbers starting from 1
        y=correct_counts,
        color_discrete_sequence=['#FF6600'],  # PwC orange color
        labels={'x': '', 'y': 'Correct Count'}
    )
    
    # Adjust axis properties
    fig.update_layout(
        xaxis={'title': 'Question Number', 'tickmode': 'linear'},
        yaxis={'range': [0, 10]}  # Setting the y-axis range from 0 to 10
    )

    # Ensure scores are displayed in a larger, bold font and centered
    fig.update_traces(
        text=[f"{count}" for count in correct_counts],  # Correct count as text annotations
        textposition="inside",  # Position the text inside the bar
        insidetextanchor='middle',  # Center the text vertically inside the bar
        textfont=dict(color="white", size=14, family="Arial Black")  # Bold and large text
    )
    
    return fig

st.set_page_config(layout="wide")

def page_upload_and_analyze():
    st.title("Upload and Select Validation")

    # Introduction and Information Section
    st.markdown("""
    ### Welcome to PwC's AI Validation Demo

    This application demonstrates PwC's capability to build robust AI validation models. Using this tool, you can 
    upload a document and an Excel sheet to validate the responses generated by a large language model (LLM).

    **Key Features:**
    - **Validation Across Criteria:** Configure multiple validation criteria like Example Answer Match, Tone of Voice, Robustness, and more.
    - **Customizable Inputs:** Specify inputs such as the desired tone of voice for a more tailored validation experience.
    - **Interactive Reporting:** Visualize validation results and download comprehensive analysis reports.

    **Note:** In a real-world scenario, our models would validate the responses generated by your own LLM implementations.
    The question-answering step in this demo is included to simulate this interaction for demonstration purposes.

    Please proceed to upload your document and Excel file to begin the validation process!
    """)

    # Checking and initializing session state for uploaded files
    if 'uploaded_document' not in st.session_state:
        st.session_state['uploaded_document'] = None
    
    if 'uploaded_excel' not in st.session_state:
        st.session_state['uploaded_excel'] = None

    # Display larger text for file upload sections
    st.markdown("<h2 style='text-align: left; font-size: 24px;'>Upload Your Document (Text/PDF):</h2>", unsafe_allow_html=True)
    uploaded_document = st.file_uploader("", type=["txt", "pdf"])
    if uploaded_document:
        st.session_state['uploaded_document'] = uploaded_document
        st.session_state['document_content'] = load_document(uploaded_document)
    
    if st.session_state['uploaded_document']:
        st.write(f"Document Uploaded: {st.session_state['uploaded_document'].name}")

    st.markdown("<h2 style='text-align: left; font-size: 24px;'>Upload Your Excel File:</h2>", unsafe_allow_html=True)
    uploaded_excel = st.file_uploader("", type=["xlsx"])
    if uploaded_excel:
        st.session_state['uploaded_excel'] = uploaded_excel
        st.session_state['dataframe'] = pd.read_excel(uploaded_excel)

    if st.session_state['uploaded_excel']:
        st.write(f"Excel File Uploaded: {st.session_state['uploaded_excel'].name}")

    # Access uploaded content from session state
    document_content = st.session_state.get('document_content', '')
    df = st.session_state.get('dataframe')

    if df is not None and document_content:
        st.subheader("Select the Validation Methods:")

        # Define criteria with associated info
        criteria_options = {
            "Example Answer": "This criterion checks how closely the generated answers match a provided example answer in the Excel file.",
            "Certain Criteria": "This criterion checks if the generated answers match with the provided criteria in the Excel file.",
            "Tone of Voice": "This criterion checks if the tone of voice that is specified matches the generated answers.",
            "Robustness": "This criterion checks for robustness using the ROUGE method.",
            "Accuracy": "This criterion checks for Accuracy using hallucination rate.",
            "Completeness": "This criterion checks if all expected information is mentioned in the summary."
        }

        # Storage for specific inputs tied to certain criteria
        specified_inputs = {}

        # Create a checklist for multiple criteria
        selected_criteria = []
        for key, info in criteria_options.items():
            if st.checkbox(key, key=key):
                selected_criteria.append(key)
                # Display the description text only if the checkbox is selected
                st.markdown(f"<small><em>{info}</em></small>", unsafe_allow_html=True)
                # Add input field directly after a specific checkbox
                if key == "Tone of Voice":
                    specified_inputs[key] = st.text_input("Specify the Tone of Voice:")

        if st.button("Submit"):
            results = []
            progress_text = st.empty()
            progress_bar = st.progress(0)

            total_steps = len(df) * len(selected_criteria)
            current_step = 0

            for criteria in selected_criteria:
                result_df, correct_counts = process_and_analyze(df, document_content, criteria, specified_inputs, total_steps, current_step, progress_bar, progress_text, llm)
                current_step += len(df)  # Increment current step by the number of rows processed
                if not result_df.empty:
                    results.append((criteria, result_df, correct_counts))

            st.session_state["results"] = results

def process_and_analyze(df, document_content, criteria, specified_input, total_steps, current_step, progress_bar, progress_text, llm):
    output_records = []
    correct_counts = []

    skip_processing = False
    missing_columns = []

    if 'Question' not in df.columns:
        missing_columns.append('Question')
        skip_processing = True

    if criteria == "Example Answer" and 'Example Answer' not in df.columns:
        missing_columns.append('Example Answer')
        skip_processing = True

    if criteria == "Certain Criteria" and 'Certain Criteria' not in df.columns:
        missing_columns.append('Certain Criteria')
        skip_processing = True

    if missing_columns:
        st.warning(f"Skipping {criteria}. Missing columns: {', '.join(missing_columns)}. Please ensure your file contains these columns.")

    if skip_processing:
        # Update progress to account for skipped steps
        progress_bar.progress(int((current_step + len(df)) / total_steps * 100))
        progress_text.text(f"Skipping {criteria}, moving to next criteria.")
        return pd.DataFrame(), []

    for idx, row in df.iterrows():
        original_question = row['Question']
        example_answer = row.get('Example Answer', '')
        certain_criteria = row.get('Certain Criteria', '')

        response = simple_query(original_question)
        expanded_questions = process_responses(response)
        answers = [answer_question(q, document_content, llm) for q in expanded_questions]

        validations = []
        for ans in answers:
            if criteria == "Example Answer":
                validation_response = validate_answer_example(ans, example_answer, llm)
            elif criteria == "Certain Criteria":
                validation_response = validate_answer_criteria(ans, certain_criteria, llm)
            elif criteria == "Tone of Voice":
                validation_response = evaluate_tone(ans, specified_input.get(criteria, ""), llm)
            elif criteria == "Robustness":
                validation_response = evaluate_robustness(ans, llm)
            elif criteria == "Accuracy":
                validation_response = evaluate_accuracy(ans, llm)
            elif criteria == "Completeness":
                validation_response = evaluate_completeness(ans, llm)

            status, reasoning = parse_validation_reason(validation_response)
            validations.append((status, reasoning, validation_response))  # Include the raw validation response

        count_correct = sum(1 for status, _, _ in validations if status == 'correct')
        correct_counts.append(count_correct)

        for q, a, (status, reasoning, raw_response) in zip(expanded_questions, answers, validations):
            record = {
                'Original Question': original_question,
                'Expanded Question': q,
                'Generated Answer': a,
                'Validation': status,
                'Reasoning': reasoning,
                'Validation Response': raw_response  # Include the validation response in the record
            }
            if criteria == "Example Answer":
                record['Example Answer'] = example_answer
            if criteria == "Certain Criteria":
                record['Certain Criteria'] = certain_criteria
            elif criteria in ["Tone of Voice", "Robustness", "Accuracy", "Completeness"]:
                record[criteria] = specified_input.get(criteria, "")
            output_records.append(record)

        current_step += 1
        progress_bar.progress(int((current_step / total_steps) * 100))
        progress_text.text(f"Processing step {current_step} of {total_steps}")

    return pd.DataFrame(output_records), correct_counts

def page_results():
    st.title("Results")
    
    if "results" in st.session_state:
        results = st.session_state["results"]

        # Top overview section for all questions
        st.subheader("Questions Overview")
        
        all_questions = set()
        for criteria, result_df, _ in results:
            all_questions.update(result_df['Original Question'].unique())

        for idx, question in enumerate(sorted(all_questions), start=1):
            st.markdown(f"**{idx}. {question}**")

        # Subheader for detailed results per criteria
        st.subheader("Results per Criteria")

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for criteria, result_df, correct_counts in results:
                if len(correct_counts) == 0 or result_df.empty:
                    st.warning(f"No data available for criteria: {criteria}. Skipping visualization and Excel export.")
                    continue

                # Create a container for the criteria results
                with st.container():
                    st.write(f"**Criteria: {criteria}**")

                    # Create columns for chart and table
                    col1, col2 = st.columns(2)  # Adjust column width ratios if needed

                    with col1:
                        # Plot chart
                        fig = create_plotly_chart(correct_counts)
                        st.plotly_chart(fig, key=f"plotly_chart_{criteria}")

                        st.markdown(f"This bar chart represents the count of correct answers derived from original questions with validations for {criteria}.")
                    
                    with col2:
                        # Table
                        st.markdown(f"**Detailed Reasoning for {criteria}**")
                        # Include 'Generated Answer' in the display dataframe
                        display_df = result_df[['Original Question', 'Expanded Question', 'Generated Answer', 'Validation', 'Reasoning']]
                        st.dataframe(display_df, height=400, use_container_width=True)

                # Write results into Excel sheet, including the generated answers
                result_df.to_excel(writer, index=False, sheet_name=f'Results_{criteria}')

        excel_data = output.getvalue()

        # Download section for Excel
        st.write("")
        st.write("---")
        st.write("### Download All Results")
        st.download_button(
            label="Download Comprehensive Analysis Results",
            data=excel_data,
            file_name='comprehensive_analysis_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.warning("No results to display. Please ensure you have processed some documents first.")

def page_overall_scores():
    st.title("Summary")

    if "results" in st.session_state:
        results = st.session_state["results"]
        question_scores_avg, criteria_scores_avg, scores_per_criteria = calculate_overall_scores(results)

        total_score = sum(question_scores_avg['Score']) / len(question_scores_avg['Score']) if len(question_scores_avg) > 0 else 0
        selected_criteria = {criteria for criteria, _, _ in results}

        st.markdown(f"<h3 style='text-align: center;'>Total overall score of all questions and criteria:", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: #FF6600;'>{total_score:.2f}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>Included Criteria: {', '.join(selected_criteria)}</h4>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Scores per Question")
            question_scores_avg['Question Number'] = range(1, len(question_scores_avg) + 1)

            fig_question = px.bar(
                question_scores_avg,
                x='Question Number',
                y='Score',
                labels={'Labeled Question': 'Original Question', 'Score': 'Average Score'},
                title='Average Scores per Question',
                color_discrete_sequence=['#FF6600']
            )

            # Set y-axis range from 0 to 1
            fig_question.update_layout(yaxis=dict(range=[0, 1]))

            fig_question.update_traces(
                textposition="inside",
                text=[f"{score:.2f}" for score in question_scores_avg['Score']],  
                textfont=dict(color="white", size=14, family="Arial Black"),
            )

            st.plotly_chart(fig_question)

            for _, row in question_scores_avg.iterrows():
                st.markdown(f"- **Q{row['Question Number']}**: {row['Question']}")

            st.markdown(f"**Criteria Selected:** {', '.join(selected_criteria)}")

        with col2:
            st.subheader("Scores per Criteria")
            fig_criteria = px.bar(
                x=list(criteria_scores_avg.keys()),
                y=list(criteria_scores_avg.values()),
                labels={'x': 'Criteria', 'y': 'Average Score'},
                title='Average Scores per Criteria',
                color_discrete_sequence=['#FF6600']
            )
            # Set y-axis range from 0 to 1
            fig_criteria.update_layout(yaxis=dict(range=[0, 1]))   

            fig_criteria.update_traces(
                textposition="inside",
                text=[f"{score:.2f}" for score in criteria_scores_avg.values()],
                textfont=dict(color="white", size=14, family="Arial Black"),
            )
            st.plotly_chart(fig_criteria)

            st.markdown("**Number of Questions per Criteria:**")
            for criteria, scores in scores_per_criteria.items():
                st.markdown(f"- {criteria}: {len(scores)} questions")
    else:
        st.warning("No results available for calculating overall scores. Please run the validation first.")

def main():
    st.sidebar.image("logo.png", use_container_width=True)
    st.sidebar.title("Navigation")

    page = st.sidebar.radio("Select Page", 
                            ("Upload and select validation", "Results", "Summary"))

    if page == "Upload and select validation":
        page_upload_and_analyze()
    elif page == "Results":
        page_results()
    elif page == "Summary":
        page_overall_scores()

if __name__ == "__main__":
    main()