import streamlit as st
import os
import base64
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document
from io import BytesIO
from crewai import Agent, Task, Crew, Process
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation

load_dotenv()

# Initialize Groq client
groq_client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

def audio_to_base64(file):
    """Convert audio file to base64 for embedding in HTML"""
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
        base64_audio = base64.b64encode(audio_bytes).decode()
    return base64_audio

def extract_section(text, start_marker, end_marker=None):
    """Extract a section of text between two markers"""
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    start_idx += len(start_marker)
    if end_marker:
        end_idx = text.find(end_marker, start_idx)
        return text[start_idx:end_idx].strip() if end_idx != -1 else text[start_idx:].strip()
    return text[start_idx:].strip()

def clean_text(text):
    """Remove markdown formatting and extra spaces"""
    text = text.replace('**', '').replace('*', '').replace('#', '')
    # Collapse multiple newlines into one
    text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
    return text

def format_report_as_html(report):
    """Format report into a clean HTML structure"""
    section_order = [
        ('Incident ID', 'Incident ID:', 'Date:'),
        ('Date', 'Date:', 'Location:'),
        ('Location', 'Location:', 'Aircraft:'),
        ('Aircraft', 'Aircraft:', 'Flight Number:'),
        ('Flight Number', 'Flight Number:', 'Type of Operation:'),
        ('Type of Operation', 'Type of Operation:', 'Passengers and Crew:'),
        ('Passengers and Crew', 'Passengers and Crew:', 'Injuries and Fatalities:'),
        ('Injuries and Fatalities', 'Injuries and Fatalities:', 'Damage:'),
        ('Damage', 'Damage:', 'Summary of Incident:'),
        ('Summary', 'Summary of Incident:', 'Factors Contributing to the Incident:'),
        ('Analysis', 'Factors Contributing to the Incident:', 'Actions Taken:'),
        ('Actions Taken', 'Actions Taken:', 'Recommendations:'),
        ('Recommendations', 'Recommendations:', None)
    ]

    sections = {}
    for i in range(len(section_order)):
        title, start_marker, end_marker_hint = section_order[i]
        end_marker = section_order[i + 1][1] if i < len(section_order) - 1 and section_order[i + 1][1] else None
        content = extract_section(report, start_marker, end_marker)
        if content:
            sections[title] = clean_text(content)
        elif title == 'Recommendations':
            content = extract_section(report, start_marker)
            if content:
                sections[title] = clean_text(content)

    html_content = """
<style>
        .report-container {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: auto;
            padding: 30px;
            background-color: #f9f9f9;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .report-title {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .section {
            margin-bottom: 20px;
        }
        .section h5 {
            color: #3498db;
        }
        .section p {
            white-space: pre-line;
        }
</style>
<div class="report-container">
<h2 class="report-title">Flight Incident Investigation Report</h2>
    """

    for title, content in sections.items():
        html_content += f'<div class="section"><h5>{title}</h5><p>{content.strip()}</p></div>'

    html_content += "</div>"
    return html_content

def generate_docx(report_text):
    """Generate a Word document from the report"""
    doc = Document()
    doc.add_heading('Flight Incident Investigation Report', 0)

    section_order = [
        ('Incident ID', 'Incident ID:', 'Date:'),
        ('Date', 'Date:', 'Location:'),
        ('Location', 'Location:', 'Aircraft:'),
        ('Aircraft', 'Aircraft:', 'Flight Number:'),
        ('Flight Number', 'Flight Number:', 'Type of Operation:'),
        ('Passengers and Crew', 'Passengers and Crew:', 'Injuries and Fatalities:'),
        ('Injuries and Fatalities', 'Injuries and Fatalities:', 'Damage:'),
        ('Summary', 'Summary of Incident:', 'Factors Contributing to the Incident:'),
        ('Analysis', 'Factors Contributing to the Incident:', 'Actions Taken:'),
        ('Actions Taken', 'Actions Taken:', 'Recommendations:'),
        ('Recommendations', 'Recommendations:', None)
    ]

    for i in range(len(section_order)):
        title, start_marker, end_marker_hint = section_order[i]
        end_marker = section_order[i + 1][1] if i < len(section_order) - 1 and section_order[i + 1][1] else None
        content = extract_section(report_text, start_marker, end_marker)
        if content:
            doc.add_heading(title, level=1)
            doc.add_paragraph(clean_text(content.strip()))
        elif title == 'Recommendations':
            content = extract_section(report_text, start_marker)
            if content:
                doc.add_heading(title, level=1)
                doc.add_paragraph(clean_text(content.strip()))

    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

def get_download_link(bio, filename):
    """Generate a download link for the document"""
    b64 = base64.b64encode(bio.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="{filename}" style="padding: 10px 20px; background-color: #3498db; color: white; text-decoration: none; border-radius: 4px; display: inline-block;">Download Report</a>'

# Custom Langchain-like LLM wrapper for Groq
class GroqLLM(BaseLLM):
    """Wrapper around Groq API."""

    client: OpenAI
    model: str = "groq/llama-3.1-8b-instant"  # Explicitly set the model with provider
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: list[str] | None = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        stop_sequences = list(stop) if isinstance(stop, tuple) else stop
        completion = self.client.chat.completions.create(
            model=self.model.split('/')[1],  # Use the base model name for the Groq API call
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop=stop_sequences,
        )
        return completion.choices[0].message.content

    def _generate(self, prompts: list[str], stop: list[str] | None = None) -> LLMResult:
        generations = []
        stop_sequences = list(stop) if isinstance(stop, tuple) else stop
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            completion = self.client.chat.completions.create(
                model=self.model.split('/')[1],  # Use the base model name for the Groq API call
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop=stop_sequences,
            )
            generations.append([Generation(text=completion.choices[0].message.content)])
        return LLMResult(generations=generations)

# Streamlit UI Configuration
st.set_page_config(
    layout="wide",
    page_title="Black Box Archive Analysis"
)
st.title("Black Box Archive Analysis")

# Initialize session state for chat
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create three columns (left for inputs, middle for report, right for chat)
left_col, mid_col, right_col = st.columns([1, 2, 1])

with left_col:
    st.header("Upload & Analyze")
    uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"], key="uploader")

    if uploaded_file is not None:
        with open("uploaded_file.mp3", "wb") as f:
            f.write(uploaded_file.getbuffer())

        base64_audio = audio_to_base64("uploaded_file.mp3")
        st.subheader("Your Uploaded Audio File")
        st.markdown(f"""
    <audio controls style="width: 100%;">
    <source src="data:audio/mp3;base64,{base64_audio}" type="audio/mp3">
                Your browser does not support the audio element.
    </audio>
        """, unsafe_allow_html=True)

        if st.button("Analyze", type="primary", key="analyze_btn"):
            with st.spinner("Transcribing audio..."):
                with open("uploaded_file.mp3", "rb") as audio_file:
                    transcript = groq_client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=audio_file,
                        response_format="text"
                    )
            st.session_state.transcript = transcript
            st.success("Transcription Complete!")
            st.text_area("Transcription", transcript, height=200)

            with st.spinner("Generating Report using CrewAI..."):
                # Initialize Groq LLM
                groq_llm = GroqLLM(client=groq_client)

                # Define Agents
                transcription_reader = Agent(
                    role="ATC Blackbox Transcription Reader",
                    goal="Read the transcription from an ATC Blackbox and clean the transcription and create a conversation flow.",
                    backstory="This agent specializes in reading the transcript from an ATC Blackbox and clean the transcript to create a conversation flow so it is easier to interpret.",
                    verbose=True,
                    allow_delegation=False,
                    llm=groq_llm
                )

                incident_report_writer = Agent(
                    role="Incident Reporter",
                    goal="Generate a detailed incident investigation report based on the transcription data.",
                    backstory="This agent specializes in creating comprehensive incident investigation reports for aviation incidents. It considers the cleaned transcription data, communication logs, and aviation protocols to provide a detailed report.",
                    verbose=True,
                    allow_delegation=False,
                    llm=groq_llm
                )

                # Task for the transcription reader agent
                clean_transcription_task = Task(
                    description='Read the transcription from an ATC Blackbox, clean it, and create a conversation flow. The transcription is: ' + transcript,
                    agent=transcription_reader,
                    expected_output='Formatted transcription as conversation.',
                )

                # Task for the incident report writer agent
                incident_report_task = Task(
                    description='Generate a detailed incident investigation report based on the cleaned transcription provided by the transcription reader.',
                    agent=incident_report_writer,
                    expected_output='Detailed incident investigation report.'
                )
                # Create Crew
                crew = Crew(
                    agents=[transcription_reader, incident_report_writer],
                    tasks=[clean_transcription_task, incident_report_task],
                    process=Process.sequential,
                    verbose=True
                )

                report_text = str(crew.kickoff())
                report_text = clean_text(report_text)
                st.session_state.report_text = report_text

                # Initialize chat with report context
                st.session_state.messages = [
                    {"role": "system", "content": f"""You are an expert aviation assistant.
                     Answer questions about this flight incident report: {report_text}"""}
                ]

with mid_col:
    st.header("Generated Report")
    if 'report_text' in st.session_state:
        st.markdown(format_report_as_html(st.session_state.report_text), unsafe_allow_html=True)
        doc_bio = generate_docx(st.session_state.report_text)
        download_link = get_download_link(doc_bio, "flight_incident_report.docx")
        st.markdown(download_link, unsafe_allow_html=True)
    else:
        st.info("Upload an audio file and click 'Analyze' to generate a report")

with right_col:
    # Custom CSS for the compact chat interface
    st.markdown("""
    <style>
        .chat-header {
            padding: 12px;
            background: #2c3e50;
            color: white;
            font-weight: bold;
            margin: 0 !important;
        }
        .chat-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            background: #f9f9f9;
        }
        .chat-input-container {
            padding: 12px;
            border-top: 1px solid #e1e1e1;
            background: white;
        }
        .user-message {
            background: #e3f2fd;
            border-radius: 18px 18px 0 18px;
            padding: 10px 14px;
            margin-left: auto;
            margin-bottom: 8px;
            max-width: 85%;
        }
        .assistant-message {
            background: #f5f5f5;
            border-radius: 18px 18px 18px 0;
            padding: 10px 14px;
            margin-bottom: 8px;
            max-width: 85%;
        }
        .welcome-message {
            background: #f5f5f5;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 12px;
        }
        .stTextInput>div>div>input {
            border-radius: 18px;
            padding: 8px 16px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main chat container with fixed height
    with st.container():

        # Header
        st.markdown('<div class="chat-header">Report Assistant</div>', unsafe_allow_html=True)

        # Content area (messages + input)
        st.markdown('<div class="chat-content">', unsafe_allow_html=True)

        # Messages area
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

        # Welcome message (only first time)
        if len(st.session_state.chat_history) == 0:
            st.markdown("""
            <div class="welcome-message">
            <b>How can I help with this report?</b><br>
            Ask about:<br>
            • Incident details<br>
            • Safety recommendations<br>
            • Contributing factors
            </div>
            """, unsafe_allow_html=True)

        # Display conversation history
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # Close chat-messages

        # Input area (inside the same container)
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input(
                "Ask about the incident report",
                key="user_query",
                label_visibility="collapsed",
                placeholder="Type your question..."
            )
            submitted = st.form_submit_button("Send", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)  # Close chat-input-container

        st.markdown('</div>', unsafe_allow_html=True)  # Close chat-content

    # Handle form submission
    if submitted and user_input.strip():
        if 'report_text' not in st.session_state:
            st.error("Please generate a report first")
        else:
            # Add user question to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Generate answer
            prompt = f"""
            Based on this flight incident report:
            {st.session_state.report_text}

            Question: {user_input}

            Provide a concise 1-2 sentence answer using only the report's information.
            If the answer isn't in the report, respond: "This information is not in the report."
            """

            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )

            answer = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Auto-scroll to bottom
            st.markdown("""
            <script>
                setTimeout(function() {
                    const container = window.parent.document.querySelector('.chat-messages');
                    container.scrollTop = container.scrollHeight;
                }, 100);
            </script>
            """, unsafe_allow_html=True)

            # Rerun to update display
            st.rerun()