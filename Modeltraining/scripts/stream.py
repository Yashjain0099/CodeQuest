import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re 
import time


MODEL_DIR = "./finalresults"


st.set_page_config(
    page_title="CodeQuest MCQ Generator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .main-header {
        text-align: center;
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 2px 2px 4px rgba(0,0,0,0.3), 0 0 10px rgba(255,255,255,0.2); }
        to { text-shadow: 2px 2px 4px rgba(0,0,0,0.3), 0 0 20px rgba(255,255,255,0.4); }
    }
    
    .sub-header {
        text-align: center;
        color: #e8eaf6;
        font-size: 1.3rem;
        font-weight: 400;
        margin-bottom: 3rem;
    }
    
    .domain-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 2px solid transparent;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .domain-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.5s;
    }
    
    .domain-card:hover::before {
        left: 100%;
    }
    
    .domain-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.2);
        border-color: #667eea;
    }
    
    .domain-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .domain-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .domain-description {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    
    .mcq-container {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border-left: 6px solid #667eea;
    }
    
    .mcq-question {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 2rem;
        line-height: 1.6;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    .option-button {
        display: block;
        width: 100%;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        background: white;
        color: #495057;
        font-size: 1.1rem;
        font-weight: 500;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .option-button:hover {
        border-color: #667eea;
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    
    .option-button.selected {
        border-color: #667eea;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        transform: translateX(5px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .option-button.correct {
        border-color: #28a745;
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        animation: pulse-correct 1s ease-in-out;
    }
    
    .option-button.incorrect {
        border-color: #dc3545;
        background: linear-gradient(135deg, #dc3545, #fd7e14);
        color: white;
        animation: shake 0.5s ease-in-out;
    }
    
    @keyframes pulse-correct {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    .option-letter {
        display: inline-block;
        width: 35px;
        height: 35px;
        background: #667eea;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 35px;
        font-weight: 600;
        margin-right: 1rem;
        font-size: 1rem;
    }
    
    .selected-domain-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .generate-btn {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 2rem auto;
        display: block;
        box-shadow: 0 8px 20px rgba(40, 167, 69, 0.3);
    }
    
    .generate-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(40, 167, 69, 0.4);
    }
    
    .back-btn {
        background: linear-gradient(135deg, #6c757d, #495057);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 2rem;
    }
    
    .back-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(108, 117, 125, 0.3);
    }
    
    .answer-reveal {
        background: linear-gradient(135deg, #17a2b8, #138496);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 10px 25px rgba(23, 162, 184, 0.3);
    }
    
    .score-display {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(255, 193, 7, 0.3);
    }
    
    .loading-container {
        text-align: center;
        padding: 2rem;
        color: white;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid rgba(255,255,255,0.3);
        border-top: 4px solid white;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .stat-item {
        background: rgba(255,255,255,0.2);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        backdrop-filter: blur(10px);
        margin: 0.5rem;
        min-width: 150px;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .reset-btn {
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(255,255,255,0.2);
        color: white;
        border: 2px solid white;
        padding: 10px 20px;
        border-radius: 25px;
        cursor: pointer;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .reset-btn:hover {
        background: white;
        color: #667eea;
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading with Caching ---
@st.cache_resource
def load_t5_model(model_path):

    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        st.success(f"✅ Model loaded successfully on {device}!")
        return model, tokenizer, device
    except Exception as e:
        st.error(f"❌ Error loading model from {model_path}: {e}")
        st.stop()

model, tokenizer, device = load_t5_model(MODEL_DIR)

# MCQ Parsing Function
def parse_mcq_output(text_output):
    question = ""
    options = {}
    correct_answer_key = ""

    # Extract  Answer part
    answer_match = re.search(r'Answer:\s*([A-D])', text_output)
    if answer_match:
        correct_answer_key = answer_match.group(1)
        text_output_without_answer = text_output[:answer_match.start()].strip()
    else:
        text_output_without_answer = text_output

    # Extract Question and Options
    parts = re.split(r'([A-D])\.\s*', text_output_without_answer)

    if parts and len(parts) > 1:
        question = parts[0].strip()
        current_key = None
        for i in range(1, len(parts)):
            if len(parts[i]) == 1 and parts[i] in ['A', 'B', 'C', 'D']:
                current_key = parts[i]
            elif current_key:
                options[current_key] = parts[i].split('A.')[0].split('B.')[0].split('C.')[0].split('D.')[0].strip()
                current_key = None

    if question.startswith(('A.', 'B.', 'C.', 'D.')):
        if "A." in text_output_without_answer:
            question = text_output_without_answer.split("A.")[0].strip()
    

    for key in ['A', 'B', 'C', 'D']:
        if key not in options:
            options[key] = ""
    
    sorted_options = {k: options[k] for k in sorted(options.keys())}
    return question, sorted_options, correct_answer_key

# MCQ Generation 
def generate_mcq(domain):
    """Generates an MCQ ."""
    prompt = f"Generate a multiple-choice question for domain: {domain}."
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=128,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    raw_mcq_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return raw_mcq_text

# Session State Initialization
if 'selected_domain' not in st.session_state:
    st.session_state.selected_domain = None
if 'generated_mcq' not in st.session_state:
    st.session_state.generated_mcq = {'question': "", 'options': {}, 'correct_answer': ""}
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None
if 'show_answer' not in st.session_state:
    st.session_state.show_answer = False
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'total_questions' not in st.session_state:
    st.session_state.total_questions = 0
if 'raw_output_text' not in st.session_state:
    st.session_state.raw_output_text = ""


if st.session_state.selected_domain:
    if st.button("🔄 Reset", key="reset", help="Go back to domain selection"):
        st.session_state.selected_domain = None
        st.session_state.selected_option = None
        st.session_state.show_answer = False
        st.session_state.generated_mcq = {'question': "", 'options': {}, 'correct_answer': ""}
        st.rerun()

# Main Application Layout
st.markdown('<h1 class="main-header">CodeQuest MCQ Generator</h1>', unsafe_allow_html=True)

if st.session_state.selected_domain is None:
    st.markdown('<p class="sub-header">AI-Powered Multiple Choice Questions for Developers</p>', unsafe_allow_html=True)
    
    # Stats section
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-item">
            <span class="stat-number">{st.session_state.total_questions}</span>
            <span class="stat-label">Questions Generated</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-item">
            <span class="stat-number">{st.session_state.score}</span>
            <span class="stat-label">Correct Answers</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        accuracy = (st.session_state.score / st.session_state.total_questions * 100) if st.session_state.total_questions > 0 else 0
        st.markdown(f"""
        <div class="stat-item">
            <span class="stat-number">{accuracy:.1f}%</span>
            <span class="stat-label">Accuracy</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-item">
            <span class="stat-number">3</span>
            <span class="stat-label">Domains</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Domain selection
    st.markdown('<h2 style="text-align: center; color: white; margin-bottom: 2rem;">Choose Your Domain</h2>', unsafe_allow_html=True)
    
    domains = {
        "Frontend": {
            "icon": "",
            "description": "HTML, CSS, JavaScript, React, Vue.js"
        },
        "Backend": {
            "icon": "", 
            "description": "Node.js, Python, Java, Databases, APIs"
        },
        "AI": {
            "icon": " ",
            "description": "Machine Learning, Deep Learning, NLP"
        }
    }
    
    
    cols = st.columns(3, gap="large")
    
    for i, (domain_name, domain_info) in enumerate(domains.items()):
        with cols[i]:
            card_html = f"""
            <div class="domain-card">
                <div class="domain-icon">{domain_info['icon']}</div>
                <h3 class="domain-title">{domain_name}</h3>
                <p class="domain-description">{domain_info['description']}</p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            if st.button(f"Select {domain_name}", key=f"select_{domain_name}", use_container_width=True):
                st.session_state.selected_domain = domain_name
                st.session_state.selected_option = None
                st.session_state.show_answer = False
                st.session_state.generated_mcq = {'question': "", 'options': {}, 'correct_answer': ""}
                st.rerun()

else:
    # --- Domain Selected View ---
    domain_icons = {"Frontend": "FRONTEND", "Backend": "BACKEND", "AI": "ARTIFICIAL INTELLIGENCE"}
    
    st.markdown(f"""
    <div class="selected-domain-header">
        <h1>{domain_icons.get(st.session_state.selected_domain, '🔹')} {st.session_state.selected_domain} MCQ Generator</h1>
        <p>Test your knowledge with AI-generated questions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(" Generate New MCQ", key="generate", use_container_width=True):
            st.session_state.selected_option = None
            st.session_state.show_answer = False
            st.session_state.generated_mcq = {'question': "", 'options': {}, 'correct_answer': ""}
            st.session_state.raw_output_text = ""

            with st.spinner(""):
                st.markdown("""
                <div class="loading-container">
                    <div class="loading-spinner"></div>
                    <p> AI is crafting your question...</p>
                </div>
                """, unsafe_allow_html=True)
                
                time.sleep(2)
                raw_output = generate_mcq(st.session_state.selected_domain)
                st.session_state.raw_output_text = raw_output
                
                question, options, correct_answer = parse_mcq_output(raw_output)
                
                if question:
                    st.session_state.generated_mcq = {
                        'question': question,
                        'options': options,
                        'correct_answer': correct_answer
                    }
                    st.session_state.total_questions += 1
                    st.rerun()
                else:
                    st.warning("⚠️ Could not parse a valid MCQ. Please try again.")
                    st.code(f"Raw output: {raw_output}")

    # Display Generated MCQ
    if st.session_state.generated_mcq['question']:
        st.markdown(f"""
        <div class="mcq-container">
            <div class="mcq-question">
                <strong>Question:</strong> {st.session_state.generated_mcq['question']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display options as clickable buttons
        for key, value in st.session_state.generated_mcq['options'].items():
            if value:
                # Determine button class based on state
                button_class = "option-button"
                if st.session_state.selected_option == key:
                    button_class += " selected"
                
                if st.session_state.show_answer:
                    if key == st.session_state.generated_mcq['correct_answer']:
                        button_class += " correct"
                    elif st.session_state.selected_option == key and key != st.session_state.generated_mcq['correct_answer']:
                        button_class += " incorrect"
                
                # Create clickable option
                option_html = f"""
                <div class="{button_class}" onclick="selectOption('{key}')">
                    <span class="option-letter">{key}</span>
                    {value}
                </div>
                """
                
                st.markdown(option_html, unsafe_allow_html=True)
                
                # Hidden button for functionality
                if st.button(f"Select {key}", key=f"opt_{key}", help=f"Select option {key}"):
                    st.session_state.selected_option = key
                    st.rerun()
        
        # Show answer button (only if option is selected)
        if st.session_state.selected_option and not st.session_state.show_answer:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("💡 Check Answer", key="check_answer", use_container_width=True):
                    st.session_state.show_answer = True
                    if st.session_state.selected_option == st.session_state.generated_mcq['correct_answer']:
                        st.session_state.score += 1
                    st.rerun()
        
        # Show answer result
        if st.session_state.show_answer:
            correct_key = st.session_state.generated_mcq['correct_answer']
            correct_option_text = st.session_state.generated_mcq['options'].get(correct_key, "Option not found")
            
            is_correct = st.session_state.selected_option == correct_key
            
            result_class = "correct" if is_correct else "incorrect"
            result_emoji = "🎉" if is_correct else "❌"
            result_text = "Correct!" if is_correct else "Incorrect!"
            
            st.markdown(f"""
            <div class="answer-reveal">
                {result_emoji} {result_text}<br>
                <strong>Correct Answer:</strong> {correct_key}. {correct_option_text}
            </div>
            """, unsafe_allow_html=True)
            
            # Show current score
            accuracy = (st.session_state.score / st.session_state.total_questions * 100) if st.session_state.total_questions > 0 else 0
            st.markdown(f"""
            <div class="score-display">
                📊 Score: {st.session_state.score}/{st.session_state.total_questions} ({accuracy:.1f}% accuracy)
            </div>
            """, unsafe_allow_html=True)
    
    # Footer actions
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Generate Another Question", use_container_width=True):
            st.session_state.selected_option = None
            st.session_state.show_answer = False
            st.session_state.generated_mcq = {'question': "", 'options': {}, 'correct_answer': ""}
            st.rerun()
    with col2:
        if st.button("Back to Domain Selection", use_container_width=True):
            st.session_state.selected_domain = None
            st.session_state.selected_option = None
            st.session_state.show_answer = False
            st.session_state.generated_mcq = {'question': "", 'options': {}, 'correct_answer': ""}
            st.rerun()

# Footer
st.markdown("""
<div class="footer">
    <p>codequest io - ai powered MCQ generator |  CodeQuest MCQ Generator</p>
</div>
""", unsafe_allow_html=True)