from flask import Flask, render_template, request, jsonify
from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Try different LLM configurations
def get_llm():
    """Get the LLM with fallback options"""
    try:
        # Option 1: Use CrewAI's built-in LLM class
        from crewai.llm import LLM
        return LLM(
            model="gemini/gemini-1.5-flash",
            api_key=api_key,
            temperature=0.7
        )
    except Exception as e1:
        print(f"Option 1 failed: {e1}")
        try:
            # Option 2: Use direct string model specification
            return {
                "model": "gemini-1.5-flash", 
                "api_key": api_key,
                "temperature": 0.7
            }
        except Exception as e2:
            print(f"Option 2 failed: {e2}")
            # Option 3: Fallback to basic configuration
            return None

llm = get_llm()

# Define Agents with minimal configuration
def create_agents():
    """Create agents with error handling"""
    try:
        journalist_agent = Agent(
            role='Investigative Journalist',
            goal='Research topics and gather information',
            backstory='You research topics thoroughly.',
            llm=llm,
            verbose=False
        )

        news_writer_agent = Agent(
            role='News Writer',
            goal='Write engaging news articles',
            backstory='You write compelling news articles.',
            llm=llm,
            verbose=False
        )

        sentence_cleaner_agent = Agent(
            role='Content Editor',
            goal='Edit articles for clarity',
            backstory='You edit and improve text.',
            llm=llm,
            verbose=False
        )

        merger_agent = Agent(
            role='Final Editor',
            goal='Finalize articles',
            backstory='You prepare final content.',
            llm=llm,
            verbose=False
        )
        
        return journalist_agent, news_writer_agent, sentence_cleaner_agent, merger_agent
        
    except Exception as e:
        print(f"Error creating agents: {e}")
        return None, None, None, None

# Create agents
journalist_agent, news_writer_agent, sentence_cleaner_agent, merger_agent = create_agents()

def create_news_crew(topic):
    """Create news crew with error handling"""
    try:
        # Simple task definitions
        research_task = Task(
            description=f"Research the topic: {topic}. Provide key facts and information.",
            agent=journalist_agent,
            expected_output="Research findings about the topic."
        )
        
        writing_task = Task(
            description=f"Write a news article about {topic} based on the research.",
            agent=news_writer_agent,
            expected_output="A complete news article."
        )
        
        cleaning_task = Task(
            description="Edit the article for grammar and clarity.",
            agent=sentence_cleaner_agent,
            expected_output="A polished article."
        )
        
        merger_task = Task(
            description="Finalize the article for publication.",
            agent=merger_agent,
            expected_output="Final publication-ready article."
        )
        
        # Create crew with minimal configuration
        crew = Crew(
            agents=[journalist_agent, news_writer_agent, sentence_cleaner_agent, merger_agent],
            tasks=[research_task, writing_task, cleaning_task, merger_task],
            process=Process.sequential,
            verbose=False  # Reduce verbosity to avoid issues
        )
        
        return crew
        
    except Exception as e:
        print(f"Error creating crew: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_news', methods=['POST'])
def generate_news():
    try:
        data = request.get_json()
        topic = data.get('topic', '')
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        # Check if agents are available
        if not all([journalist_agent, news_writer_agent, sentence_cleaner_agent, merger_agent]):
            return jsonify({
                'success': False,
                'error': 'Agents not properly initialized. Please check your API key and try again.'
            }), 500
        
        print(f"Generating news for topic: {topic}")
        
        # Create crew
        crew = create_news_crew(topic)
        if not crew:
            return jsonify({
                'success': False,
                'error': 'Failed to create crew. Please try again.'
            }), 500
        
        print("Crew created successfully")
        
        # Execute crew
        result = crew.kickoff()
        print("Crew execution completed")
        
        return jsonify({
            'success': True,
            'article': str(result)
        })
        
    except Exception as e:
        print(f"Error in generate_news: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Provide a more helpful error message
        error_msg = str(e)
        if "API" in error_msg.upper():
            error_msg = "API connection failed. Please check your internet connection and API key."
        elif "MODEL" in error_msg.upper():
            error_msg = "Model configuration error. Please try again."
        else:
            error_msg = f"An error occurred: {error_msg}"
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/test_api')
def test_api():
    """Test endpoint to check if API is working"""
    try:
        if not api_key:
            return jsonify({'status': 'error', 'message': 'API key not found'})
        
        # Try to create a simple agent
        test_agent = Agent(
            role='Test Agent',
            goal='Test the API',
            backstory='Testing agent.',
            llm=llm,
            verbose=False
        )
        
        return jsonify({'status': 'success', 'message': 'API is working'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("Starting AI News Writer...")
    print(f"API Key configured: {'Yes' if api_key else 'No'}")
    print(f"LLM configured: {'Yes' if llm else 'No'}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
