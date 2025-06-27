from flask import Flask, render_template, request, jsonify
from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Initialize LLM with CrewAI's built-in support
from crewai.llm import LLM

llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=api_key,
    temperature=0.7
)

# Define AI Code Generator Agents
flowchart_agent = Agent(
    role='Flowchart Generator',
    goal='Create detailed flowcharts and visual representations of application logic',
    backstory='You are an expert system analyst who creates clear, comprehensive flowcharts and diagrams to visualize application workflow and logic.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

choice_provider_agent = Agent(
    role='Technology Choice Provider',
    goal='Recommend the best technology stack based on requirements',
    backstory='You are a senior architect who evaluates requirements and recommends optimal technology choices between Streamlit, Flask with HTML/CSS, or pure HTML/CSS solutions.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

architecture_agent = Agent(
    role='System Architecture Designer',
    goal='Design comprehensive system architecture based on chosen technology',
    backstory='You are a system architect who creates detailed technical architecture, including file structure, component relationships, and system design patterns.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

code_generator_agent = Agent(
    role='Code Generator',
    goal='Generate complete, functional code based on architecture specifications',
    backstory='You are an expert full-stack developer who writes clean, efficient, and well-structured code in multiple programming languages and frameworks.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

design_enhancer_agent = Agent(
    role='Design & Animation Enhancer',
    goal='Enhance code with modern UI/UX, animations, and advanced features',
    backstory='You are a creative frontend developer who specializes in modern UI/UX design, CSS animations, and interactive user experiences.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

error_fixer_agent = Agent(
    role='Code Error Fixer',
    goal='Identify and fix all errors, bugs, and issues in the code',
    backstory='You are a debugging expert who systematically identifies and resolves code errors, ensuring clean, bug-free, and optimized code.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

testing_agent = Agent(
    role='Code Testing Specialist',
    goal='Test code thoroughly to ensure error-free execution and proper functionality',
    backstory='You are a QA engineer who creates comprehensive test cases, validates functionality, and ensures code reliability and performance.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

delivery_agent = Agent(
    role='Code Delivery Specialist',
    goal='Package and present the final code solution with proper documentation',
    backstory='You are a technical documentation expert who organizes, formats, and presents code solutions with clear instructions, architecture diagrams, and user guides.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

def create_code_generation_crew(project_description, requirements):
    # Define Tasks for AI Code Generation
    flowchart_task = Task(
        description=f"""Create a detailed flowchart for the project: {project_description}
        
        Requirements: {requirements}
        
        Generate:
        1. Application workflow diagram
        2. User interaction flow
        3. Data flow diagram
        4. Component relationship diagram
        5. Process flow with decision points
        
        Provide a detailed textual representation of the flowchart.""",
        agent=flowchart_agent,
        expected_output="A comprehensive flowchart description with all workflow components and relationships."
    )
    
    choice_task = Task(
        description=f"""Analyze the project requirements and recommend the best technology stack:
        
        Project: {project_description}
        Requirements: {requirements}
        
        Available Options:
        1. Streamlit (for data-focused, rapid prototyping)
        2. Flask with HTML/CSS (for full-stack web applications)
        3. Pure HTML/CSS/JavaScript (for frontend-only solutions)
        
        Provide:
        1. Recommended technology choice with reasoning
        2. Pros and cons of the chosen technology
        3. Alternative options if applicable
        4. Required dependencies and setup""",
        agent=choice_provider_agent,
        expected_output="Technology recommendation with detailed justification and setup requirements."
    )
    
    architecture_task = Task(
        description=f"""Design comprehensive system architecture based on the chosen technology:
        
        Project: {project_description}
        Technology Choice: [From previous task]
        
        Create:
        1. Detailed file structure
        2. Component architecture
        3. Database design (if needed)
        4. API endpoints (if applicable)
        5. Security considerations
        6. Deployment strategy
        7. Configuration requirements""",
        agent=architecture_agent,
        expected_output="Complete system architecture with file structure, components, and technical specifications."
    )
    
    code_generation_task = Task(
        description=f"""Generate complete, functional code based on the architecture:
        
        Project: {project_description}
        Architecture: [From previous task]
        
        Generate:
        1. All necessary code files
        2. Configuration files
        3. Requirements/dependencies
        4. Database schemas (if needed)
        5. API implementations
        6. Frontend components
        7. Main application logic
        
        Ensure code is clean, well-commented, and follows best practices.""",
        agent=code_generator_agent,
        expected_output="Complete codebase with all necessary files and implementations."
    )
    
    design_enhancement_task = Task(
        description="""Enhance the generated code with modern design and animations:
        
        Add:
        1. Modern, responsive UI design
        2. CSS animations and transitions
        3. Interactive elements
        4. Loading states and progress indicators
        5. Error handling with user-friendly messages
        6. Mobile-responsive design
        7. Accessibility features
        8. Performance optimizations
        
        Focus on creating an engaging, professional user experience.""",
        agent=design_enhancer_agent,
        expected_output="Enhanced code with modern UI/UX, animations, and interactive features."
    )
    
    error_fixing_task = Task(
        description="""Review and fix all errors in the code:
        
        Check for:
        1. Syntax errors
        2. Logic errors
        3. Import/dependency issues
        4. Configuration problems
        5. Security vulnerabilities
        6. Performance issues
        7. Compatibility problems
        8. Code optimization opportunities
        
        Provide clean, error-free, optimized code.""",
        agent=error_fixer_agent,
        expected_output="Clean, error-free, and optimized code with all issues resolved."
    )
    
    testing_task = Task(
        description="""Test the code comprehensively:
        
        Perform:
        1. Unit testing
        2. Integration testing
        3. User interface testing
        4. Performance testing
        5. Error handling testing
        6. Edge case testing
        7. Cross-browser/platform testing
        8. Security testing
        
        Provide test results and ensure everything works correctly.""",
        agent=testing_agent,
        expected_output="Comprehensive test results confirming error-free execution and proper functionality."
    )
    
    delivery_task = Task(
        description="""Package and present the final code solution:
        
        Provide:
        1. Complete, organized codebase
        2. Detailed README with setup instructions
        3. Architecture documentation
        4. API documentation (if applicable)
        5. User guide
        6. Deployment instructions
        7. Troubleshooting guide
        8. Future enhancement suggestions
        
        Present everything in a professional, easy-to-understand format.""",
        agent=delivery_agent,
        expected_output="Complete, well-documented code solution ready for deployment and use."
    )
    
    # Create Crew
    crew = Crew(
        agents=[
            flowchart_agent, choice_provider_agent, architecture_agent, 
            code_generator_agent, design_enhancer_agent, error_fixer_agent,
            testing_agent, delivery_agent
        ],
        tasks=[
            flowchart_task, choice_task, architecture_task,
            code_generation_task, design_enhancement_task, error_fixing_task,
            testing_task, delivery_task
        ],
        process=Process.sequential,
        verbose=True
    )
    
    return crew

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_code', methods=['POST'])
def generate_code():
    try:
        data = request.get_json()
        project_description = data.get('project_description', '')
        requirements = data.get('requirements', '')
        
        if not project_description:
            return jsonify({'error': 'Project description is required'}), 400
        
        print(f"🚀 Starting code generation for project: {project_description}")
        
        # Create and run the crew
        crew = create_code_generation_crew(project_description, requirements)
        print("✅ AI Code Generation Crew created successfully")
        
        result = crew.kickoff()
        print("🎉 Code generation completed successfully")
        
        return jsonify({
            'success': True,
            'generated_code': str(result),
            'project_description': project_description,
            'requirements': requirements,
            'stats': {
                'agents_used': 8,
                'processing_phases': [
                    'Flowchart Generation',
                    'Technology Selection',
                    'Architecture Design',
                    'Code Generation',
                    'Design Enhancement',
                    'Error Fixing',
                    'Testing',
                    'Final Delivery'
                ],
                'word_count': len(str(result).split()) if result else 0,
                'lines_of_code': str(result).count('\n') if result else 0,
                'processing_time': '45-120 seconds'
            }
        })
        
    except Exception as e:
        print(f"❌ Error in generate_code: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f"Failed to generate code: {str(e)}"
        }), 500

@app.route('/agent_status', methods=['GET'])
def agent_status():
    """Endpoint to get agent status"""
    return jsonify({
        'agents': [
            {'name': 'Flowchart Generator', 'status': 'ready', 'emoji': '📊', 'phase': 1},
            {'name': 'Technology Choice Provider', 'status': 'ready', 'emoji': '�', 'phase': 2},
            {'name': 'Architecture Designer', 'status': 'ready', 'emoji': '🏗️', 'phase': 3},
            {'name': 'Code Generator', 'status': 'ready', 'emoji': '💻', 'phase': 4},
            {'name': 'Design Enhancer', 'status': 'ready', 'emoji': '🎨', 'phase': 5},
            {'name': 'Error Fixer', 'status': 'ready', 'emoji': '�', 'phase': 6},
            {'name': 'Testing Specialist', 'status': 'ready', 'emoji': '🧪', 'phase': 7},
            {'name': 'Delivery Specialist', 'status': 'ready', 'emoji': '�', 'phase': 8}
        ]
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'AI Code Generator'})

@app.route('/project_templates', methods=['GET'])
def get_project_templates():
    """Get sample project templates for inspiration"""
    templates = [
        {
            'name': 'E-commerce Website',
            'description': 'Full-stack e-commerce platform with payment integration',
            'tech_stack': 'React, Node.js, MongoDB, Stripe',
            'features': ['Product catalog', 'Shopping cart', 'User authentication', 'Payment processing']
        },
        {
            'name': 'Task Management App',
            'description': 'Project management tool with team collaboration',
            'tech_stack': 'Vue.js, Express.js, PostgreSQL',
            'features': ['Task tracking', 'Team collaboration', 'File sharing', 'Real-time updates']
        },
        {
            'name': 'Social Media Dashboard',
            'description': 'Analytics dashboard for social media management',
            'tech_stack': 'React, Python Flask, Chart.js',
            'features': ['Analytics visualization', 'Post scheduling', 'Multi-platform support']
        },
        {
            'name': 'Blog Platform',
            'description': 'Content management system for blogging',
            'tech_stack': 'Next.js, Prisma, PostgreSQL',
            'features': ['Rich text editor', 'Comment system', 'SEO optimization', 'User management']
        }
    ]
    
    return jsonify({
        'success': True,
        'templates': templates
    })

@app.route('/generation_status/<job_id>', methods=['GET'])
def get_generation_status(job_id):
    """Get the status of a code generation job (placeholder for future implementation)"""
    return jsonify({
        'success': True,
        'job_id': job_id,
        'status': 'processing',
        'progress': 75,
        'current_phase': 'Code Generation',
        'estimated_time_remaining': '30 seconds'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
