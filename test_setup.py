#!/usr/bin/env python3
"""
Simple test script to verify CrewAI setup
"""
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

# Load environment variables
load_dotenv()

def test_crewai_setup():
    """Test basic CrewAI functionality"""
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found!")
        return False
    
    print("✅ API key found")
    
    try:
        # Initialize LLM
        llm = LLM(
            model="gemini/gemini-1.5-flash",
            api_key=api_key,
            temperature=0.7
        )
        print("✅ LLM initialized successfully")
        
        # Create a simple agent
        test_agent = Agent(
            role='Test Writer',
            goal='Write a simple test response',
            backstory='You are a test agent.',
            llm=llm,
            verbose=True
        )
        print("✅ Agent created successfully")
        
        # Create a simple task
        test_task = Task(
            description="Write a short paragraph about artificial intelligence.",
            agent=test_agent,
            expected_output="A brief paragraph about AI."
        )
        print("✅ Task created successfully")
        
        # Create crew
        crew = Crew(
            agents=[test_agent],
            tasks=[test_task],
            process=Process.sequential,
            verbose=True
        )
        print("✅ Crew created successfully")
        
        # Execute the crew
        print("🚀 Starting crew execution...")
        result = crew.kickoff()
        print("✅ Crew executed successfully!")
        print(f"Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing CrewAI setup...")
    success = test_crewai_setup()
    
    if success:
        print("\n🎉 All tests passed! Your setup is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
