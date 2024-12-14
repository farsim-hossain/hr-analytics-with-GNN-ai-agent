from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_groq import ChatGroq
from neo4j import GraphDatabase
import torch
from datetime import datetime
import os
import litellm

# Set up Groq
from dotenv import load_dotenv
import os

# Load environment variables at the start
load_dotenv()

# Set LiteLLM configuration
litellm.set_verbose = True

# Configure LiteLLM with the API key
import litellm
litellm.api_key = os.getenv("GROQ_API_KEY")

class HRAgentSystem:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, model_path):
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
# Update the ChatGroq initialization
        self.chat_model = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="groq/mixtral-8x7b-32768"  # Add 'groq/' prefix to specify the provider
        )

    def analyze_performance(self, query: str) -> str:
        with self.neo4j_driver.session() as session:
            underperformers = session.run("""
                MATCH (e:Employee)
                WHERE e.performance_gap < -0.5 AND e.experience_years > 5
                RETURN e.id, e.performance_rating, e.predicted_performance, e.performance_gap
            """).data()
            
            weak_collabs = session.run("""
                MATCH (e1:Employee)-[r:COLLABORATED_WITH]->(e2:Employee)
                WHERE r.collaboration_strength < 0.3
                RETURN e1.id, e2.id, r.collaboration_strength
            """).data()
            
            report = "Performance Analysis Report:\n\n"
            report += f"Found {len(underperformers)} underperforming experienced employees\n"
            report += f"Detected {len(weak_collabs)} weak collaboration links\n\n"
            
            for emp in underperformers:
                report += f"Employee {emp['e.id']}: Gap = {emp['e.performance_gap']:.2f}\n"
            
            return report

    def build_teams(self, query: str) -> str:
        with self.neo4j_driver.session() as session:
            high_performers = session.run("""
                MATCH (e:Employee)
                WHERE e.performance_rating > 4.0
                RETURN e.id, e.skills, e.performance_rating
            """).data()
            
            potential_teams = session.run("""
                MATCH (e1:Employee), (e2:Employee)
                WHERE e1.id <> e2.id
                AND e1.performance_rating > 3.5
                AND e2.performance_rating > 3.5
                AND NOT (e1)-[:COLLABORATED_WITH]-(e2)
                RETURN e1.id, e2.id, e1.skills, e2.skills
                LIMIT 5
            """).data()
            
            report = "Team Building Recommendations:\n\n"
            report += f"Found {len(high_performers)} high performers for team core\n"
            report += f"Identified {len(potential_teams)} potential new team combinations\n\n"
            
            for team in potential_teams:
                report += f"Suggested Team: Employee {team['e1.id']} + Employee {team['e2.id']}\n"
            
            return report

    def create_agents(self):
        performance_tool = Tool(
            name="analyze_performance",
            func=self.analyze_performance,
            description="Analyzes team performance and identifies issues"
        )

        team_building_tool = Tool(
            name="build_teams",
            func=self.build_teams,
            description="Recommends optimal team formations"
        )

        performance_analyzer = Agent(
            role="Performance Analysis Expert",
            goal="Analyze team dynamics and identify performance issues",
            backstory="Expert in analyzing HR data and team performance metrics",
            tools=[performance_tool],
            verbose=True,
            llm=self.chat_model
        )

        team_builder = Agent(
            role="Team Formation Specialist",
            goal="Suggest optimal team formations for projects",
            backstory="Specialist in team composition and collaboration optimization",
            tools=[team_building_tool],
            verbose=True,
            llm=self.chat_model
        )
        
        return [performance_analyzer, team_builder]


def main():
    hr_system = HRAgentSystem(
        neo4j_uri="neo4j+s://522e63e9.databases.neo4j.io",
        neo4j_user="neo4j",
        neo4j_password="z7BR2qxJJbfxUvCiCTk1CqX9ED_jpBplZn8xd0ZXpuM",
        model_path="models/team_performance_gnn.pth"
    )
    
    agents = hr_system.create_agents()
    
    tasks = [
        Task(
            description="Analyze team performance and identify issues",
            expected_output="A detailed report of performance issues and team dynamics",
            agent=agents[0]
        ),
        Task(
            description="Recommend optimal team formations",
            expected_output="A list of recommended team formations with justifications",
            agent=agents[1]
        )
    ]

    
    hr_crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential
    )
    
    result = hr_crew.kickoff()
    print("\n=== HR Analysis Results ===")
    print(result)

if __name__ == "__main__":
    main()
