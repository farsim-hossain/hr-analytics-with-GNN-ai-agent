from groq import Groq
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import torch
import json

class HRAnalysisAgent:
    def __init__(self):
        load_dotenv()
        
        # Initialize Neo4j connection
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
        )
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        # Load GNN model with weights_only=True
        self.model = torch.load('models/team_performance_gnn.pth', weights_only=True)
        
    def get_employee_data(self, employee_id):
        with self.neo4j_driver.session() as session:
            query = """
            MATCH (e:Employee)
            WHERE e.id = $emp_id
            RETURN e {
                .*,
                predicted_performance: e.predicted_performance,
                performance_gap: e.performance_gap,
                experience_years: e.experience_years,
                performance_rating: e.performance_rating
            } as employee_data
            """
            result = session.run(query, emp_id=employee_id).single()
            return result['employee_data'] if result else None
        
    def generate_analysis_report(self, employee_data):
        prompt = f"""
        Generate a detailed HR analysis report for Employee ID {employee_data['id']}:

        Performance Metrics:
        - Current Performance Rating: {employee_data['performance_rating']}
        - Predicted Performance: {employee_data['predicted_performance']}
        - Performance Gap: {employee_data['performance_gap']}
        - Years of Experience: {employee_data['experience_years']}

        Provide insights on:
        1. Current performance evaluation
        2. Performance trajectory and potential
        3. Development opportunities
        4. Specific recommendations for improvement
        
        Format as a structured report with clear sections.
        """
        
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.7,
        )
        
        return response.choices[0].message.content
