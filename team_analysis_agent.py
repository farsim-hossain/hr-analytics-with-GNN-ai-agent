import json

class TeamAnalysisAgent:
    def __init__(self, hr_agent):
        self.hr_agent = hr_agent
        
    def analyze_team_dynamics(self, team_id):
        with self.hr_agent.neo4j_driver.session() as session:
            # Query matching your existing database structure
            query = """
            MATCH (e:Employee)
            WHERE e.team_id = $team_id
            WITH collect(e {.*}) as team_members,
                 avg(e.performance_rating) as avg_performance,
                 avg(e.predicted_performance) as avg_predicted,
                 avg(e.performance_gap) as avg_gap
            RETURN team_members, avg_performance, avg_predicted, avg_gap
            """
            result = session.run(query, team_id=team_id).single()
            
            if not result:
                return "No team data found"
            
            # Enhanced prompt using actual data structure
            prompt = f"""
            Analyze the team dynamics based on the following metrics:
            Average Performance Rating: {result['avg_performance']}
            Average Predicted Performance: {result['avg_predicted']}
            Average Performance Gap: {result['avg_gap']}
            Number of Team Members: {len(result['team_members'])}
            
            Team Performance Analysis:
            1. Performance Metrics
            2. Collaboration Patterns
            3. Skills and Experience Distribution
            4. Recommendations for Improvement
            """
            
            response = self.hr_agent.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.7,
            )
            
            return response.choices[0].message.content