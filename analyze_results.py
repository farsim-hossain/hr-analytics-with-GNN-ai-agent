from neo4j import GraphDatabase
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class PerformanceAnalyzer:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def load_predictions(self, model_path='models/team_performance_gnn.pth'):
        predictions_df = pd.read_csv('data/predictions.csv')
        return predictions_df
        
    def update_neo4j_with_predictions(self, predictions_df):
        with self.driver.session() as session:
            # Add predictions as node properties
            query = """
            MATCH (e:Employee)
            WHERE e.id = $emp_id
            SET e.predicted_performance = $pred_perf,
                e.performance_gap = $perf_gap
            """
            
            for _, row in predictions_df.iterrows():
                session.run(query, {
                    'emp_id': int(row['employee_id']),
                    'pred_perf': float(row['predicted_performance']),
                    'perf_gap': float(row['actual_performance'] - row['predicted_performance'])
                })

    def identify_performance_loopholes(self):
        with self.driver.session() as session:
            # Find underperforming employees with strong features
            query = """
            MATCH (e:Employee)
            WHERE e.performance_gap < -0.5 
            AND e.experience_years > 5
            RETURN e.id as employee_id, 
                   e.experience_years as experience,
                   e.performance_rating as actual_performance,
                   e.predicted_performance as predicted_performance,
                   e.performance_gap as gap
            ORDER BY e.performance_gap ASC
            """
            loopholes = session.run(query).data()
            
            # Analyze weak collaborations
            weak_collabs_query = """
            MATCH (e1:Employee)-[r:COLLABORATED_WITH]->(e2:Employee)
            WHERE r.collaboration_strength < 0.3
            AND (e1.performance_rating > 3.5 OR e2.performance_rating > 3.5)
            RETURN e1.id as emp1_id, 
                   e2.id as emp2_id,
                   r.collaboration_strength as collab_strength,
                   r.communication_frequency as comm_freq
            """
            weak_collaborations = session.run(weak_collabs_query).data()
            
            # Team dynamics analysis
            team_query = """
            MATCH (e:Employee)
            WITH avg(e.performance_rating) as avg_performance,
                avg(e.predicted_performance) as avg_predicted,
                stDev(e.performance_gap) as performance_variance
            RETURN avg_performance, avg_predicted, performance_variance
            """

            team_metrics = session.run(team_query).data()[0]
            
            return loopholes, weak_collaborations, team_metrics

    def generate_recommendations(self):
        with self.driver.session() as session:
            # Find potential mentorship pairs
            mentorship_query = """
            MATCH (senior:Employee), (junior:Employee)
            WHERE senior.experience_years > 8 
            AND junior.performance_gap < -0.3
            AND NOT (senior)-[:COLLABORATED_WITH]-(junior)
            RETURN senior.id as mentor_id, 
                   junior.id as mentee_id,
                   senior.performance_rating as mentor_performance,
                   junior.performance_gap as mentee_gap
            LIMIT 5
            """
            mentorship_recommendations = session.run(mentorship_query).data()
            
            return mentorship_recommendations

def main():
    # Neo4j cloud credentials
    URI = "neo4j uri"
    USERNAME = ""
    PASSWORD = ""
    
    analyzer = PerformanceAnalyzer(URI, USERNAME, PASSWORD)
    
    # Load and update predictions
    predictions_df = analyzer.load_predictions()
    analyzer.update_neo4j_with_predictions(predictions_df)
    
    # Analyze results
    loopholes, weak_collaborations, team_metrics = analyzer.identify_performance_loopholes()
    recommendations = analyzer.generate_recommendations()
    
    # Generate report
    print("\n=== Team Performance Analysis Report ===")
    print("\nTeam Metrics:")
    print(f"Average Performance: {team_metrics['avg_performance']:.2f}")
    print(f"Average Predicted Performance: {team_metrics['avg_predicted']:.2f}")
    print(f"Performance Variance: {team_metrics['performance_variance']:.2f}")
    
    print("\nPerformance Loopholes:")
    for loophole in loopholes:
        print(f"Employee {loophole['employee_id']}: Gap = {loophole['gap']:.2f}")
    
    print("\nWeak Collaborations:")
    for collab in weak_collaborations:
        print(f"Employees {collab['emp1_id']} and {collab['emp2_id']}: "
              f"Strength = {collab['collab_strength']:.2f}")
    
    print("\nMentorship Recommendations:")
    for rec in recommendations:
        print(f"Mentor {rec['mentor_id']} -> Mentee {rec['mentee_id']}")

if __name__ == "__main__":
    main()
