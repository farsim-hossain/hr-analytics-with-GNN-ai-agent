import graphviz
from neo4j import GraphDatabase
import pandas as pd
import numpy as np

class NetworkVisualizer:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def generate_neo4j_query(self):
        return """
        MATCH (e1:Employee)-[r:COLLABORATED_WITH]->(e2:Employee)
        WITH e1, r, e2, COUNT {(e1)-[:COLLABORATED_WITH]-()} as degree_centrality
        SET e1.size = degree_centrality,
            e1.color = CASE 
                WHEN e1.performance_gap < -0.5 THEN '#FF0000'
                WHEN e1.performance_gap > 0.5 THEN '#00FF00'
                ELSE '#FFFF00'
            END,
            e1.cluster = CASE
                WHEN e1.performance_rating > 4.0 THEN 'High Performers'
                WHEN e1.performance_rating < 3.0 THEN 'Needs Improvement'
                ELSE 'Average Performers'
            END
        RETURN e1, r, e2

        """

    
    def create_graphviz_visualization(self):
        dot = graphviz.Digraph(comment='Team Performance Network')
        dot.attr(rankdir='LR')
        
        with self.driver.session() as session:
            # Get nodes
            nodes = session.run("""
                MATCH (e:Employee)
                RETURN e.id as id, 
                       e.performance_rating as performance,
                       e.predicted_performance as predicted,
                       e.performance_gap as gap
            """).data()
            
            # Get edges
            edges = session.run("""
                MATCH (e1:Employee)-[r:COLLABORATED_WITH]->(e2:Employee)
                RETURN e1.id as source, 
                       e2.id as target,
                       r.collaboration_strength as strength
            """).data()
            
            # Add nodes to visualization
            for node in nodes:
                color = 'red' if node['gap'] < -0.5 else 'green' if node['gap'] > 0.5 else 'yellow'
                dot.node(str(node['id']), 
                        f"Employee {node['id']}\nPerf: {node['performance']:.2f}",
                        color=color,
                        style='filled',
                        fillcolor=color + ':white')
            
            # Add edges
            for edge in edges:
                width = str(1 + float(edge['strength']) * 3)
                dot.edge(str(edge['source']), 
                        str(edge['target']),
                        penwidth=width)
        
        return dot

def main():
    URI = "neo4j uri"
    USERNAME = ""
    PASSWORD = ""
    
    visualizer = NetworkVisualizer(URI, USERNAME, PASSWORD)
    
    # Generate Neo4j visualization query
    neo4j_query = visualizer.generate_neo4j_query()
    print("\nNeo4j Browser Query (Copy and run in Neo4j Browser):")
    print(neo4j_query)
    
    # Create Graphviz visualization
    dot = visualizer.create_graphviz_visualization()
    
    # Save visualizations
    dot.render('visualizations/team_network', format='png', cleanup=True)
    print("\nGraphviz visualization saved as 'visualizations/team_network.png'")

if __name__ == "__main__":
    main()
