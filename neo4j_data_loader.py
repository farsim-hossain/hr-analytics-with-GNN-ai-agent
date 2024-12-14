from neo4j import GraphDatabase
import pandas as pd

class Neo4jLoader:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_employee_nodes(self, employees_df):
        with self.driver.session() as session:
            for _, employee in employees_df.iterrows():
                query = """
                CREATE (e:Employee {
                    id: $emp_id,
                    experience_years: $experience,
                    performance_rating: $performance,
                    hours_worked_weekly: $hours,
                    skills: $skills
                })
                """
                session.run(query, {
                    'emp_id': int(employee['employee_id']),
                    'experience': int(employee['experience_years']),
                    'performance': float(employee['performance_rating']),
                    'hours': int(employee['hours_worked_weekly']),
                    'skills': employee['skills']
                })

    def create_relationships(self, interactions_df):
        with self.driver.session() as session:
            for _, interaction in interactions_df.iterrows():
                query = """
                MATCH (e1:Employee {id: $emp1_id})
                MATCH (e2:Employee {id: $emp2_id})
                CREATE (e1)-[r:COLLABORATED_WITH {
                    project_id: $proj_id,
                    communication_frequency: $comm_freq,
                    collaboration_strength: $collab_strength
                }]->(e2)
                """
                session.run(query, {
                    'emp1_id': int(interaction['employee1_id']),
                    'emp2_id': int(interaction['employee2_id']),
                    'proj_id': int(interaction['project_id']),
                    'comm_freq': int(interaction['communication_frequency']),
                    'collab_strength': float(interaction['collaboration_strength'])
                })

    def export_graph_data(self):
        with self.driver.session() as session:
            # Export nodes
            nodes = session.run("""
                MATCH (e:Employee)
                RETURN e.id AS id, 
                       e.experience_years AS experience,
                       e.performance_rating AS performance,
                       e.hours_worked_weekly AS hours
            """)
            
            # Export edges
            edges = session.run("""
                MATCH (e1:Employee)-[r:COLLABORATED_WITH]->(e2:Employee)
                RETURN e1.id AS source, 
                       e2.id AS target,
                       r.collaboration_strength AS weight,
                       r.communication_frequency AS communication
            """)
            
            return list(nodes), list(edges)

def main():
    # Neo4j cloud credentials
    URI = "neo4j+s://522e63e9.databases.neo4j.io"
    USERNAME = "neo4j"
    PASSWORD = "z7BR2qxJJbfxUvCiCTk1CqX9ED_jpBplZn8xd0ZXpuM"

    # Load CSV data
    employees_df = pd.read_csv('data/employees.csv')
    interactions_df = pd.read_csv('data/interactions.csv')

    # Initialize loader
    loader = Neo4jLoader(URI, USERNAME, PASSWORD)

    try:
        # Clear existing data
        loader.clear_database()

        # Create nodes and relationships
        loader.create_employee_nodes(employees_df)
        loader.create_relationships(interactions_df)

        # Export graph data for GNN
        nodes, edges = loader.export_graph_data()
        
        # Save exported data
        pd.DataFrame(nodes).to_csv('data/neo4j_nodes.csv', index=False)
        pd.DataFrame(edges).to_csv('data/neo4j_edges.csv', index=False)

    finally:
        loader.close()

if __name__ == "__main__":
    main()