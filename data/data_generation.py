import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta

# Generate employee data
def generate_employee_data(num_employees=50):
    np.random.seed(42)
    
    skills = ['Python', 'Java', 'SQL', 'Leadership', 'Communication', 
              'Project Management', 'Data Analysis', 'UI/UX', 'DevOps']
    
    employees = []
    for emp_id in range(num_employees):
        employee = {
            'employee_id': emp_id,
            'experience_years': np.random.randint(1, 15),
            'performance_rating': round(np.random.normal(3.5, 0.5), 2),
            'hours_worked_weekly': np.random.randint(35, 50),
            'skills': np.random.choice(skills, size=np.random.randint(2, 5), replace=False).tolist()
        }
        employees.append(employee)
    
    return pd.DataFrame(employees)

# Generate project performance data
def generate_project_data(num_projects=20):
    projects = []
    start_date = datetime(2023, 1, 1)
    
    for proj_id in range(num_projects):
        project = {
            'project_id': proj_id,
            'start_date': start_date + timedelta(days=np.random.randint(0, 180)),
            'duration_weeks': np.random.randint(4, 24),
            'deadline_met': np.random.choice([True, False], p=[0.8, 0.2]),
            'quality_score': round(np.random.normal(8, 1), 2)
        }
        projects.append(project)
    
    return pd.DataFrame(projects)

# Generate interaction/collaboration data
def generate_interaction_data(employees_df, projects_df):
    interactions = []
    
    for project in projects_df.itertuples():
        team_size = np.random.randint(3, 8)
        team = np.random.choice(employees_df['employee_id'], size=team_size, replace=False)
        
        # Generate collaborations within the team
        for i in range(len(team)):
            for j in range(i+1, len(team)):
                interaction = {
                    'employee1_id': team[i],
                    'employee2_id': team[j],
                    'project_id': project.project_id,
                    'communication_frequency': np.random.randint(5, 50),
                    'collaboration_strength': round(np.random.uniform(0.1, 1.0), 2)
                }
                interactions.append(interaction)
    
    return pd.DataFrame(interactions)

# Create graph structure
def create_graph_structure(employees_df, interactions_df):
    G = nx.Graph()
    
    # Add nodes (employees)
    for _, employee in employees_df.iterrows():
        G.add_node(employee['employee_id'], 
                  experience=employee['experience_years'],
                  performance=employee['performance_rating'],
                  hours=employee['hours_worked_weekly'])
    
    # Add edges (interactions)
    for _, interaction in interactions_df.iterrows():
        G.add_edge(interaction['employee1_id'],
                  interaction['employee2_id'],
                  weight=interaction['collaboration_strength'],
                  communication=interaction['communication_frequency'])
    
    return G

# Main execution
def main():
    # Generate all data
    employees_df = generate_employee_data()
    projects_df = generate_project_data()
    interactions_df = generate_interaction_data(employees_df, projects_df)
    
    # Create graph
    G = create_graph_structure(employees_df, interactions_df)
    
    # Save data
    employees_df.to_csv('data/employees.csv', index=False)
    projects_df.to_csv('data/projects.csv', index=False)
    interactions_df.to_csv('data/interactions.csv', index=False)
    
    # Save graph
    nx.write_gexf(G, 'data/employee_network.gexf')
    
    return employees_df, projects_df, interactions_df, G

if __name__ == "__main__":
    employees_df, projects_df, interactions_df, G = main()
