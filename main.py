from hr_analysis_agent import HRAnalysisAgent
from team_analysis_agent import TeamAnalysisAgent

def main():
    hr_agent = HRAnalysisAgent()
    team_agent = TeamAnalysisAgent(hr_agent)
    
    # Use actual employee and team IDs from your database
    try:
        # Get all employee IDs first
        with hr_agent.neo4j_driver.session() as session:
            query = "MATCH (e:Employee) RETURN e.id as id LIMIT 1"
            employee_result = session.run(query).single()
            
            if employee_result:
                employee_id = employee_result['id']
                employee_data = hr_agent.get_employee_data(employee_id)
                
                if employee_data:
                    print("\n=== Individual Employee Analysis ===")
                    individual_report = hr_agent.generate_analysis_report(employee_data)
                    print(individual_report)
                
                # Use the employee's team_id for team analysis
                team_id = employee_data.get('team_id') if employee_data else None
                if team_id:
                    print("\n=== Team Analysis Report ===")
                    team_report = team_agent.analyze_team_dynamics(team_id)
                    print(team_report)
                    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        
if __name__ == "__main__":
    main()
