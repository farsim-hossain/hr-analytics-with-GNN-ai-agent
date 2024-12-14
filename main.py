from hr_analysis_agent import HRAnalysisAgent
from team_analysis_agent import TeamAnalysisAgent



def main():
    hr_agent = HRAnalysisAgent()
    team_agent = TeamAnalysisAgent(hr_agent)
    
    # Specify the employee IDs you want to analyze
    employee_ids = [2, 4]  # Add any employee IDs you want to analyze
    
    for employee_id in employee_ids:
        try:
            employee_data = hr_agent.get_employee_data(employee_id)
            
            if employee_data:
                print(f"\n=== Individual Employee Analysis (Employee ID: {employee_id}) ===")
                individual_report = hr_agent.generate_analysis_report(employee_data)
                print(individual_report)
            
                # Team analysis for each employee's team
                team_id = employee_data.get('team_id')
                if team_id:
                    print(f"\n=== Team Analysis Report (Team ID: {team_id}) ===")
                    team_report = team_agent.analyze_team_dynamics(team_id)
                    print(team_report)
                    
        except Exception as e:
            print(f"Error analyzing employee {employee_id}: {str(e)}")
        
if __name__ == "__main__":
    main()

