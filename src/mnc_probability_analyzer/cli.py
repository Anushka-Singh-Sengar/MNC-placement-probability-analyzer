import argparse
from pathlib import Path
import joblib
import pandas as pd


FEATURES = {
    'Google':   ['CGPA', 'Total Problems Solved', 'LeetCode Solved'],
    'Microsoft':['Technical Projects', 'Internships', 'Certifications', 'Total Problems Solved', 'Total Skills'],
    'Amazon':   ['Technical Projects', 'Internships', 'Certifications', 'LeetCode Solved', 'Teamwork Experience'],
    'Infosys':  ['Internships', 'Technical Projects', 'Certifications', 'LeetCode Solved', 'Teamwork Experience'],
}


def load_model(models_dir: str, company: str):
    path = Path(models_dir) / f"{company.lower()}.joblib"
    obj = joblib.load(path)
    return obj['model'], obj['features']


def interactive_mode(models_dir: str):
    print("\n=== MNC Placement Probability Analyzer ===")
    
    while True:
        # Company selection
        print("\nAvailable companies:")
        companies = list(FEATURES.keys())
        for i, comp in enumerate(companies, 1):
            print(f"{i}. {comp}")
        
        try:
            choice = int(input("\nEnter company number (or 0 to exit): "))
            if choice == 0:
                print("\nGoodbye!")
                break
            if choice < 1 or choice > len(companies):
                print(f"Please enter a number between 1 and {len(companies)}")
                continue
                
            company = companies[choice - 1]
            model, features = load_model(models_dir, company)
            
            # Get user input for each feature
            print(f"\nEnter your details for {company}:")
            data = {}
            for feature in features:
                while True:
                    try:
                        val = float(input(f"- {feature}: "))
                        data[feature] = val
                        break
                    except ValueError:
                        print("Please enter a valid number")
            
            # Make prediction
            df = pd.DataFrame([data])
            pred = model.predict(df)[0]
            pct = max(0.0, min(1.0, float(pred))) * 100
            
            # Show result and suggestions
            print(f"\n=== {company} Readiness: {pct:.2f}% ===")
            
            # Simple suggestions based on score
            if pct < 30:
                print("\nAreas to improve:")
                print("- Focus on core DSA and problem-solving")
                print("- Work on personal projects")
            elif pct < 60:
                print("\nYou're on the right track!")
                print("- Practice more coding problems")
                print("- Consider an internship")
            else:
                print("\nGreat job! Keep it up!")
                print("- Prepare for behavioral interviews")
                print("- Practice system design")
            
            print("\n" + "="*30)
            
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")


def main():
    parser = argparse.ArgumentParser(description='Predict company readiness from saved models.')
    parser.add_argument('--company', help='Company name (optional, for non-interactive mode)', 
                       choices=['Google','Microsoft','Amazon','Infosys'])
    parser.add_argument('--models_dir', default='models', help='Directory containing trained models')
    
    # Add feature arguments for non-interactive mode
    all_features = set(f for feats in FEATURES.values() for f in feats)
    for f in all_features:
        parser.add_argument(f"--{f.replace(' ', '_')}", type=float, help=f'Value for {f}')
    
    args = parser.parse_args()
    
    # Run in interactive mode if no company specified
    if not args.company:
        interactive_mode(args.models_dir)
        return
    
    # Non-interactive mode
    try:
        model, feats = load_model(args.models_dir, args.company)
        data = {}
        missing = []
        for f in feats:
            key = f.replace(' ', '_')
            val = getattr(args, key)
            if val is None:
                missing.append(f)
            else:
                data[f] = val

        if missing:
            raise SystemExit(f"Missing required features for {args.company}: {', '.join(missing)}")

        df = pd.DataFrame([data])
        pred = model.predict(df)[0]
        pct = max(0.0, min(1.0, float(pred))) * 100
        print(f"Predicted {args.company} Readiness: {pct:.2f}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTry running without arguments for interactive mode:")
        print("  python -m src.mnc_probability_analyzer.cli")
        print("\nOr provide all required features for non-interactive mode:")
        print(f"  python -m src.mnc_probability_analyzer.cli --company {args.company} \
--CGPA 8.5 --Total_Problems_Solved 200 ...")


if __name__ == '__main__':
    main()
