'''Main module for the project'''
from preprocessing import process_case_folder
import os

if __name__=="__main__":
    print("Main file")
    # Construct the source and destination paths with current working directory
    source_path = os.path.join(os.getcwd(), "..", "data", "cases")
    output_path = os.path.join(os.getcwd(), "..", "ast_data")
    process_case_folder(source_path, output_path)
