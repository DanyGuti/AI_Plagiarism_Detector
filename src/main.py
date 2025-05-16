'''Main module for the project'''
from preprocessing import process_case_folder

if __name__=="__main__":
    print("Main file")
    source_path = "data/cases"
    output_path = "ast_data"
    process_case_folder(source_path, output_path)
