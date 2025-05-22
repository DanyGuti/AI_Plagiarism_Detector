'''Main module for the project'''
# from preprocessing import process_all_files
import os
from preprocessing.dataset_processor import create_dictionary_from_files,\
    create_data_split_from_dict

if __name__=="__main__":
    print("Main file")
    # Construct the source and destination paths with current working directory
    # source_path = os.path.join(os.getcwd(), "..", "data", "cases")
    # output_path = os.path.join(os.getcwd(), "..", "ast_data")
    # process_all_files(source_path, output_path)
    # Make all trees as dictionaries
    # {plag-case01-T01: matrix} txt files:
    # train/case-01/plag-T01.txt
    # train/case-01/non-plag-T01.txt
    # val/case-01/plag-T01.txt
    # val/case-01/non-plag-T01.txt

    # grab all non-plagiarized and plagiarized files
    # pass through the generation of the feature matrix
    create_data_split_from_dict(create_dictionary_from_files(
        os.path.join(
            os.getcwd(),
            "..",
            "ast_data",
        )
    ))
